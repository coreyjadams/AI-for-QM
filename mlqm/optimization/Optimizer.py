import torch
import numpy

import logging
# Set up logging:
logger = logging.getLogger()

class Optimizer(object):

    def __init__(self, delta, eps, npt, indeces_flat):
        self.eps = eps
        self.delta = delta
        self.npt = npt
        self.indeces_flat = indeces_flat
        self.vt = 0.
        self.vtm1 = 0.
        self.gamma = 0.5
        self.eta = 0.5

    def par_dist(self, dp_i, S_ij):
        D_ij = S_ij * (dp_i * dp_i.T)
        dist = torch.sum(D_ij.flatten())
        return dist

    def sr(self, energy, dpsi_i, dpsi_i_EL, dpsi_ij, x_s, x_a, hamiltonian, potential, wavefunction):
        f_i = dpsi_i * energy - dpsi_i_EL
        S_ij = dpsi_ij - dpsi_i * dpsi_i.T
        i = 0
        energy_d_min=100.
        for n in range (20):
            S_ij_d = torch.clone(torch.detach(S_ij))
            S_ij_d += 2**i * self.eps * torch.eye(self.npt)
##            S_ij_d += self.eps * torch.eye(self.npt)
            i += 1
            try:
               U_ij = torch.cholesky(S_ij_d, upper=True, out=None)
               positive_definite = True
            except RuntimeError:
               positive_definite = False
               logger.error("Warning, Cholesky did not find a positive definite matrix")
            if (positive_definite):
               dp_i = torch.cholesky_solve(f_i, U_ij, upper=True, out=None)
               dp_i = self.delta * dp_i #/ 2**i
               dp_0 = 1. - self.delta * energy - torch.sum(dpsi_i * dp_i)

               print("self.delta * energy", self.delta * energy)
               print("torch.sum(dpsi_i * dp_i)", torch.sum(dpsi_i * dp_i))
               print("sandro", 1. - self.delta * energy - torch.sum(dpsi_i * dp_i))

#               dp_i =  ( dp_i / dp_0 )

# update with momentum      
               dp_i = self.gamma * self.vtm1 + self.eta * ( dp_i / dp_0 )

               dist = self.par_dist(dp_i, S_ij)
               dist_reg = self.par_dist(dp_i, S_ij_d)
               dist_norm = self.par_dist(dp_i, dpsi_i * dpsi_i.T)
               delta_p = wavefunction.recover_flattened(dp_i, self.indeces_flat, wavefunction)
               delta_p = [ delta_p_i for delta_p_i in delta_p]

               psi_d, psi_d_err, energy_d_a, energy_d_err_a = self.par_dist_full(delta_p, x_a, hamiltonian, potential, wavefunction )
               torch.set_printoptions(precision=6)
               logger.debug(f"dist full x_a = {psi_d.data:.8f}, err = {psi_d_err.data:.8f}")
               logger.debug(f"energy diff x_a = {energy_d_a.data:.6f}, err = {energy_d_err_a.data:.6f}")
               psi_d, psi_d_err, energy_d, energy_d_err = self.par_dist_full(delta_p, x_s, hamiltonian, potential, wavefunction )
               torch.set_printoptions(precision=6)
               logger.debug(f"dist param = {dist.data:.8f}")
               logger.debug(f"dist reg = {dist_reg.data:.6f}")
               logger.debug(f"dist norm = {dist_norm.data:.6f}")
               logger.debug(f"dist full = {psi_d.data:.8f}, err = {psi_d_err.data:.8f}")
               logger.debug(f"energy diff = {energy_d.data:.6f}, err = {energy_d_err.data:.6f}")
               if ( dist < 0.1 and psi_d > 0.9 and energy_d_a < energy_d_min):
                  energy_d_min = energy_d_a
                  energy_d_err_min = energy_d_err_a
                  self.vt = dp_i
                  delta_p_min = delta_p
			
        logger.debug(f"energy diff min = {energy_d_min.data:.6f}, err = {energy_d_err_min.data:.6f}")
        self.vtm1 = self.vt
        return delta_p_min


    def par_dist_full(self, delta_p, x_s, hamiltonian, potential, wavefunction):
        log_wpsi_o = wavefunction(x_s)
        energy_o, energy_jf_o = hamiltonian.energy(wavefunction, potential, x_s)
        energy_o_sum = torch.mean(energy_o) 
        energy2_o_sum = torch.mean(energy_o**2)
        energy_o_err=torch.sqrt((energy2_o_sum - energy_o_sum**2) / x_s.shape[0])
        logger.info(f"energy_old = {energy_o_sum.data:.3f}, err = {energy_o_err.data:.6f}")

# Set new parameters
        for (p, dp) in zip (wavefunction.parameters(),delta_p):
            p.data = p.data + dp
        log_wpsi_n = wavefunction(x_s)

        psi_norm = torch.exp( ( log_wpsi_n - log_wpsi_o) )
        psi2_norm = psi_norm**2

#        psi_norm = abs(psi_n / psi_o)**2
        psi2_norm_sum = torch.mean(psi2_norm)
        energy_n, energy_jf_n = hamiltonian.energy(wavefunction, potential, x_s)
        energy_n *= psi2_norm
        energy_n_sum = torch.mean(energy_n) / psi2_norm_sum
        energy2_n_sum = torch.mean(energy_n**2) / psi2_norm_sum
        energy_n_err=torch.sqrt((energy2_n_sum - energy_n_sum**2) / x_s.shape[0] )
        logger.info(f"energy_new = {energy_n_sum.data:.3f}, err = {energy_n_err.data:.6f}")
# Correlated energy difference
        energy_d = energy_n / psi2_norm_sum - energy_o
        energy_d_sum= torch.mean(energy_d)
        energy2_d_sum= torch.mean(energy_d**2)
        energy_d_err=torch.sqrt((energy2_d_sum - energy_d_sum**2) / x_s.shape[0] )
# Correlated wave function difference
# Here sqrt(psi_norm_sum) denotes sqrt(<psi_p' | psi_p'>) = sqrt ( int dx |psi_p'(x)|^2) = sqrt ( int dx |psi_p(x)|^2 ( |psi_p'(x)|^2 / |psi_p(x)|^2) )
#        psi_d = (abs(psi_n/torch.sqrt(psi2_norm_sum) - psi_o)/abs(psi_o))**2
#        psi_d_sum= torch.mean(psi_d)
#        psi2_d_sum= torch.mean(psi_d**2)
#        psi_d_err=torch.sqrt((psi2_d_sum - psi_d_sum**2) / x_s.shape[0] )

        psi_d_sum = torch.mean(psi_norm)**2 / psi2_norm_sum
        psi_d_err = torch.sqrt( (psi2_norm_sum - torch.mean(psi_norm)**2) / x_s.shape[0])

        
# Set back old parameters
        for (p, dp) in zip (wavefunction.parameters(),delta_p):
            p.data = p.data - dp
        return psi_d_sum, psi_d_err, energy_d_sum, energy_d_err

