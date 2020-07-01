import torch
import numpy

class Estimator(object):
    """ Accumulate block and totalk averages and errors
    """
    def __init__(self, nopt : int, info=None):
        if info is not None: 
            print(f"Set the following estimators: E, E2,E_jf,E2_jf,acc,weight,Psi_i,H*Psi_i,Psi_ij ")
        self.nopt = nopt

    def reset(self):
        self.energy_blk = 0
        self.energy_blk = 0
        self.energy2_blk = 0
        self.energy_jf_blk = 0
        self.energy2_jf_blk = 0
        self.rho_blk = 0
        self.rho2_blk = 0
        self.acceptance_blk = 0
        self.weight_blk = 0
        self.dpsi_i_blk = 0
        self.dpsi_i_EL_blk = 0
        self.dpsi_ij_blk = 0

        self.energy_tot = 0
        self.energy2_tot = 0
        self.energy_jf_tot = 0
        self.energy2_jf_tot = 0
        self.rho_tot = 0
        self.rho2_tot = 0
        self.acceptance_tot = 0
        self.weight_tot = 0
        self.dpsi_i_tot = 0
        self.dpsi_i_EL_tot = 0
        self.dpsi_ij_tot = 0
        
        self.nav = 0

    def addval(self,energy,energy_jf,acceptance,weight,dpsi_i,dpsi_i_EL,dpsi_ij,rho) :
        self.energy_blk += energy * weight
        self.energy2_blk += energy**2 * weight
        self.energy_jf_blk += energy_jf * weight 
        self.energy2_jf_blk += energy_jf**2 * weight
        self.rho_blk += rho * weight
        self.rho2_blk += rho**2 * weight
        self.acceptance_blk += acceptance * weight
        self.dpsi_i_blk += dpsi_i * weight 
        self.dpsi_i_EL_blk += dpsi_i_EL * weight 
        self.dpsi_ij_blk += dpsi_ij * weight
        self.weight_blk += weight

    def addblk(self) :
        self.energy_tot += self.energy_blk / self.weight_blk
        self.energy2_tot += (self.energy_blk / self.weight_blk)**2
        self.energy_jf_tot += self.energy_jf_blk / self.weight_blk
        self.energy2_jf_tot += (self.energy_jf_blk / self.weight_blk)**2

        if (self.nopt == 0):
            self.rho_tot += self.rho_blk / self.weight_blk
            self.rho2_tot += (self.rho_blk / self.weight_blk)**2

        self.acceptance_tot += self.acceptance_blk / self.weight_blk
        self.dpsi_i_tot += self.dpsi_i_blk / self.weight_blk
        self.dpsi_i_EL_tot += self.dpsi_i_EL_blk / self.weight_blk
        self.dpsi_ij_tot += self.dpsi_ij_blk / self.weight_blk

        self.energy_blk = 0
        self.energy2_blk = 0
        self.energy_jf_blk = 0
        self.energy2_jf_blk = 0
        self.rho_blk = 0
        self.rho2_blk = 0
        self.acceptance_blk = 0
        self.weight_blk = 0
        self.dpsi_i_blk = 0
        self.dpsi_i_EL_blk = 0
        self.dpsi_ij_blk = 0
        self.nav += 1

    def average(self):
        self.energy = self.energy_tot / self.nav
        self.energy2 = self.energy2_tot / self.nav
        self.energy_jf = self.energy_jf_tot / self.nav
        self.energy2_jf = self.energy2_jf_tot / self.nav
        self.rho = self.rho_tot / self.nav
        self.rho2 = self.rho2_tot / self.nav
        self.acceptance = self.acceptance_tot / self.nav
        self.dpsi_i = self.dpsi_i_tot / self.nav
        self.dpsi_i_EL = self.dpsi_i_EL_tot / self.nav
        self.dpsi_ij = self.dpsi_ij_tot / self.nav
        error = torch.sqrt((self.energy2 - self.energy**2) / (self.nav-1))
        error_jf = torch.sqrt((self.energy2_jf - self.energy_jf**2) / (self.nav-1))
#        print("self.rho2=", self.rho2)
#        print("self.rho**2=", self.rho**2) 
#        exit()
        if (self.nopt == 0):
            error_rho = torch.sqrt((self.rho2 - self.rho**2) / (self.nav-1))
        else:
            error_rho = 0
        return error, error_jf, error_rho



