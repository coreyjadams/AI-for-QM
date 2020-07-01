import torch
import numpy as np

from mlqm import H_BAR


class HarmonicOscillator_mc(object):
    """Harmonic Oscillator Potential
    
    Implementation of the quantum harmonic oscillator hamiltonian
    """

    def __init__(self, mass : float, hbar : float, nwalk : int, ndim : int, npart : int):

        object.__init__(self)

        self.ndim = ndim
        if self.ndim < 1 or self.ndim > 3: 
            raise Exception("Dimension must be 1, 2, or 3")

        self.mass = mass
        self.hbar = hbar
        self.hbar2m = self.hbar**2 / self.mass / 2.
        self.nwalk = nwalk
        self.npart = npart

        if (self.npart == 2):
           self.alpha = 1.
        elif (self.npart > 2):
           self.alpha = -1.

        # Several objects get stored for referencing, if needed, after energy computation:
        self.pe = None
        self.ke = None
        self.ke_jf= None

        self.nrho = 100
        self.rho_min = 0.
        self.rho_max= 5.
        self.r_rho = torch.linspace(self.rho_min, self.rho_max, steps=self.nrho+1)
        self.v_rho = torch.zeros(size = [self.nrho])
        for i in range (self.nrho):
            self.v_rho[i] = self.r_rho[i+1]**3 - self.r_rho[i]**3
        self.v_rho = 4./3. * np.pi * self.v_rho
#        print("self.r_rho", self.r_rho)
#        print("self.v_rho", self.v_rho)
#        exit()

    def potential_energy(self, potential, inputs):
        "Returns potential energy"
        
        v_ij = torch.zeros(size=[self.nwalk,6])
        gr3b = torch.zeros(size=[self.nwalk,self.npart])
        V_ijk = torch.zeros(size=[self.nwalk])
        for i in range (self.npart-1):
            for j in range (i+1,self.npart):
                x_ij = inputs[:,i,:]-inputs[:,j,:]
                r_ij = torch.sqrt(torch.sum(x_ij**2,dim=1))
                v_ij += potential.pionless_2b(r_ij)
                if (self.npart > 2 ):
                   t_ij = potential.pionless_3b(r_ij)
                   gr3b[:,i] += t_ij
                   gr3b[:,j] += t_ij
                   V_ijk -= t_ij**2
        V_ijk += 0.5 * torch.sum(gr3b**2, dim = 1)
        self.pe = v_ij[:,0] + self.alpha * v_ij[:,2] + V_ijk

        return self.pe

    def kinetic_energy(self, wavefunction, inputs):
        "Returns kinetic energy"
        inputs.requires_grad_(True)
        wavefunction.zero_grad()
        log_psi = wavefunction(inputs)
        vt = torch.ones(size=[self.nwalk])
        
        d_log_psi = torch.autograd.grad(log_psi, inputs, vt, create_graph=True, retain_graph=True)[0]
        d_psi = d_log_psi
        self.ke_jf = torch.sum(d_psi**2, (1,2))
        self.ke_jf = self.ke_jf * self.hbar2m
        
        self.ke = 0
        for i in range (self.ndim):
            for j in range (self.npart):
                d_log_psi_ij = d_log_psi[:,j,i]
                d2_log_psi = torch.autograd.grad(d_log_psi_ij, inputs, vt, retain_graph=True)[0]
                d2_log_psi_ii_jj = d2_log_psi[:,j,i]
                d2_psi_ij = d2_log_psi_ii_jj + d_log_psi_ij**2
                self.ke -= d2_psi_ij
        self.ke = self.ke * self.hbar2m
        inputs.requires_grad_(False)
        return self.ke, self.ke_jf

    def energy(self, wavefunction, potential, inputs):
        """Computes the expectation value of energy of the supplied wavefunction.
        
        Computes the integral of the wavefunction in this potential
        
        Arguments:
            wavefunction {Wavefunction model} -- Callable wavefunction object
            inputs {torch.Tensor} -- Tensor of shape [N, dimension], must have graph enabled
        
        Returns:
            torch.tensor - Energy of shape [1]
        """

        self.nwalk = inputs.size()[0]
      
        ke, ke_jf = self.kinetic_energy(wavefunction,inputs)
        pe= self.potential_energy(potential, inputs)
        
        energy_jf = ke_jf + pe

        energy = ke + pe

 
        return energy, energy_jf


    def density(self, inputs):
        """Computes the expectation value of the single-nucleon density
        """
        
        mean = torch.mean(inputs, dim=1)
        xinputs = inputs - mean[:,None,:]
        rinputs = torch.sqrt(torch.sum(xinputs**2,dim=2))
        rho = torch.histc(rinputs, bins=self.nrho, min=self.rho_min, max=self.rho_max) 
        rho = rho / self.v_rho
        return rho

    def density_print(self, rho, error_rho):

        print("rho")
        for i in range (self.nrho):
            print((self.r_rho[i].item()+self.r_rho[i+1].item())/2.,rho[i].item(),error_rho[i].item())
        return
        







