from jax import numpy

class Estimator(object):
    """ Accumulate block and totalk averages and errors
    """
    def __init__(self,*,info=None):
        if info is not None:
            print(f"Set the following estimators: E, E2,E_jf,E2_jf,acc,weight,Psi_i,H*Psi_i,Psi_ij ")

    def reset(self):
        self.energy = 0
        self.energy2 = 0
        self.energy_jf = 0
        self.energy2_jf = 0
        self.acceptance = 0
        self.weight = 0
        self.dpsi_i = 0
        self.dpsi_i_EL = 0
        self.dpsi_ij = 0

    def accumulate(self,energy,energy_jf,acceptance,weight,dpsi_i,dpsi_i_EL,dpsi_ij,estim_wgt) :
        self.energy += energy/estim_wgt
        self.energy2 += (energy/estim_wgt)**2
        self.energy_jf += energy_jf/estim_wgt
        self.energy2_jf += (energy_jf/estim_wgt)**2
        self.acceptance += acceptance/estim_wgt
        self.weight += weight/estim_wgt
        self.dpsi_i += dpsi_i/estim_wgt
        self.dpsi_i_EL += dpsi_i_EL/estim_wgt
        self.dpsi_ij += dpsi_ij/estim_wgt

    def finalize(self,nav):
        self.energy /= nav
        self.energy2 /= nav
        self.energy_jf /= nav
        self.energy2_jf /= nav
        self.acceptance /= nav
        self.dpsi_i /= nav
        self.dpsi_i_EL /= nav
        self.dpsi_ij /= nav
        error= numpy.sqrt((self.energy2 - self.energy**2) / (nav-1))
        error_jf = numpy.sqrt((self.energy2_jf - self.energy_jf**2) / (nav-1))
        return error, error_jf
