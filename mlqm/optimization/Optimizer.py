import torch
import numpy

import logging
# Set up logging:
logger = logging.getLogger()

class Optimizer(object):

    def __init__(self,delta,eps,npt):
        self.eps=eps
        self.delta=delta
        self.npt=npt

    def par_dist(self, dp_i, S_ij):
        D_ij = S_ij * (dp_i * dp_i.T)
        dist = torch.sum(D_ij.flatten())
        return dist

    def sr(self,energy,dpsi_i,dpsi_i_EL,dpsi_ij):
        f_i= (self.delta * ( dpsi_i * energy - dpsi_i_EL )).double()
        S_ij = dpsi_ij - dpsi_i * dpsi_i.T
        i = 0
        while True:
            S_ij_d = torch.clone(torch.detach(S_ij)).double()
            S_ij_d += 2**i * self.eps * torch.eye(self.npt)
            i += 1
            det_test = torch.det(S_ij_d)
            torch.set_printoptions(precision=8)
            try:
               U_ij = torch.cholesky(S_ij_d, upper=True, out=None)
               positive_definite = True
            except RuntimeError:
               positive_definite = False
               logger.error("Warning, Cholesky did not find a positive definite matrix")
            if (positive_definite):
               dp_i = torch.cholesky_solve(f_i, U_ij, upper=True, out=None)
               dp_0 = 1. - self.delta * energy - torch.sum(dpsi_i*dp_i)
               dp_i = dp_i / dp_0
               dist = self.par_dist(dp_i, S_ij)
               dist_reg = self.par_dist(dp_i, S_ij_d)
               dist_norm = self.par_dist(dp_i, dpsi_i * dpsi_i.T)
               torch.set_printoptions(precision=8)
               logger.debug(f"dist param = {dist.data}")
               logger.debug(f"dist reg = {dist_reg.data}")
               logger.debug(f"dist norm = {dist_norm.data}")
               dp_i = dp_i.float()
               if (dist < 0.001 and dist_norm < 0.2):
                  break
        return dp_i
