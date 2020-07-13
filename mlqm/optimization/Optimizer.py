import tensorflow as tf
import numpy

import logging
# Set up logging:
logger = logging.getLogger()
#
# class Optimizer(object):
#
#     def __init__(self,delta : float ,eps : float, npt : int):
#         '''Create an optimizer to compute the gradients for the SR update.
#
#         Arguments:
#             delta {float} --
#             eps {float} --
#             npt {int} -- Number of parameters in the wave function
#         '''
#
#         self.eps    = tf.convert_to_tensor(eps, dtype=tf.float64)
#         self.delta  = tf.convert_to_tensor(delta, dtype=tf.float64)
#         self.npt    = npt
#
#         self.i      = tf.Variable(0., dtype=tf.float64)
#
#     @tf.function
#     def par_dist(self, dp_i, S_ij):
#
#         # This replaces the double for loop.  Don't do that in python.
#         D_ij = S_ij * (dp_i * tf.transpose(dp_i))
#         dist = tf.reduce_sum(D_ij)
#         return dist
#
#     @tf.function
#     def s_ij(self, Oi_Oj, Oi):
#         S_ij = Oi_Oj - Oi * tf.transpose(Oi)
#         return S_ij
#
#     @tf.function
#     def f_i(self, delta, energy, Oi, Oi_Energy):
#         F = delta*(Oi * energy  - Oi_Energy )
#         return F
#
#     @tf.function
#     def S_ij_to_regularized(self, S_ij, eps, npt, i):
#         S_ij_d = S_ij + tf.convert_to_tensor(2., dtype=tf.float64)**i * eps * tf.eye(npt, dtype=tf.float64)
#         return S_ij_d
#
#     @tf.function
#     def pd_check(self, S_ij, eps, npt, i):
#
#
#         positive_definite = False
#
#         while not positive_definite:
#             S_ij_d = self.S_ij_to_regularized(S_ij, eps, npt, i)
#             try:
#     ################################################################################
#                 # Note: TF doesn't support upper cholesky and cholesky_solve...
#                 U_ij = tf.linalg.cholesky(S_ij_d)
#                 # This returns the lower triangular matrix
#     ################################################################################
#                 positive_definite = True
#                 return U_ij, S_ij_d, i
#             except:
#                 positive_definite = False
#                 logger.warning(f"Warning, Cholesky did not find a positive definite matrix on attempt {i}")
#             i.assign_add(1.0)
#
#     # @tf.function
#     def sr(self,energy,dpsi_i,dpsi_i_EL,dpsi_ij):
#         '''
#         Perform the computation of weight updates.
#         Does a cholesky decomposition and solve
#         '''
#
#         # For this, we work in double precision.
#         # May revisit later ways to speed this up...
#
#         # First, cast everything to double precision:
#         energy      = tf.cast(energy,    tf.float64)
#         dpsi_i      = tf.cast(dpsi_i,    tf.float64)
#         dpsi_i_EL   = tf.cast(dpsi_i_EL, tf.float64)
#         dpsi_ij     = tf.cast(dpsi_ij,   tf.float64)
#
#         # Compute S_ij, F_i:
#         S_ij = self.s_ij(dpsi_ij, dpsi_i)
#         F_i = self.f_i(self.delta, energy, dpsi_i, dpsi_i_EL)
#
#         self.i.assign(0.0)
#
#         # Next, we need S_ij to be positive definite.
#         U_ij, S_ij_d, i = self.pd_check(S_ij, self.eps, self.npt, self.i)
#
#         # Now, it is definitely positive definite
#
#         # With U_ij, we can do the cholesky_solve:
#         dp_i, dist, dist_reg, dist_norm = self.solve_and_check(
#             U_ij,
#             F_i,
#             tf.cast(dpsi_i, tf.float64),
#             tf.cast(energy, tf.float64),
#             tf.cast(S_ij, tf.float64),
#             S_ij_d,
#             tf.cast(self.delta, tf.float64))
#
#         while dist >= 0.001 and dist_norm >= 0.2:
#             i.assign_add(1.0)
#
#             # If we're running again, we are too far from the linear approximation
#             # Adjust the regularized matrix and try again:
#             S_ij_d =  self.S_ij_to_regularized(S_ij, self.eps, self.npt, i)
#             # Re-factor the matrix:
#             U_ij   = tf.linalg.cholesky(S_ij_d)
#             dp_i, dist, dist_reg, dist_norm = self.solve_and_check(
#                 U_ij,
#                 F_i,
#                 tf.cast(dpsi_i, tf.float64),
#                 tf.cast(energy, tf.float64),
#                 tf.cast(S_ij, tf.float64),
#                 S_ij_d,
#                 tf.cast(self.delta, tf.float64))
#
#
#         return tf.cast(dp_i, tf.float32)
#
#     @tf.function
#     def solve_and_check(self, U_ij, F_i, dpsi_i, energy, S_ij, S_ij_d, delta):
#         '''
#         Use the cholesky decomposition to solve the linear equations, as well
#         as run distance checks to verify we are close to the linear approximation
#         '''
#         ################################################################################
#         dp_i = tf.linalg.cholesky_solve(U_ij, F_i)
#         ################################################################################
#
#         dp_0 = 1. - delta * energy - tf.reduce_sum(dpsi_i*dp_i)
#         dp_i = dp_i / dp_0
#
#         # Compute distances to measure the linear approximation:
#         dist      = self.par_dist(dp_i, S_ij)
#         dist_reg  = self.par_dist(dp_i, S_ij_d)
#         dist_norm = self.par_dist(dp_i, dpsi_i * tf.transpose(dpsi_i))
#
#         return dp_i, dist, dist_reg, dist_norm


class Optimizer(object):

    def __init__(self,delta,eps):
        self.eps    = tf.convert_to_tensor(eps, dtype=tf.float64)
        self.delta  = tf.convert_to_tensor(delta, dtype=tf.float64)
        # self.npt    = npt
#
    @tf.function
    def par_dist(self, dp_i, S_ij):
        D_ij = S_ij * (dp_i * tf.transpose(dp_i))
        dist = tf.reduce_sum(D_ij)
        return dist

    def sr(self,energy,dpsi_i,dpsi_i_EL,dpsi_ij):
        # f_i= tf.cast(tf.cast(self.delta, tf.float32) * ( dpsi_i * energy - dpsi_i_EL ), tf.float64)
        f_i= tf.cast(self.delta * ( dpsi_i * energy - dpsi_i_EL ), tf.float64)
        S_ij = dpsi_ij - dpsi_i * tf.transpose(dpsi_i)
        i = 0
        while True:
            S_ij_d = tf.cast(S_ij, tf.float64)
            S_ij_d += 2**i * self.eps * tf.eye(self.npt, dtype=tf.float64)
            i += 1
            try:
               U_ij = tf.linalg.cholesky(S_ij_d)

               positive_definite = True
            except:
               positive_definite = False
               logger.error(f"Warning, Cholesky did not find a positive definite matrix on attempt {i}")
            if (positive_definite):
               dp_i = tf.linalg.cholesky_solve(U_ij, f_i)
               dp_0 = 1. - self.delta * tf.cast(energy, tf.float64) - tf.reduce_sum(tf.cast(dpsi_i, tf.float64)*dp_i)
               dp_i = dp_i / dp_0
               dist = self.par_dist(dp_i, tf.cast(S_ij, tf.float64))
               dist_reg = self.par_dist(dp_i, S_ij_d)
               dist_norm = self.par_dist(dp_i, tf.cast(dpsi_i * tf.transpose(dpsi_i), tf.float64) )

               logger.debug(f"dist param = {dist}")
               logger.debug(f"dist reg = {dist_reg}")
               logger.debug(f"dist norm = {dist_norm}")
               # dp_i = tf.cast(dp_i, tf.float32)
               if (dist < 0.001 and dist_norm < 0.2):
                  break
        return dp_i
