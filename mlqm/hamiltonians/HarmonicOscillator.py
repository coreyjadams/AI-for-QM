from jax import numpy, grad, vmap, jit

import logging
logger = logging.getLogger()

from mlqm import H_BAR

@jit
def potential_energy(inputs, M, omega):
    """Return potential energy

    Calculate and return the PE.

    Arguments:
        inputs {numpy.ndarray} -- Tensor of shape [N, dimension], must have graph enabled
    Returns:
        numpy.ndarray - potential energy of shape [1]
    """

    # Potential calculation
    # < x | H | psi > / < x | psi > = < x | 1/2 w * x**2  | psi > / < x | psi >  = 1/2 w * x**2
    # x Squared needs to contract over spatial dimensions:
    x_squared = numpy.sum(inputs**2, axis=(2))
    pe = (0.5 * M * omega**2 ) * x_squared
    return pe


   

@jit
def kinetic_energy_jf(dlogw_dx, M):
    """Return Kinetic energy

    Calculate and return the KE directly

    Otherwise, exception

    Arguments:
        logw_of_x {numpy.ndarray} -- Computed derivative of the wavefunction

    Returns:
        numpy.ndarray - potential energy of shape [1]
    """
    # < x | KE | psi > / < x | psi > =  1 / 2m [ < x | p | psi > / < x | psi >  = 1/2 w * x**2

    # Contract d2_w_dx over spatial dimensions:
    ke_jf = (H_BAR**2 / (2 * M)) * numpy.sum(dlogw_dx**2, axis=(1,2))
    return numpy.reshape(ke_jf, [-1, 1])


@jit
def kinetic_energy(KE_JF, d2logw_dx2, M):
    """Return Kinetic energy

    If the potential energy is already computed, and no arguments are supplied,
    return the cached value

    If all arguments are supplied, calculate and return the KE.

    Otherwise, exception

    Arguments:
        logw_of_x {numpy.ndarray} -- Computed derivative of the wavefunction

    Returns:
        numpy.ndarray - potential energy of shape [1]
    """
    ke = -(H_BAR**2 / (2 * M)) * numpy.sum(d2logw_dx2, axis=(1,2))
    return numpy.reshape(ke, [-1, 1])  - KE_JF

def HarmonicOscillatorEnergy(wavefunction, wavefunction_state, inputs, M, omega):


    """Compute the expectation valye of energy of the supplied wavefunction.

    Computes the integral of the wavefunction in this potential

    Arguments:
        wavefunction {Wavefunction model} -- Callable wavefunction object
        inputs {numpy.ndarray} -- Tensor of shape [N, dimension], must have graph enabled
        delta {numpy.ndarray} -- Integral Computation 'dx'

    Returns:
        numpy.ndarray - Energy of shape [1]
    """


    # This function takes the inputs
    # And computes the expectation value of the energy at each input point


    w_func = lambda inputs, state=wavefunction_state  : wavefunction(inputs, state)

    logw_of_x = wavefunction(inputs, wavefunction_state)

    print("HO inputs shape:", inputs.shape)
    print("HO logw_of_x.shape: ", logw_of_x.shape)

    # dlogw_dx_fn   = grad(w_func)
    # d2logw_dx2_fn = grad(dlogw_dx_fn)

    print("HO run derivative")
    # Get the derivative of logw_of_x with respect to inputs
    dlogw_dx   = vmap(grad(w_func), 0)(inputs)
    
    print("HO dlogw_dx.shape: ", dlogw_dx.shape)

    d2logw_dx2 = vmap(d2logw_dx2_fn)(inputs)
    # Get the derivative of dlogw_dx with respect to inputs (aka second derivative)

    print("HO d2logw_dx2.shape: ", d2logw_dx2.shape)

    # hessians = tf.hessians(logw_of_x, inputs)
    # d2_hessian = numpy.sum(hessians[0], axis=(1,2,4,5))
    # d2logw_dx2 = tf.linalg.diag_part(d2_hessian)


    # Potential energy depends only on the wavefunction
    pe = self.potential_energy(inputs=inputs, M = M, omega=omega)

    # KE by parts needs only one derivative
    ke_jf = self.kinetic_energy_jf(dlogw_dx=dlogw_dx, M=M)

    # True, directly, uses the second derivative
    ke_direct = self.kinetic_energy(KE_JF = ke_jf, d2logw_dx2 = d2logw_dx2, M=M)

    # Total energy computations:
    energy = tf.squeeze(pe + ke_direct)
    energy_jf = tf.squeeze(pe + ke_jf)

    return energy, energy_jf
