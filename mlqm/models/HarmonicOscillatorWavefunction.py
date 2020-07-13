from jax import numpy, jit
import jax



"""Implememtation of the harmonic oscillator wave funtions

Create a polynomial, up to `degree` in every dimension `n`, that is the
exact solution to the harmonic oscillator wave function.

"""

def create_HarmonicOscillatorState(n : int, nparticles : int, degree : int, alpha : float):
    """Initializer

    Create a harmonic oscillator wave function

    Arguments:
        n {int} -- Dimension of the oscillator (1 <= n <= 3)
        nparticles {int} -- Number of particles
        degree {int} -- Degree of the solution (broadcastable to n)
        alpha {float} -- Alpha parameter (m * omega / hbar)

    Raises:
        Exception -- [description]
    """

    if n < 1 or n > 3:
        raise Exception("Dimension must be 1, 2, or 3 for HarmonicOscillatorWavefunction")

    if nparticles > 1:
        raise Exception("HarmonicOscillatorWavefunction is only for 1 particle for testing.")

    # Use numpy to broadcast to the right dimension:
    degree = numpy.asarray(degree, dtype=numpy.int32)
    degree = numpy.broadcast_to(degree, (n,))

    # Degree of the polynomial:
    if numpy.min(degree) < 0 or numpy.max(degree) > 4:
        raise Exception("Only the first 5 hermite polynomials are supported")

    alpha = numpy.asarray(alpha, dtype=numpy.int32)
    alpha = numpy.broadcast_to(alpha, (n,))

    # Normalization:
    norm = numpy.power(alpha / numpy.pi, 0.25)
    norm = numpy.prod(norm)


    # Craft the polynomial coefficients:

    # Add one to the degree since they start at "0"
    # Polynomial is of shape [degree, largest_dimension]
    polynomial = numpy.zeros(shape=(max(degree) + 1, n))
    #  Loop over the coefficents and set them:

    # Loop over dimension:
    polynomial_norm = numpy.zeros(shape=(n,))
    for _n in range(n):
        # Loop over degree:
        _d = degree[_n]
        if _d == 0:
            jax.ops.index_update(polynomial, jax.ops.index[0,_n], 1.0)
        elif _d == 1:
            jax.ops.index_update(polynomial, jax.ops.index[1,_n], 2.0)
        elif _d == 2:
            jax.ops.index_update(polynomial, jax.ops.index[0,_n], -2.0)
            jax.ops.index_update(polynomial, jax.ops.index[2,_n], 4.0)
        elif _d == 3:
            jax.ops.index_update(polynomial, jax.ops.index[1,_n], -12.0)
            jax.ops.index_update(polynomial, jax.ops.index[3,_n], 8.0)
        elif _d == 4:
            jax.ops.index_update(polynomial, jax.ops.index[0,_n], 12.0)
            jax.ops.index_update(polynomial, jax.ops.index[2,_n], -48.0)
            jax.ops.index_update(polynomial, jax.ops.index[4,_n], 16.0)

        # Set the polynomial normalization as a function of the degree
        # For each dimension:
        jax.ops.index_update(polynomial_norm, jax.ops.index[_n],
            1.0 / numpy.sqrt(2**_d * numpy.prod(numpy.arange(_d) + 1.)))


    exp = numpy.sqrt(alpha)
    
    state = {
        'n'             : n,
        'degree'        : degree,
        'exponent'      : exp,
        'normalization' : norm,
        'polynomial'    : polynomial,
        'poly_norm'     : polynomial_norm,

    }

    return state

def HarmonicOscillatorWavefunction(inputs, harmonic_oscillator_state):

    # Slice the inputs to restrict to just one particle:
    # print(inputs.shape)
    y = numpy.reshape(inputs, [-1,1])
    # jax.ops.index_update(polynomial, jax.ops.index[0,_n], 12.0)

    # Create the output tensor with the right shape, plus the constant term:
    polynomial_result = numpy.zeros(y.shape)

    # This is a somewhat basic implementation:
    # Loop over degree:
    for d in range(max(harmonic_oscillator_state['degree']) + 1):
        # Loop over dimension:

        # This is raising every element in the input to the d power (current degree)
        # This gets reduced by summing over all degrees for a fixed dimenions
        # Then they are reduced by multiplying over dimensions
        poly_term = y**d


        # Multiply every element (which is the dth power) by the appropriate
        # coefficient in it's dimension
        res_vec = poly_term * harmonic_oscillator_state['polynomial'][d]

        # Add this to the result:
        polynomial_result += res_vec


    # Multiply the results across dimensions at every point:
    polynomial_result = numpy.prod(polynomial_result, axis=1)

    # restrict the BC to just one particle:
    exponent_term = numpy.sum((harmonic_oscillator_state['exponent'] * y)**2, axis=-1)
    boundary_condition = numpy.exp(- (exponent_term) / 2.)

    total_normalization = harmonic_oscillator_state['normalization'] * numpy.prod(harmonic_oscillator_state['poly_norm'])
    epsilon = 1e-16
    # Add epsilon here to prevent underflow
    wavefunction = numpy.log(boundary_condition * polynomial_result * total_normalization + epsilon)


    return wavefunction
