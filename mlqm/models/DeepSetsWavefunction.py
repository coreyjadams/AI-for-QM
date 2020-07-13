from jax import numpy, nn, random
from mlqm import DEFAULT_TENSOR_TYPE


def create_DeepSetsState(ndim : int, nparticles: int, rng_key : random.PRNGKey ):
    '''Deep Sets wavefunction for symmetric particle wavefunctions

    Implements a deep set network for multiple particles in the same system

    Arguments:
        ndim {int} -- Number of dimensions
        nparticles {int} -- Number of particls

    Raises:
        Exception -- [description]
    '''

    if ndim < 1 or ndim > 3:
       raise Exception("Dimension must be 1, 2, or 3 for DeepSetsWavefunction")

    state = {}

    state['ndim'] = ndim
    state['nparticles'] = nparticles

    # A model is a list of functions and associated weights.
    individual_net = []

    key, subkey = random.split(rng_key)
    individual_net.append(['dense', nn.initializers.glorot_uniform()(subkey, [ndim,32]) ])
    individual_net.append(['softplus', None])

    aggregate_net = []
    key, subkey = random.split(rng_key)
    aggregate_net.append(['dense', nn.initializers.glorot_uniform()(subkey, [32,32])])
    aggregate_net.append(['softplus', None])
    key, subkey = random.split(rng_key)
    aggregate_net.append(['dense', nn.initializers.glorot_uniform()(subkey, [32,1])])


    state['individual_net'] = individual_net
    state['aggregate_net']  = aggregate_net

    return state

def DeepSetsWavefunction(inputs, state):


    # Because of the way Jax vectorizes, we have to force the shape:
    inputs = numpy.reshape(inputs, [-1, state['nparticles'], state['ndim']])

    # Mean subtract for all particles:
    # if state['nparticles'] > 1:
    print("inputs.shape: ", inputs.shape)
    mean = numpy.mean(inputs, axis=1)
    print("mean.shape: ", mean.shape)
    mean = numpy.reshape(mean, [-1,1,state['ndim']])
    print("mean.shape: ", mean.shape)
    xinputs = inputs - mean
    print("xinputs.shape: ", xinputs.shape)
    # else:
    #     xinputs = inputs

    x = []
    # Individual net per particle:
    for p in range(state['nparticles']):
        _x = xinputs[:,p,:]
        for layer in state['individual_net']:
            if layer[0] == 'dense':
                _x = numpy.dot(_x, layer[1])
            elif layer[0] == 'softplus':
                _x = nn.softplus(_x)

        x.append(_x)



    # Symmetric function:
    x = numpy.sum(x, axis=0)


    # Aggregate net:
    for layer in state['aggregate_net']:
        if layer[0] == 'dense':
            x = numpy.dot(x, layer[1])
        elif layer[0] == 'softplus':
            x = nn.softplus(x)

    x = numpy.reshape(x, [-1])

    # Compute the initial boundary condition, which the network will slowly overcome
    # boundary_condition = tf.math.abs(self.normalization_weight * tf.reduce_sum(xinputs**self.normalization_exponent, axis=(1,2))
    boundary_condition = -0.1 * numpy.sum(xinputs**2, axis=(1,2))
    boundary_condition = numpy.reshape(boundary_condition, [-1])

    return x + boundary_condition
