import pytest
from jax import numpy


from .MetropolisSampler import MetropolisSampler

from ..models import HarmonicOscillatorWavefunction


@pytest.mark.parametrize("dimension",  [1,2,3])
@pytest.mark.parametrize("nparticles", [1,2,3])
def test_create_metropolis_sampler(dimension, nparticles, nwalkers = 10):

    # # Limit delta to 1/10 resolution in tests
    # ndim        : int, 
    # nwalkers    : int, 
    # nparticles  : int, 
    # initializer : callable, 
    # init_params : iter 

    sampler = MetropolisSampler(dimension, 
        nwalkers = nwalkers, 
        nparticles = nparticles,
        initializer = tf.random.normal, 
        init_params = {"mean": 0.0, "stddev" : 0.2})

    a = sampler.sample()

    assert True


@pytest.mark.parametrize("dimension",  [1,])
@pytest.mark.parametrize("nparticles", [1,])
def test_kick_metropolis_sampler(dimension, nparticles, nwalkers = 10):

    assert False

    # model = HarmonicOscillatorWavefunction()

    sampler = MetropolisSampler(dimension, 
        nwalkers = nwalkers, 
        nparticles = nparticles,
        initializer = tf.random.normal, 
        init_params = {"mean": 0.0, "stddev" : 0.2})

    kicker = tf.random.normal
    kicker_params = {"mean" : 0.0, "stddev" : 0.5}

    sampler.kick()

    a = sampler.sample()
