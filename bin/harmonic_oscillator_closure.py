import sys, os
import pathlib
import configparser
import time

import argparse

from jax import numpy, jit, random

import logging
logger = logging.getLogger()

# Create a handler for STDOUT:
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

'''
This script runs the nieve, lattice-based optimization.

'''


################

# Add the local folder to the import path:
mlqm_dir = os.path.dirname(os.path.abspath(__file__))
mlqm_dir = os.path.dirname(mlqm_dir)
sys.path.insert(0,mlqm_dir)
from mlqm.samplers     import Estimator


class exec(object):

    def __init__(self):

        # This technique is taken from: https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
        parser = argparse.ArgumentParser(
            description='Run ML Based QMC Calculations',
            usage='python exec.py config.ini [<args>]')

        parser.add_argument("-c", "--config-file",
            type        = pathlib.Path,
            required    = True,
            help        = "Python configuration file to describe this run.")

        self.args = parser.parse_args()


        # Open the config file:
        self.config = configparser.ConfigParser()
        self.config.read(self.args.config_file)

        self.dimension  = int(self.config['General']['dimension'])
        self.nparticles = int(self.config['General']['nparticles'])
        self.degree     = int(self.config['General']['degree'])

        self.key = random.PRNGKey(int(self.config['General']['seed']))

        self.build_sampler()
        self.build_hamiltonian()

    def build_sampler(self):

        # First, check the sampler has all required config items:
        sampler = self.config["Sampler"]

        required_keys = ["nwalkers"]

        for key in required_keys:
            if key not in sampler:
                raise Exception(f"Configuration for Sampler missing key {key}")

        self.nwalkers   = int(sampler['nwalkers'])


        from mlqm.samplers import MetropolisSampler

        self.sampler = MetropolisSampler(
            n           = self.dimension,
            nparticles  = self.nparticles,
            nwalkers    = self.nwalkers,
            initializer = random.normal,
            init_params = {"mean": 0.0, "stddev" : 0.2},
            rng_key     = self.key)

        return

    def build_hamiltonian(self):

        from mlqm.hamiltonians import HarmonicOscillatorEnergy

        self.m      = float(self.config["Hamiltonian"]["mass"]),
        self.omega  = float(self.config["Hamiltonian"]["omega"])

        # Capture the function
        self.hamiltonian = HarmonicOscillatorEnergy

    def run(self):
        print("running")
        x = self.sampler.sample()

        # For each dimension, randomly pick a degree
        degree = [ 1 for d in range(self.dimension)]

        from mlqm.models import HarmonicOscillatorWavefunction, create_HarmonicOscillatorState
        self.wavefunction_state = create_HarmonicOscillatorState(
            n          = self.dimension,
            nparticles = self.nparticles,
            degree     = self.degree,
            alpha      = 1)

        self.wavefunction = HarmonicOscillatorWavefunction

        from mlqm.optimization import Optimizer
        from mlqm.samplers     import Estimator

        optimization = self.config['Optimization']

        sr_parameters = ["neq","nav","nprop","nvoid","nblock"]
        for key in sr_parameters:
            if key not in optimization:
                raise Exception(f"Configuration for Optimization missing key {key}")


        # Run the wave function once to initialize all it's weights
        _ = self.wavefunction(x, self.wavefunction_state)


        energy, energy_by_parts = self.hamiltonian(
            self.wavefunction, self.wavefunction_state, x, self.m, self.omega)

        for i in range(int(optimization['iterations'])):
            start = time.time()

            energy, error, energy_jf, error_jf, acceptance = self.walk(
                int(optimization['neq']),
                int(optimization['nav']),
                int(optimization['nprop']),
                int(optimization['nvoid']))

            # Update the parameters:
            for i_param in range(len(self.wavefunction.trainable_variables)):
                self.wavefunction.trainable_variables[i_param].assign_add(delta_p[i_param])

            if i % 1 == 0:
                logger.info(f"step = {i}, energy = {energy.numpy():.3f}, err = {error.numpy():.3f}")
                logger.info(f"step = {i}, energy_jf = {energy_jf.numpy():.3f}, err = {error_jf.numpy():.3f}")
                logger.info(f"acc  = {acceptance.numpy():.3f}")
                logger.info(f"time = {time.time() - start:.3f}")


    def walk(self, neq, nav, nprop, nvoid):
        nblock = neq + nav
        nstep = nprop * nvoid
        block_estimator = Estimator(info=None)
        block_estimator.reset()
        total_estimator = Estimator(info=None)
        total_estimator.reset()

        # Metropolis sampler will sample configurations uniformly:
        # Sample initial configurations uniformy between -sig and sig
        x_original = self.sampler.sample()

        kicker = random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.2}


        # This function operates with the following loops:
        #  - There are nblock "blocks" == neq + nav
        #    - the first 'neq' are not used for gradient calculations
        #    - the next 'nav' are used for accumulating wavefunction measurements
        #  - In each block, there are nsteps = nprop * nvoid
        #    - The walkers are kicked at every step.
        #    - if the step is a multiple of nvoid, AND the block is bigger than neq, the calculations are done.

        for i_block in range(nblock):
            block_estimator.reset()
            if (i_block == neq) :
               total_estimator.reset()

            for i_prop_step in range(nprop):
                # Here, we update the position of each particle for SR:
                acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=nvoid)
                x_current = self.sampler.sample()


                # Compute energy and accumulate estimators within a given block
                if i_block >= neq :
                    energy, energy_jf = self.hamiltonian.energy(self.wavefunction, x_current)
                    energy = energy / self.nwalkers
                    energy_jf = energy_jf / self.nwalkers


                    block_estimator.accumulate(
                        numpy.sum(energy),
                        numpy.sum(energy_jf),
                        acceptance,
                        1.,
                        0., # dpsi_i,
                        0., # dpsi_i_EL,
                        0., # dpsi_ij,
                        1.)
    # Accumulate block averages
            if ( i_block >= neq ):
                total_estimator.accumulate(
                    block_estimator.energy,
                    block_estimator.energy_jf,
                    block_estimator.acceptance,
                    0,
                    0., # block_estimator.dpsi_i,
                    0., # block_estimator.dpsi_i_EL,
                    0., # block_estimator.dpsi_ij,
                    block_estimator.weight)

        error, error_jf = total_estimator.finalize(nav)
        energy = total_estimator.energy
        energy_jf = total_estimator.energy_jf
        acceptance = total_estimator.acceptance



        return energy, error, energy_jf, error_jf, acceptance




if __name__ == "__main__":
    e = exec()
    e.run()
