import sys, os
import pathlib
import configparser
import time

import argparse

from jax import numpy, jit, random

import logging
logger = logging.getLogger()

# Create a handler for STDOUT, but only on the root rank:
# if not MPI_AVAILABLE or hvd.rank() == 0:
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Add the local folder to the import path:
mlqm_dir = os.path.dirname(os.path.abspath(__file__))
mlqm_dir = os.path.dirname(mlqm_dir)
sys.path.insert(0,mlqm_dir)
from mlqm.samplers     import Estimator
from mlqm.optimization import Optimizer
from mlqm import DEFAULT_TENSOR_TYPE


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
        #
        # if MPI_AVAILABLE:
        #     self.rank = hvd.rank()
        #     self.size = hvd.size()
        #     self.local_rank = hvd.local_rank()

        # Open the config file:
        self.config = configparser.ConfigParser()
        self.config.read(self.args.config_file)


        optimization = self.config['Optimization']
        sr_parameters = ["iterations","nequilibrations","naverages","nobservations","nvoid", "delta", "eps"]
        for key in sr_parameters:
            if key not in optimization:
                raise Exception(f"Configuration for Optimization missing key {key}")

        self.iterations = int(optimization['iterations'])
        self.nequilibrations = int(optimization['nequilibrations'])
        self.naverages       = int(optimization['naverages'])
        self.nobservations   = int(optimization['nobservations'])
        self.nvoid           = int(optimization['nvoid'])
        self.delta           = float(optimization['delta'])
        self.eps             = float(optimization['eps'])

        self.key = random.PRNGKey(int(self.config['General']['seed']))


        self.dimension  = int(self.config['General']['dimension'])
        self.nparticles = int(self.config['General']['nparticles'])
        self.save_path  = self.config["General"]["model_save_path"]


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

    def check_potential_parameters(self, potential, parameters, config):

        for key in parameters:
            if key not in config:
                raise Exception(f"Configuration for {potential} missing key {key}")

    def build_hamiltonian(self):

        # First, ask for the type of hamiltonian
        kind = self.config["Hamiltonian"]["form"]
        if kind == "HarmonicOscillator":

            from mlqm.hamiltonians import HarmonicOscillatorEnergy

            self.m      = float(self.config["Hamiltonian"]["mass"]),
            self.omega  = float(self.config["Hamiltonian"]["omega"])

            # Capture the function
            self.hamiltonian = HarmonicOscillatorEnergy

        elif kind == "AtomicPotential":
            from mlqm.hamiltonians import AtomicPotential

            required_keys = ["mass", "Z"]
            self.check_potential_parameters(kind, required_keys, self.config["Hamiltonian"])

            self.hamiltonian = AtomicPotential(
                mu          = float(self.config["Hamiltonian"]["mass"]),
                Z           = int(self.config["Hamiltonian"]["Z"]),
            )
        elif kind == "NuclearPotential":
            from mlqm.hamiltonians import NuclearPotential

            required_keys = ["mass", "Z"]
            self.check_potential_parameters(kind, required_keys, self.config["Hamiltonian"])


            self.hamiltonian = NuclearPotential(
                M           = float(self.config["Hamiltonian"]["mass"]),
                omega       = float(self.config["Hamiltonian"]["omega"]),
            )
        else:
            raise Exception(f"Unknown potential requested: {kind}")

    def run(self):

        x = self.sampler.sample()

        # For each dimension, randomly pick a degree
        degree = [ 1 for d in range(self.dimension)]


        from mlqm.models import create_DeepSetsState, DeepSetsWavefunction
        self.wavefunction_state = create_DeepSetsState(
            ndim       = self.dimension,
            nparticles = self.nparticles,
            rng_key    = self.key,
        )


        self.wavefunction = DeepSetsWavefunction

        # Run the wave function once to initialize all it's weights
        initial_wavefunction = self.wavefunction(x, self.wavefunction_state)

        print(initial_wavefunction.shape)

        # We attempt to restore the weights:
        try:
            self.wavefunction.load_weights(self.save_path)
            print("Loaded weights!")
        except:
            pass



        # if MPI_AVAILABLE:
        #     # We have to broadcast the wavefunction parameter here:
        #     hvd.broadcast_variables(self.wavefunction.variables, 0)
        #

        from mlqm.optimization import Optimizer
        from mlqm.samplers     import Estimator




        # Create an optimizer
        self.optimizer = Optimizer(
            delta   = self.delta,
            eps     = self.eps)

        # energy, energy_by_parts = self.hamiltonian.energy(self.wavefunction, x)
        energy, energy_by_parts = self.hamiltonian(
            self.wavefunction, self.wavefunction_state, x, self.m, self.omega)

        for i in range(self.iterations):
            start = time.time()

            energy, error, energy_jf, error_jf, acceptance, delta_p = self.sr_step(
                self.nequilibrations,
                self.naverages,
                self.nobservations,
                self.nvoid)

            # Update the parameters:
            for i_param in range(len(self.wavefunction.trainable_variables)):
                self.wavefunction.trainable_variables[i_param].assign_add(delta_p[i_param])

            if i % 1 == 0:
                logger.info(f"step = {i}, energy = {energy.numpy():.3f}, err = {error.numpy():.3f}")
                logger.info(f"step = {i}, energy_jf = {energy_jf.numpy():.3f}, err = {error_jf.numpy():.3f}")
                logger.info(f"acc  = {acceptance.numpy():.3f}")
                logger.info(f"time = {time.time() - start:.3f}")


    def jacobian(self, x_current, wavefunction):
        tape = tf.GradientTape()

        with tape:
            log_wpsi = wavefunction(x_current)

        jac = tape.jacobian(log_wpsi, wavefunction.trainable_variables)

        # Grab the original shapes ([1:] means everything except first dim):
        jac_shape = [j.shape[1:] for j in jac]
        # get the flattened shapes:
        flat_shape = [[-1, tf.reduce_prod(js)] for js in jac_shape]
        # Reshape the

        # We have the flat shapes and now we need to make the jacobian into a single matrix

        flattened_jacobian = [tf.reshape(j, f) for j, f in zip(jac, flat_shape)]

        flattened_jacobian = tf.concat(flattened_jacobian, axis=-1)

        return flattened_jacobian, flat_shape

    def compute_O_observables(self, flattened_jacobian, energy):

        # dspi_i is the reduction of the jacobian over all walkers.
        # In other words, it's the mean gradient of the parameters with respect to inputs.
        # This is effectively the measurement of O^i in the paper.
        dpsi_i = tf.reduce_mean(flattened_jacobian, axis=0)
        dpsi_i = tf.reshape(dpsi_i, [-1,1])

        # To compute <O^m O^n>
        dpsi_ij = tf.linalg.matmul(flattened_jacobian, flattened_jacobian, transpose_a = True) / self.nwalkers

        # Computing <O^m H>:
        dpsi_i_EL = tf.linalg.matmul(tf.reshape(energy, [1,self.nwalkers]), flattened_jacobian)
        # This makes this the same shape as the other tensors
        dpsi_i_EL = tf.reshape(dpsi_i_EL, [-1, 1])

        return dpsi_i, dpsi_ij, dpsi_i_EL

    def sr_step(self, nequilibrations, naverages, nobservations, nvoid):
        nblock = nequilibrations + naverages
        block_estimator = Estimator(info=None)
        total_estimator = Estimator(info=None)

        # Metropolis sampler will sample configurations uniformly:
        # Sample initial configurations uniformy between -sig and sig
        x_original = self.sampler.sample()



        # This function operates with the following loops:
        #  - There are nblock "blocks" == nequilibrations + naverages
        #    - the first 'nequilibrations' are not used for gradient calculations
        #    - the next 'naverages' are used for accumulating wavefunction measurements
        #  - In each block, there are nsteps = nobservations * nvoid
        #    - The walkers are kicked at every step.
        #    - if the step is a multiple of nvoid, AND the block is bigger than nequilibrations, the calculations are done.
        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.2}

        # First do a void walk to equilabrate / thermalize after a new configuration:
        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=1)
        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=nvoid*nequilibrations*naverages)
        total_estimator, flat_shape = self.sr_walk_and_compute(
            naverages = tf.constant(naverages),
            nobservations = tf.constant(nobservations),
            nvoid = nvoid,
            block_estimator = block_estimator,
            total_estimator = total_estimator,
            wavefunction = self.wavefunction,
            sampler = self.sampler)

        error, error_jf = total_estimator.finalize(naverages)
        energy = total_estimator.energy
        energy_jf = total_estimator.energy_jf
        acceptance = total_estimator.acceptance
        dpsi_i = total_estimator.dpsi_i
        dpsi_i_EL = total_estimator.dpsi_i_EL
        dpsi_ij = total_estimator.dpsi_ij
        # if MPI_AVAILABLE:
        #     # Here, we have to do a reduction over all params used to calculate gradients
        #     energy    = hvd.allreduce(energy)
        #     dpsi_i    = hvd.allreduce(dpsi_i)
        #     dpsi_i_EL = hvd.allreduce(dpsi_i_EL)
        #     dpsi_ij   = hvd.allreduce(dpsi_ij)
###########################################################################################
# NOTE: THE ABOVE REDUCTION WILL MESS UP THE ERROR CALCULATIONS
###########################################################################################
        # logger.info(f"psi norm{tf.reduce_mean(log_wpsi)}")
        dp_i = self.optimizer.sr(energy,dpsi_i,dpsi_i_EL,dpsi_ij)

        # Here, we recover the shape of the parameters of the network:
        running_index = 0
        gradient = []
        for length in flat_shape:
            l = length[-1].numpy()
            end_index = running_index + l
            gradient.append(dp_i[running_index:end_index])
            running_index += l
        shapes = [ p.shape for p in self.wavefunction.trainable_variables ]
        delta_p = [ tf.reshape(g, s) for g, s in zip(gradient, shapes)]

        return energy, error, energy_jf, error_jf, acceptance, delta_p

    def sr_step_optimized(self, nequilibrations, naverages, nobservations, nvoid):

        nblock = nequilibrations + naverages
        block_estimator = Estimator(info=None)
        total_estimator = Estimator(info=None)

        # Metropolis sampler will sample configurations uniformly:
        # Sample initial configurations uniformy between -sig and sig
        x_original = self.sampler.sample()



        # This function operates with the following loops:
        #  - There are nblock "blocks" == nequilibrations + naverages
        #    - the first 'nequilibrations' are not used for gradient calculations
        #    - the next 'naverages' are used for accumulating wavefunction measurements
        #  - In each block, there are nsteps = nobservations * nvoid
        #    - The walkers are kicked at every step.
        #    - if the step is a multiple of nvoid, AND the block is bigger than nequilibrations, the calculations are done.
        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.2}

        # First do a void walk to equilabrate / thermalize after a new configuration:
        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=1)
        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=nvoid)

        block_estimator.reset()
        # Kick the walkers nvoid times:
        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=nvoid)
        x_current = self.sampler.sample()

        # Energy here is computed per walker.
        energy, energy_jf = self.hamiltonian.energy(self.wavefunction, x_current)
        flattened_jacobian, flat_shape = self.jacobian(x_current, self.wavefunction)

        # Now, reshape
        
        # Dividing by the number of walkers to normalize the energy
        energy = energy / self.nwalkers
        energy_jf = energy_jf / self.nwalkers


        dpsi_i, dpsi_ij, dpsi_i_EL = self.compute_O_observables(flattened_jacobian, energy)

        block_estimator.accumulate(
            tf.reduce_sum(energy),
            tf.reduce_sum(energy_jf),
            acceptance,
            1.,
            dpsi_i,
            dpsi_i_EL,
            dpsi_ij,
            1.)

        # Outside of the i_prop loop, we accumulate into the total_estimator:
        total_estimator.accumulate(
            block_estimator.energy,
            block_estimator.energy_jf,
            block_estimator.acceptance,
            0,
            block_estimator.dpsi_i,
            block_estimator.dpsi_i_EL,
            block_estimator.dpsi_ij,
            block_estimator.weight)


        error, error_jf = total_estimator.finalize(naverages)
        energy = total_estimator.energy
        energy_jf = total_estimator.energy_jf
        acceptance = total_estimator.acceptance
        dpsi_i = total_estimator.dpsi_i
        dpsi_i_EL = total_estimator.dpsi_i_EL
        dpsi_ij = total_estimator.dpsi_ij
        # if MPI_AVAILABLE:
        #     # Here, we have to do a reduction over all params used to calculate gradients
        #     energy    = hvd.allreduce(energy)
        #     dpsi_i    = hvd.allreduce(dpsi_i)
        #     dpsi_i_EL = hvd.allreduce(dpsi_i_EL)
        #     dpsi_ij   = hvd.allreduce(dpsi_ij)
###########################################################################################
# NOTE: THE ABOVE REDUCTION WILL MESS UP THE ERROR CALCULATIONS
###########################################################################################
        # logger.info(f"psi norm{tf.reduce_mean(log_wpsi)}")
        dp_i = self.optimizer.sr(energy,dpsi_i,dpsi_i_EL,dpsi_ij)

        # Here, we recover the shape of the parameters of the network:
        running_index = 0
        gradient = []
        for length in flat_shape:
            l = length[-1].numpy()
            end_index = running_index + l
            gradient.append(dp_i[running_index:end_index])
            running_index += l
        shapes = [ p.shape for p in self.wavefunction.trainable_variables ]
        delta_p = [ tf.reshape(g, s) for g, s in zip(gradient, shapes)]

        return energy, error, energy_jf, error_jf, acceptance, delta_p


    # @tf.function
    def sr_walk_and_compute(self,naverages,nobservations,nvoid,block_estimator,total_estimator, wavefunction, sampler):
        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.2}

        # Now, we loop over the number of blocks.
        # For each block, accumulate nobservations times, with nvoid steps in between.
        total_estimator.reset()

        for i_av in tf.range(naverages):
            block_estimator.reset()
            for i_prop in tf.range(nobservations):
                # Kick the walkers nvoid times:
                acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=nvoid)
                x_current = self.sampler.sample()

                # Energy here is computed per walker.
                energy, energy_jf = self.hamiltonian.energy(self.wavefunction, x_current)
                # Dividing by the number of walkers to normalize the energy
                energy = energy / self.nwalkers
                energy_jf = energy_jf / self.nwalkers

                flattened_jacobian, flat_shape = self.jacobian(x_current, self.wavefunction)

                dpsi_i, dpsi_ij, dpsi_i_EL = self.compute_O_observables(flattened_jacobian, energy)

                block_estimator.accumulate(
                    tf.reduce_sum(energy),
                    tf.reduce_sum(energy_jf),
                    acceptance,
                    1.,
                    dpsi_i,
                    dpsi_i_EL,
                    dpsi_ij,
                    1.)
            # Outside of the i_prop loop, we accumulate into the total_estimator:
            total_estimator.accumulate(
                block_estimator.energy,
                block_estimator.energy_jf,
                block_estimator.acceptance,
                0,
                block_estimator.dpsi_i,
                block_estimator.dpsi_i_EL,
                block_estimator.dpsi_ij,
                block_estimator.weight)



        return total_estimator, flat_shape

    def finalize(self):
        # Take the network and snapshot it to file:
        self.wavefunction.save_weights(self.save_path)

if __name__ == "__main__":
    e = exec()
    # with tf.profiler.experimental.Profile('logdir'):
    e.run()
    e.finalize()
