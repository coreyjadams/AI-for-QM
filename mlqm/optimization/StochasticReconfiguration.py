import tensorflow as tf
import numpy

import logging
# Set up logging:
logger = logging.getLogger()

from mlqm.hamiltonians import Hamiltonian
from mlqm.optimization import Optimizer
from mlqm.samplers     import Estimator, MetropolisSampler


try:
    import horovod.tensorflow as hvd
    hvd.init()
    from mpi4py import MPI

    # This is to force each rank onto it's own GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
    MPI_AVAILABLE=True
except:
    MPI_AVAILABLE=False


class StochasticReconfiguration(object):

    def __init__(self,
            sampler                   : MetropolisSampler,
            wavefunction              : callable,
            hamiltonian               : Hamiltonian,
            optimizer                 : Optimizer,
            n_observable_measurements : int,
            n_void_steps              : int,
            n_walkers_per_observation : int,
            n_concurrent_obs_per_rank : int,
        ):

        # Store the objects:
        self.sampler      = sampler
        self.wavefunction = wavefunction
        self.hamiltonian  = hamiltonian
        self.optimizer    = optimizer

        # Store the measurement configurations:
        self.n_observable_measurements = n_observable_measurements
        self.n_void_steps              = n_void_steps
        self.n_walkers_per_observation = n_walkers_per_observation
        self.n_concurrent_obs_per_rank = n_concurrent_obs_per_rank

        # MPI Enabled?
        if MPI_AVAILABLE:
            self.size = hvd.size()
            self.rank = hvd.rank()
        else:
            self.size = 1
            self.rank = 1

        self.estimator = Estimator()




    @tf.function
    def batched_jacobian(self, nobs, x_current_arr, wavefunction, jac_fnc):
        ret_jac = []
        # ret_shape = []
        for i in range(nobs):
            flattened_jacobian, flat_shape = jac_fnc(x_current_arr[i], wavefunction)
            ret_jac.append(flattened_jacobian)
            # ret_shape.append(flat_shape)

        return ret_jac, flat_shape





    @tf.function
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



    @tf.function
    def compute_O_observables(self, flattened_jacobian, energy):

        # dspi_i is the reduction of the jacobian over all walkers.
        # In other words, it's the mean gradient of the parameters with respect to inputs.
        # This is effectively the measurement of O^i in the paper.
        dpsi_i = tf.reduce_mean(flattened_jacobian, axis=0)
        dpsi_i = tf.reshape(dpsi_i, [-1,1])

        # To compute <O^m O^n>
        dpsi_ij = tf.linalg.matmul(flattened_jacobian, flattened_jacobian, transpose_a = True) / self.n_walkers_per_observation

        # Computing <O^m H>:
        dpsi_i_EL = tf.linalg.matmul(tf.reshape(energy, [1,self.n_walkers_per_observation]), flattened_jacobian)
        # This makes this the same shape as the other tensors
        dpsi_i_EL = tf.reshape(dpsi_i_EL, [-1, 1])

        return dpsi_i, dpsi_ij, dpsi_i_EL

    def update_model(self):

        if self.latest_gradients is not None:

            # Update the parameters:
            for i_param in range(len(self.wavefunction.trainable_variables)):
                # print(f"Gradient: {self.latest_gradients[i_param]}")
                # print(f"Variable: {self.wavefunction.trainable_variables[i_param]}")
                # print(f"Ratio: {tf.reduce_mean(tf.abs(self.latest_gradients[i_param] / self.wavefunction.trainable_variables[i_param]))}")
                self.wavefunction.trainable_variables[i_param].assign_add(self.latest_gradients[i_param])


    def equilibrate(self, n_equilibrations):

        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.4}

        acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=n_equilibrations)

        return acceptance

    def sr_step(self):

        metrics = {}
        self.latest_gradients = None


        kicker = tf.random.normal
        kicker_params = {"mean": 0.0, "stddev" : 0.4}

        self.estimator.reset()

        # We need to know how many times to loop over the walkers and metropolis step.
        # The total number of observations is set: self.n_observable_measurements
        # There is an optimization to walk in parallel with n_concurrent_obs_per_rank
        # Without MPI, the number of loops is then n_observable_measurements / n_concurrent_obs_per_rank
        # WITH MPI, we have to reduce the number of loops by the total number of ranks.

        n_loops_total = int(self.n_observable_measurements / self.n_concurrent_obs_per_rank)

        if MPI_AVAILABLE:
            n_loops_total /= self.size

        # logger.debug(" -- Coordinating loop length")

        # We do a check that n_loops_total * n_concurrent_obs_per_rank matches expectations:
        if n_loops_total * self.n_concurrent_obs_per_rank*self.size != self.n_observable_measurements:
            exception_str = "Total number of observations to compute is unexpected!\n"
            exception_str += f"  Expected to have {self.n_observable_measurements}, have:\n"
            exception_str += f"  -- A loop of {self.n_concurrent_obs_per_rank} observations"
            exception_str += f" for {n_loops_total} loops over {self.size} ranks"
            exception_str += f"  -- ({self.n_concurrent_obs_per_rank})*({n_loops_total}"
            exception_str += f")*({self.size}) != {self.n_observable_measurements}\n"
            raise Exception(exception_str)


        for i_loop in range(n_loops_total):
            # logger.debug(f" -- evaluating loop {i_loop} of {n_loops_total}")

            # First do a void walk to thermalize after a new configuration.
            # By default, this will use the previous walkers as a starting configurations.
            #   This one does all the kicks in a compiled function.
            acceptance = self.sampler.kick(self.wavefunction, kicker, kicker_params, nkicks=self.n_void_steps)


            # Get the current walker locations:
            x_current  = self.sampler.sample()

            # Compute the observables:
            energy, energy_jf, ke_jf, ke_direct, pe = self.hamiltonian.energy(self.wavefunction, x_current)

            # Here, we split the energy and other objects into sizes of nwalkers_per_observation
            # if self.n_concurrent_obs_per_rank != 1:
            x_current  = tf.split(x_current, self.n_concurrent_obs_per_rank, axis=0)
            energy     = tf.split(energy,    self.n_concurrent_obs_per_rank, axis=0)
            energy_jf  = tf.split(energy_jf, self.n_concurrent_obs_per_rank, axis=0)
            ke_jf      = tf.split(ke_jf,     self.n_concurrent_obs_per_rank, axis=0)
            ke_direct  = tf.split(ke_direct, self.n_concurrent_obs_per_rank, axis=0)
            pe         = tf.split(pe,        self.n_concurrent_obs_per_rank, axis=0)

            # For each observation, we compute the jacobian.
            # flattened_jacobian is a list, flat_shape is just one instance
            flattened_jacobian, flat_shape = self.batched_jacobian(
                self.n_concurrent_obs_per_rank, x_current, self.wavefunction, self.jacobian)

            # Here, if MPI is available, we can do a reduction (sum) over walker variables

            # Now, compute observables, store them in an estimator:

            for i_obs in range(self.n_concurrent_obs_per_rank):
                obs_energy      = energy[i_obs]     / self.n_walkers_per_observation
                obs_energy_jf   = energy_jf[i_obs]  / self.n_walkers_per_observation

                dpsi_i, dpsi_ij, dpsi_i_EL = self.compute_O_observables(flattened_jacobian[i_obs], obs_energy)

                self.estimator.accumulate(
                    tf.reduce_sum(obs_energy),
                    tf.reduce_sum(obs_energy_jf),
                    acceptance,
                    1.,
                    dpsi_i,
                    dpsi_i_EL,
                    dpsi_ij,
                    1.)


        # INTERCEPT HERE with MPI to allreduce the estimator objects.
        if MPI_AVAILABLE:
            self.estimator.allreduce()

        # At this point, we need to average the observables that feed into the optimizer:
        error, error_jf = self.estimator.finalize(self.n_observable_measurements)

        metrics = {}
        metrics["energy/ke_jf"]     = tf.reduce_mean(tf.reduce_sum(ke_jf, axis=0) / self.n_walkers_per_observation)
        metrics["energy/ke"]        = tf.reduce_mean(tf.reduce_sum(ke_direct, axis=0) / self.n_walkers_per_observation)
        metrics["energy/pe"]        = tf.reduce_mean(tf.reduce_sum(pe, axis=0) / self.n_walkers_per_observation)
        metrics["energy/ke_jf_std"] = tf.math.reduce_std(ke_jf)
        metrics["energy/ke_std"]    = tf.math.reduce_std(ke_direct)
        metrics["energy/pe_std"]    = tf.math.reduce_std(pe)


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
        dp_i = self.optimizer.sr(
            self.estimator.tensor_dict["energy"],
            self.estimator.tensor_dict["dpsi_i"],
            self.estimator.tensor_dict["dpsi_i_EL"],
            self.estimator.tensor_dict["dpsi_ij"])

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

        metrics['energy/energy']     = self.estimator.tensor_dict["energy"]
        metrics['energy/error']      = error
        metrics['energy/energy_jf']  = self.estimator.tensor_dict["energy_jf"]
        metrics['energy/error_jf']   = error_jf
        metrics['metropolis/acceptance'] = self.estimator.tensor_dict["acceptance"]

        self.latest_gradients = delta_p

        return  metrics