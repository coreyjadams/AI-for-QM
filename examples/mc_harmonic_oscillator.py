# Python built ins:
import sys, os
import time
import logging

# Frameworks:
import numpy 
import torch

# Add the local folder to the import path:
top_folder = os.path.dirname(os.path.abspath(__file__))
top_folder = os.path.dirname(top_folder)
sys.path.insert(0,top_folder)

#from mlqm.samplers      import CartesianSampler
from mlqm.hamiltonians  import HarmonicOscillator_mc
from mlqm.models        import NeuralWavefunction
from mlqm.hamiltonians  import NuclearPotential
from mlqm.samplers      import Estimator
from mlqm.optimization  import Optimizer


sig = 0.2
dx = 0.2
neq = 10
nav = 20
nprop = 10
nvoid = 100
nwalk = 1000
nopt = 50
ndim = 3
npart = 4
seed = 17
mass = 1.
omega = 1.
delta = 0.001
eps = 0.0001
pot_name = 'pionless_4'
module_load = True

# Module save
model_save_path = f"./{pot_name}_nucleus_{npart}.model"

# Set up logging:
logger = logging.getLogger()
# Create a file handler:
hdlr = logging.FileHandler(f'nucleus_{npart}_{pot_name}.log')
# Add formatting to the log:
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
#ch = logging.StreamHandler()
logger.addHandler(hdlr) 
#logger.addHandler(ch)
# Set the default level. Levels here: https://docs.python.org/2/library/logging.html
logger.setLevel(logging.DEBUG)

logger.info(f"sig = {sig}")
logger.info(f"dx = {dx}")
logger.info(f"neq = {neq}")
logger.info(f"nav = {nav}")
logger.info(f"nprop = {nprop}")
logger.info(f"nvoid = {nvoid}")
logger.info(f"nwalk = {nwalk}")
logger.info(f"nopt = {nopt}")
logger.info(f"ndim = {ndim}")
logger.info(f"npart = {npart}")
logger.info(f"seed = {seed}")
logger.info(f"mass = {mass}")
logger.info(f"omega = {omega}")
logger.info(f"delta = {delta}")
logger.info(f"eps = {eps}")
logger.info(f"potential model = {pot_name}")
logger.info("\n")



# Initialize Seed
torch.manual_seed(seed)

# Initialize neural wave function and compute the number of parameters
wavefunction = NeuralWavefunction(ndim, npart)
wavefunction.count_parameters()

# Initialize Potential
potential = NuclearPotential(nwalk, pot_name)

# Initialize Hamiltonian 
hamiltonian =  HarmonicOscillator_mc(mass, omega, nwalk, ndim, npart)

#Initialize Optimizer
opt=Optimizer(delta,eps,wavefunction.npt)

# Initialize Flattener
params_flat, indeces_flat = wavefunction.flatten_params(wavefunction.parameters())

if module_load:
   logger.info(f"module loaded")
   wavefunction.load_state_dict(torch.load(model_save_path))

# Propagation
def energy_metropolis(neq, nav, nprop, nvoid, hamiltonian, wavefunction):
    nblock = neq + nav
    nstep = nprop * nvoid
    block_estimator = Estimator(info=None)
    block_estimator.reset()
    total_estimator = Estimator(info=None)
    total_estimator.reset()
# Sample initial configurations uniformy between -sig and sig
    x_o = torch.normal(0., sig, size=[nwalk, npart, ndim])
    for i in range (nblock):
        block_estimator.reset()
        if (i == neq) :
           total_estimator.reset()
        for j in range (nstep):
            with torch.no_grad(): 
                log_wpsi_o = wavefunction(x_o)
# Gaussian transition probability 
                x_n = x_o + torch.normal(0., dx, size=[nwalk, npart, ndim])
                log_wpsi_n = wavefunction(x_n)
# Accepance probability |psi_n|**2 / |psi_o|**2
                prob = 2 * ( log_wpsi_n - log_wpsi_o )
                accept = torch.ge(prob, torch.log(torch.rand(size=[nwalk])) )
                x_o = torch.where(accept.view([nwalk,1,1]), x_n, x_o)
                acceptance = torch.mean(accept.float())
# Compute energy and accumulate estimators within a given block
            if ( (j+1) % nvoid == 0 and i >= neq ):
                energy, energy_jf = hamiltonian.energy(wavefunction, potential, x_o)
                energy = energy / nwalk
                energy_jf = energy_jf / nwalk
                energy.detach_()
                energy_jf.detach_()

# Compute < O^i >, < H O^i >,  and < O^i O^j > 
                log_wpsi = wavefunction(x_o)
                jac = torch.zeros(size=[nwalk,wavefunction.npt])
                for n in range(nwalk):
                    wavefunction.zero_grad()
                    params = wavefunction.parameters()
                    dpsi_dp = torch.autograd.grad(log_wpsi[n], params, retain_graph=True)
                    dpsi_i_n = wavefunction.flatten_grad(dpsi_dp)
                    jac[n,:] = torch.t(dpsi_i_n)
#                log_wpsi_n.detach_()
                dpsi_i = torch.sum(jac, dim=0) / nwalk
                dpsi_i = dpsi_i.view(-1,1)
                dpsi_i_EL = torch.matmul(energy, jac).view(-1,1)
                dpsi_ij = torch.mm(torch.t(jac), jac) / nwalk

#                print("dpsi_i", dpsi_i)
#                print("dpsi_ij", dpsi_ij)
#                exit()
                block_estimator.accumulate(torch.sum(energy),torch.sum(energy_jf),acceptance,1.,dpsi_i,dpsi_i_EL,dpsi_ij,1.)
# Accumulate block averages
        if ( i >= neq ):
            total_estimator.accumulate(block_estimator.energy,block_estimator.energy_jf,block_estimator.acceptance,0,block_estimator.dpsi_i,
                block_estimator.dpsi_i_EL,block_estimator.dpsi_ij,block_estimator.weight)

    error, error_jf = total_estimator.finalize(nav)
    energy = total_estimator.energy
    energy_jf = total_estimator.energy_jf
    acceptance = total_estimator.acceptance
    dpsi_i = total_estimator.dpsi_i
    dpsi_i_EL = total_estimator.dpsi_i_EL
    dpsi_ij = total_estimator.dpsi_ij

    logger.info(f"psi norm = {torch.mean(log_wpsi)}")

    with torch.no_grad(): 
        dp_i = opt.sr(energy,dpsi_i,dpsi_i_EL,dpsi_ij)
        gradient = wavefunction.recover_flattened(dp_i, indeces_flat, wavefunction)
        delta_p = [ g for g in gradient]

    return energy, error, energy_jf, error_jf, acceptance, delta_p

t0 = time.time()
energy, error, energy_jf, error_jf, acceptance, delta_p = energy_metropolis(neq, nav, nprop, nvoid, hamiltonian, wavefunction)
t1 = time.time()
logger.info(f"initial_energy {energy, error}")
logger.info(f"initial_jf_energy {energy_jf, error_jf}")
logger.info(f"initial_acceptance {acceptance}")
logger.info(f"elapsed time {t1 - t0}")

for i in range(nopt):

        # Compute the energy:
        energy, error, energy_jf, error_jf, acceptance, delta_p = energy_metropolis(neq, nav, nprop, nvoid, hamiltonian, wavefunction)
        
        for (p, dp) in zip (wavefunction.parameters(),delta_p):
            p.data = p.data + dp 
            
        if i % 1 == 0:
            logger.info(f"step = {i}, energy = {energy.data:.3f}, err = {error.data:.3f}")
            logger.info(f"step = {i}, energy_jf = {energy_jf.data:.3f}, err = {error_jf.data:.3f}")
            logger.info(f"acc = {acceptance.data:.3f}")

# This saves the model:
torch.save(wavefunction.state_dict(), model_save_path)

