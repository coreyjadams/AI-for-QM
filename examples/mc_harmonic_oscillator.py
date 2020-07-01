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
nav = 10
nprop = 10
nvoid = 200
nwalk = 200
nopt = 40
ndim = 3
npart = 4
seed = 17
mass = 938.95
hbar = 197.327
delta = 0.001
eps = 0.0001
conf = 0.1
pot_name = 'pionless_4'
module_load = False
module_write = False

#torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_default_dtype(torch.float64)

# Module save
model_save_path = f"./{pot_name}_nucleus_{npart}.model"

# Set up logging:
logger = logging.getLogger()
# Create a file handler:
hdlr = logging.FileHandler(f'{pot_name}_nucleus_{npart}.log')
# Add formatting to the log:
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
ch = logging.StreamHandler()
logger.addHandler(hdlr) 
logger.addHandler(ch)
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
logger.info(f"hbar = {hbar}")
logger.info(f"delta = {delta}")
logger.info(f"eps = {eps}")
logger.info(f"conf = {conf}")
logger.info(f"potential model = {pot_name}")
logger.info("\n")

# Initialize Seed
torch.manual_seed(seed)

# Initialize neural wave function and compute the number of parameters
wavefunction = NeuralWavefunction(ndim, npart, conf)
wavefunction.count_parameters()
#wavefunction.double()

# Initialize Potential
potential = NuclearPotential(nwalk, pot_name)

# Initialize Hamiltonian 
hamiltonian =  HarmonicOscillator_mc(mass, hbar, nwalk, ndim, npart)

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
    estimator = Estimator(nopt, info=None)
    estimator.reset()
# Sample initial configurations uniformy between -sig and sig
    x_o = torch.normal(0., sig, size=[nwalk, npart, ndim])
    x_s = []    
    for i in range (nblock):
        if (i == neq) :
           estimator.reset()
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

#                print("x_s before=", x_s)
#                print("x_o=", x_o)
#                x_s = torch.cat((x_o, x_s), 0)
                x_s.append(x_o)
#                print("x_s after=", torch.cat(x_s,dim=0))
#                exit()
                energy, energy_jf = hamiltonian.energy(wavefunction, potential, x_o)
                energy.detach_()
                energy_jf.detach_()
                if ( nopt == 0 ):
                    rho = hamiltonian.density(x_o) / nwalk
                else:
                    rho = 0

# Compute < O^i >, < H O^i >,  and < O^i O^j > 
                if (nopt > 0):
                    log_wpsi = wavefunction(x_o)
                    jac = torch.zeros(size=[nwalk,wavefunction.npt])
                    for n in range(nwalk):
                        wavefunction.zero_grad()
                        params = wavefunction.parameters()
                        dpsi_dp = torch.autograd.grad(log_wpsi[n], params, retain_graph=True)
                        dpsi_i_n = wavefunction.flatten_grad(dpsi_dp)
                        jac[n,:] = torch.t(dpsi_i_n)
                    log_wpsi.detach_()
                    dpsi_i = torch.sum(jac, dim=0) / nwalk
                    dpsi_i = dpsi_i.view(-1,1)
                    dpsi_i_EL = torch.matmul(energy, jac).view(-1,1) / nwalk
                    dpsi_ij = torch.mm(torch.t(jac), jac) / nwalk
                else:
                    dpsi_i = 0
                    dpsi_i_EL = 0
                    dpsi_ij = 0
                energy = torch.mean(energy) 
                energy_jf = torch.mean(energy_jf) 
                estimator.addval(energy,energy_jf,acceptance,1.,dpsi_i,dpsi_i_EL,dpsi_ij,rho)

# Accumulate block averages
        if ( i >= neq ):
            estimator.addblk()

    error, error_jf, error_rho = estimator.average()
    energy = estimator.energy
    energy_jf = estimator.energy_jf
    acceptance = estimator.acceptance
    dpsi_i = estimator.dpsi_i
    dpsi_i_EL = estimator.dpsi_i_EL
    dpsi_ij = estimator.dpsi_ij
    rho = estimator.rho

    if (nopt == 0 ):
        hamiltonian.density_print(rho, error_rho)

    logger.info(f"psi norm = {torch.mean(log_wpsi_o)}")
    if (nopt > 0) :
        with torch.no_grad(): 
            dp_i = opt.sr(energy,dpsi_i,dpsi_i_EL,dpsi_ij)
            gradient = wavefunction.recover_flattened(dp_i, indeces_flat, wavefunction)
            delta_p = [ g for g in gradient]
    else:
        delta_p = 0

    x_s = torch.cat(x_s,dim=0)
    energy_s, energy_jf_s = hamiltonian.energy(wavefunction, potential, x_s)
    print("energy_s=",torch.mean(energy_s).data)
    print("energy=",energy.data)
#    exit()
    return energy, error, energy_jf, error_jf, acceptance, delta_p

# Call propagation once for timing purposes
t0 = time.time()
energy, error, energy_jf, error_jf, acceptance, delta_p = energy_metropolis(neq, nav, nprop, nvoid, hamiltonian, wavefunction)
t1 = time.time()
logger.info(f"initial_energy {energy, error}")
logger.info(f"initial_jf_energy {energy_jf, error_jf}")
logger.info(f"initial_acceptance {acceptance}")
logger.info(f"elapsed time {t1 - t0}")

# Optimization
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
if  module_write:
    torch.save(wavefunction.state_dict(), model_save_path)


