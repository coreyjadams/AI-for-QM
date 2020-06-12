import numpy
import torch

#from .ExponentialBoundaryCondition import ExponentialBoundaryCondition

class NeuralWavefunction(torch.nn.Module):
    """Create a neural network eave function in N dimensions
    
    Boundary condition, if not supplied, is gaussian in every dimension
    
    Extends:
        torch.nn.Module
    """
    def __init__(self, ndim : int, npart: int, boundary_condition :torch.nn.Module = None):
        torch.nn.Module.__init__(self)
        
        self.ndim = ndim
#        if self.ndim < 1 or self.ndim > 3: 
#            raise Exception("Dimension must be 1, 2, or 3 for NeuralWavefunction")

        self.npart = npart
        
        # Create a boundary condition if needed:
#        if boundary_condition is None:
#            self.bc = ExponentialBoundaryCondition(self.ndim)
#        else:
#            self.bc = boundary_condition

        self.layer1 = torch.nn.Linear(self.ndim * self.npart, 64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.layer3 = torch.nn.Linear(64, 1, bias = False)

# Test solution psi = exp(-(a*x+b)**2/2)
#        self.layer1 = torch.nn.Linear(self.ndim, 1)
#        self.layer1.weight.data.fill_(0.7)
#        self.layer1.bias.data.fill_(0.1)

#        self.exponent = torch.nn.Parameter(2. * torch.ones(self.ndim), requires_grad=True)
#        self.shift = torch.nn.Parameter(torch.ones(1), requires_grad=True)
#        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
#        self.beta = torch.nn.Parameter(0.5 * torch.ones(1), requires_grad=True)


    def forward(self, inputs):
        mean = torch.mean(inputs, dim=1)
        xinputs = inputs - mean[:,None,:]
        x = torch.flatten(xinputs, start_dim = 1)
        x = self.layer1(x)
        x = torch.nn.functional.softplus(x, beta=1.)
        x = self.layer2(x)
        x = torch.nn.functional.softplus(x, beta=1.)
        x = self.layer3(x)
        x = x.view([x.shape[0],])
        x = x - torch.sum( 0.1 * xinputs**2, dim=(1,2))
        result = x
        return result

    def forward2(self, inputs):
        inputs_12 = torch.zeros(size = [800,self.ndim])
        inputs_12 = inputs[:,1,:] - inputs[:,0,:] 
        x_12 = self.layer1(inputs_12)
        x_12 = torch.nn.functional.softplus(x_12, beta=1.)
        x_12 = self.layer2(x_12)
        x_12 = torch.nn.functional.softplus(x_12, beta=1.)
        x_12 = self.layer3(x_12)
        x_12 = x_12.view([x_12.shape[0],]) 
        result = x_12
 
        inputs_21 = torch.zeros(size = [800,self.ndim])
        inputs_21 = inputs[:,0,:] - inputs[:,1,:] 
        x_21 = self.layer1(inputs_21)
        x_21 = torch.nn.functional.softplus(x_21, beta=1.)
        x_21 = self.layer2(x_21)
        x_21 = torch.nn.functional.softplus(x_21, beta=1.)
        x_21 = self.layer3(x_21)
        x_21 = x_21.view([x_21.shape[0],]) 
        result += x_21

        result -= torch.sum( 0.05 * inputs_12**2, dim=1)
        return result
        

# Exact wave function
#        inputs_12 = torch.zeros(size = [8000,self.ndim])
#        inputs_12 = inputs[:,1,:] - inputs[:,0,:] 
#        r_12 = torch.sqrt(torch.sum(inputs_12**2,dim=1))
#        a0 = 4.14877 
#        a1 = 23.5426  
#        a2 = -3.64539
#        a3 = -8.11022
#        a4 = 14.4502
#        a5 = -5.00758
#        a6 = 0.765287
#        ll = 2.94036
#        mm = 0.799155
#        f_12 = (a0 + a1*r_12 + a2*r_12**2 + a3*r_12**3 + a4*r_12**4 + a5*r_12**5 +a6*r_12**6) * torch.exp(-ll*r_12**mm)
#        f_12 = self.shift*f_12
#        result = torch.log(f_12)
#        for i in range (200):
#            print(r_12[i].item(), f_12[i].item())
#        exit()

#        print("inputs_12", inputs_12)
#        print("self.shift", self.shift)
#        r2 = torch.sum(inputs_12**2,dim=1)
#        f_12 =  (1. + self.beta * r2) * torch.exp(- self.alpha  * r2)
#        result = torch.log(f_12)
#        print("f_12", result)
#        exit()
        

    def importance(self, inputs):
#        result = inputs.view([inputs.shape[0],]) 
        result = torch.sum( 0.01 * inputs**4, dim=(1,2))
        print("inputs=", inputs)
        print("wf_I result=", result)
        exit()
        return result

    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def count_parameters(self):
        self.np =sum(p.numel() for p in self.parameters() )
        self.npt =sum(p.numel() for p in self.parameters() if p.requires_grad)

    def flatten_params(self, parameters):
    #flattens all parameters into a single column vector. Returns the dictionary to recover them
    #:param: parameters: a generator or list of all the parameters
    #:return: a dictionary: {"params": [#params, 1],
    #"indices": [(start index, end index) for each param] **Note end index in uninclusive**
    
        l = [torch.flatten(p) for p in parameters]
        indices = []
        s = 0
        for p in l:
            size = p.shape[0]
            indices.append((s, s+size))
            s += size
        flat = torch.cat(l).view(-1, 1)
        return flat, indices


    def flatten_grad(self, parameters):
    #flattens all the gradients parameters into a single column vector.
    
        l = [torch.flatten(p) for p in parameters]
        flat = torch.cat(l).view(-1, 1)
        return flat

    def recover_flattened(self, flat_params, indices, model):
        """
        Gives a list of recovered parameters from their flattened form
        :param flat_params: [#params, 1]
        :param indices: a list detaling the start and end index of each param [(start, end) for param]
        :param model: the model that gives the params with correct shapes
        :return: the params, reshaped to the ones in the model, with the same order as those in the model
        """
        l = [flat_params[s:e] for (s, e) in indices]
        for i, p in enumerate(model.parameters()):
            l[i] = l[i].view(*p.shape)
        return l


