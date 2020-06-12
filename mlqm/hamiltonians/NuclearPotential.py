import torch
import numpy as np

#from mlqm import H_BAR
#from mlqm import M


class NuclearPotential(object):
    def __init__(self, nwalk, pot_name):
        self.nwalk=nwalk
        if (pot_name == 'pionless_2'):
            self.vkr = 2.0
            self.v0r = -133.3431
            self.v0s = -9.0212
            self.ar3b = np.sqrt(68.48830)
        elif (pot_name == 'pionless_4'):
            self.vkr = 4.0
            self.v0r = -487.6128
            self.v0s = -17.5515
            self.ar3b = np.sqrt(677.79890)
        elif (pot_name == 'pionless_6'):
            self.vkr = 6.0
            self.v0r = -1064.5010
            self.v0s = -26.0830
            self.ar3b = np.sqrt(2652.65100)
        object.__init__(self)

    def pionless_2b(self, rr):
        pot_2b=torch.zeros(self.nwalk,6)
        x = self.vkr * rr
        vr = torch.exp(-x**2 / 4.0)
        pot_2b[:,0] = self.v0r * vr
        pot_2b[:,2] = self.v0s * vr
        return pot_2b

    def pionless_3b(self, rr):
        pot_3b = torch.zeros(self.nwalk)
        x = self.vkr * rr
        vr = torch.exp(-x**2 / 4.0)
        pot_3b = self.ar3b * vr
        return pot_3b



