from .Fittable_Model import Fittable_Model
from ..grids import Grid
import numpy as np
import astropy.units as u

class Disk(Fittable_Model):

    default_params = {"mass": 1e-3*u.Msun,
                               "rin": 0.1*u.au,
                               "rout": 100.0*u.au,
                               "gamma": 1.0,
                               "h_0": 0.05*u.au,
                               "beta": 1.0}

    model_name = "disk"
    density_coordinates = "cylindrical"

    def surface_density(self, r):
        sigma0 = ((2.0 - self.gamma) * self.mass / (2.0 * np.pi * self.rout**2))
        sigma = sigma0 * (r / self.rout)**(-self.gamma) * np.exp(-(r / self.rout)**(2.0 - self.gamma))
        return sigma.to(u.g / u.cm**2)
    
    def scale_height(self, r):
        h = self.h_0 * (r / (1*u.au))**self.beta
        return h.to(u.au)
    
    def density(self, r, z):
        sigma = self.surface_density(r)
        h = self.scale_height(r)
        rho = sigma / (np.sqrt(2 * np.pi) * h) * np.exp(-0.5 * (z / h)**2)
        return rho.to(u.g / u.cm**3)
    