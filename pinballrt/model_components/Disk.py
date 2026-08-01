from .Fittable_Model import Fittable_Model
import pandas as pd
import numpy as np
import astropy.units as u


class Disk(Fittable_Model):

    model_name = "disk"
    model_default_params = pd.DataFrame.from_dict({
        "disk_mass":  {"value": -3., "prior_min": -5., "prior_max": 0., 
        "Fixed": False, "unit": u.Msun, "log":True, "component": model_name},
        "disk_rin": {"value": -1., "prior_min": -2., "prior_max": 0., 
        "Fixed": False, "unit": u.au, "log":True, "component": model_name},
        "disk_rout": {"value": 2., "prior_min": 1., "prior_max": 3., 
        "Fixed": False, "unit": u.au, "log":True, "component": model_name},
        "gamma": {"value": 1., "prior_min": 0., "prior_max": 2., 
        "Fixed": False,  "component": model_name},
        "h0": {"value": 0.05, "prior_min": 0.01, "prior_max": 0.3, 
        "Fixed": False, "unit": u.au, "component": model_name},
        "beta": {"value": 1., "prior_min": 0.5, "prior_max": 1.5, 
        "Fixed": False,  "component": model_name}
    }, orient='index')

    density_coordinates = "cylindrical"

    def density(self, r, z):
        mass  = self.disk_mass.cgs.value
        rin   = self.disk_rin.cgs.value
        rout  = self.disk_rout.cgs.value
        gamma = self.gamma
        h0    = self.h0.cgs.value
        beta  = self.beta

        sigma0 = ((2.0 - gamma) * mass / (2.0 * np.pi * rout**2))
        sigma = sigma0 * (r / rout)**(-gamma) * np.exp(-(r / rout)**(2.0 - gamma))

        h = h0 * (r / (1*u.au).cgs.value)**beta

        rho = sigma / (np.sqrt(2 * np.pi) * h) * np.exp(-0.5 * (z / h)**2)
        return rho * u.g / u.cm**3
