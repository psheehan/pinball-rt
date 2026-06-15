from .Fittable_Model import Fittable_Model
from ..grids import Grid
from scipy.integrate import trapezoid
import numpy as np
import astropy.units as u

class Envelope(Fittable_Model):

    default_params = {"rho0": 2e-12*u.g/u.cm**3,
                      "rmin": 0.1*u.au,
                      "rmax": 1000*u.au,
                      "pl": 1.5,
                      "cavpl": 1.0,
                      "cavrrfact":0.2}

    model_name = "envelope"
    density_coordinates = "spherical"

    def density(self, r, theta, phi):
        rho = self.rho0 * (r / self.rmin)**(-self.pl)
        return rho.to(u.g / u.cm**3)