import astropy.units as u
import numpy as np
import pandas as pd

from .Fittable_Model import Fittable_Model


class Disk:
    def __init__(self):
        self.model_name = "disk"
        self.density_coordinates = "cylindrical"
        model_default_params = pd.DataFrame.from_dict(
            {
                "disk_mass": {
                    "value": -3.0,
                    "prior_min": -5.0,
                    "prior_max": 0.0,
                    "Fixed": False,
                    "unit": u.Msun,
                    "log": True,
                    "component": self.model_name,
                },
                "disk_rin": {
                    "value": -1.0,
                    "prior_min": -2.0,
                    "prior_max": 0.0,
                    "Fixed": False,
                    "unit": u.au,
                    "log": True,
                    "component": self.model_name,
                },
                "disk_rout": {
                    "value": 2.0,
                    "prior_min": 1.0,
                    "prior_max": 3.0,
                    "Fixed": False,
                    "unit": u.au,
                    "log": True,
                    "component": self.model_name,
                },
                "gamma": {
                    "value": 1.0,
                    "prior_min": 0.0,
                    "prior_max": 2.0,
                    "Fixed": False,
                    "component": self.model_name,
                },
                "h0": {
                    "value": 0.05,
                    "prior_min": 0.01,
                    "prior_max": 0.3,
                    "Fixed": False,
                    "unit": u.au,
                    "component": self.model_name,
                },
                "beta": {
                    "value": 1.0,
                    "prior_min": 0.5,
                    "prior_max": 1.5,
                    "Fixed": False,
                    "component": self.model_name,
                },
            },
            orient="index",
        )

    def density(self, r, z):
        mass = self.disk_mass.cgs.value
        rin = self.disk_rin.cgs.value
        rout = self.disk_rout.cgs.value
        gamma = self.gamma
        h0 = self.h0.cgs.value
        beta = self.beta

        sigma0 = (2.0 - gamma) * mass / (2.0 * np.pi * rout**2)
        sigma = sigma0 * (r / rout) ** (-gamma) * np.exp(-((r / rout) ** (2.0 - gamma)))

        h = h0 * (r / (1 * u.au).cgs.value) ** beta

        rho = sigma / (np.sqrt(2 * np.pi) * h) * np.exp(-0.5 * (z / h) ** 2)
        return rho * u.g / u.cm**3
