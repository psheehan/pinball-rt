import astropy.units as u
import numpy as np
import pandas as pd

from .ModelComponent import ModelComponent
from .Parameters import Parameter, ParameterSet


class Disk(ModelComponent):
    name = "Disk"
    default_parameters = ParameterSet(
        [
            Parameter(
                name="disk_mass",
                value=-3.0,
                unit=u.Msun,
                prior_range=(-5.0, 0.0),
                fixed=False,
                log=True,
                component=name,
            ),
            Parameter(
                name="disk_rin",
                value=-1.0,
                unit=u.au,
                prior_range=(-5.0, 0.0),
                fixed=False,
                log=True,
                component=name,
            ),
            Parameter(
                name="disk_rout",
                value=2.0,
                unit=u.au,
                prior_range=(1.0, 3.0),
                fixed=False,
                log=True,
                component=name,
            ),
            Parameter(
                name="h0",
                value=0.05,
                unit=u.au,
                prior_range=(0.01, 0.3),
                fixed=False,
                component=name,
            ),
            Parameter(
                name="gamma",
                value=1.0,
                prior_range=(0.0, 2.0),
                fixed=False,
                component=name,
            ),
            Parameter(
                name="beta",
                value=1.0,
                prior_range=(0.5, 1.5),
                fixed=False,
                component=name,
            ),
        ]
    )

    density_coordinates = "cylindrical"

    def density(self, r, z):
        mass = self.disk_mass.cgs.value
        rout = self.disk_rout.cgs.value
        gamma = self.gamma
        h0 = self.h0.cgs.value
        beta = self.beta

        sigma0 = (2.0 - gamma) * mass / (2.0 * np.pi * rout**2)
        sigma = sigma0 * (r / rout) ** (-gamma) * np.exp(-((r / rout) ** (2.0 - gamma)))

        h = h0 * (r / (1 * u.au).cgs.value) ** beta

        rho = sigma / (np.sqrt(2 * np.pi) * h) * np.exp(-0.5 * (z / h) ** 2)
        return rho * u.g / u.cm**3
