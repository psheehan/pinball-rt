from pinballrt.sources import BlackbodyStar, DiffuseSource, EnergySource, ExternalSource
from pinballrt.grids import UniformCartesianGrid, UniformSphericalGrid, LogUniformSphericalGrid
from pinballrt.model import Model

import astropy.units as u
from astropy.modeling import models
import numpy as np
import os

import dill

import pytest

test_data = [
    (UniformCartesianGrid, {"ncells":9, "dx":2.0*u.au}, 98.0),
    (UniformSphericalGrid, {"ncells":9, "dr":2.0*u.au}, 93.0),
    (LogUniformSphericalGrid, {"ncells":9, "rmin":0.1*u.au, "rmax":20.0*u.au}, 73.0),
]

@pytest.mark.parametrize("grid_class,grid_kwargs,percentile", test_data)
def test_grid_pickle(grid_class, grid_kwargs, percentile, return_vals=False):
    """
    Test the end-to-end functionality of the UniformCartesianGrid model running all the way through.
    """

    # Set up the dust.

    d = os.path.join(os.path.dirname(__file__), "data/diana_wice.dst")

    # Set up the grid.
    model = Model(grid=grid_class, grid_kwargs=grid_kwargs)

    density = np.ones(model.grid.shape)*1.0e-16 * u.g / u.cm**3
    amax = np.ones(model.grid.shape) * u.cm
    if isinstance(model.grid, UniformCartesianGrid):
        amax[4, 4, 4] = 1.0 * u.micron
    else:
        amax[0, :, :] = 1.0 * u.micron

    model.set_physical_properties(density=density, dust=d, amax=amax, p=3.5)
    model.add_sources([BlackbodyStar(),
                       DiffuseSource(model.grid, lambda nu: 4*np.pi**2 * u.steradian * (0.035*u.R_sun)**2 * models.BlackBody(2000.*u.K)(nu), 10.*u.au**-3),
                       EnergySource(model.grid, 0.001*u.L_sun * u.au**-3), 
                       ExternalSource(model.grid, models.BlackBody(2.7*u.K))])

    result = dill.loads(dill.dumps(model.grid))
