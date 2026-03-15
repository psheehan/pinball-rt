from pinballrt.dust import load
from pinballrt.sources import Star
from pinballrt.grids import UniformCartesianGrid, UniformSphericalGrid, LogUniformSphericalGrid
from pinballrt.model import Model
from pinballrt.utils import calculate_Qvalue

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import torch
import os

import pytest

test_data = [
    (UniformCartesianGrid, {"ncells":9, "dx":2.0*u.au}, 98.0),
    (UniformSphericalGrid, {"ncells":9, "dr":2.0*u.au}, 93.0),
    (LogUniformSphericalGrid, {"ncells":9, "rmin":0.1*u.au, "rmax":20.0*u.au}, 73.0),
]

@pytest.mark.parametrize("grid_class,grid_kwargs,percentile", test_data)
def test_E2E(grid_class, grid_kwargs, percentile, return_vals=False):
    """
    Test the end-to-end functionality of the UniformCartesianGrid model running all the way through.
    """

    # Set up the dust.

    d = os.path.join(os.path.dirname(__file__), "data/yso.dst")

    # Set up the star.

    star = Star()
    star.set_blackbody_spectrum()

    # Set up the grid.
    model = Model(grid=grid_class, grid_kwargs=grid_kwargs)

    density = np.ones(model.grid.shape)*1.0e-16 * u.g / u.cm**3

    model.add_density(density, d)
    model.add_star(star)

    model.thermal_mc(nphotons=100000, use_ml_step=False, Qthresh=1.045, Delthresh=1.02)

    image = model.make_image(npix=256, pixel_size=0.2*u.arcsec, lam=np.array([1., 1000.])*u.micron, incl=45.*u.degree, pa=45.*u.degree, distance=1.*u.pc, nphotons=1000000)

    # Do the checks.

    if not return_vals:
        # Load the comparison data.
        temperature = np.load(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_temperature.npz"))['temperature']
        Q = calculate_Qvalue(temperature, model.grid.grid.temperature.numpy(), percentile=99.0)
        assert Q < 1.045, f"Temperature difference exceeds tolerance: {Q}"

        scattering = np.load(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_scattering.npz"))['scattering']
        Q = calculate_Qvalue(scattering, model.grid.scattering.numpy(), percentile=percentile, clip=0.1)
        assert Q < 1.15, f"scattering difference exceeds tolerance: {Q}"

        base_image = xr.open_dataset(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_image.nc"))
        Q = calculate_Qvalue(image.intensity, base_image.intensity, percentile=99.0, clip=0.1)
        assert Q < 1.025, f"Image difference exceeds tolerance: {Q}"
    else:
        return model.grid.grid.temperature.numpy(), model.grid.scattering.numpy(), image

def update_test(test, grid_class):
    found = False
    for data in test_data:
        if data[0] == grid_class:
            found = True
            break

    if not found:
        raise ValueError(f"Grid class {grid_class} not found in test data. Please add it to the test_data list in test_E2E.py.")
    
    temperature, scattering, image = test(grid_class, data[1], data[2], return_vals=True)

    np.savez(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_temperature.npz"), temperature=temperature)
    np.savez(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_scattering.npz"), scattering=scattering)
    image.to_netcdf(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_image.nc"))