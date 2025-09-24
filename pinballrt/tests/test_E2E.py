from pinballrt.dust import load
from pinballrt.sources import Star
from pinballrt.grids import UniformCartesianGrid, UniformSphericalGrid, LogUniformSphericalGrid
from pinballrt.model import Model

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os

import pytest

test_data = [
    (UniformCartesianGrid, {"ncells":9, "dx":2.0*u.au}),
    (UniformSphericalGrid, {"ncells":9, "dr":2.0*u.au}),
    (LogUniformSphericalGrid, {"ncells":9, "rmin":0.1*u.au, "rmax":20.0*u.au}),
]

@pytest.mark.parametrize("grid_class,grid_kwargs", test_data)
def test_E2E(grid_class, grid_kwargs, return_vals=False):
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

    model.thermal_mc(nphotons=1000000, use_ml_step=False)

    image = model.make_image(npix=256, pixel_size=0.1*u.arcsec, lam=np.array([1., 1000.])*u.micron, incl=45.*u.degree, pa=45.*u.degree, distance=1.*u.pc)

    # Do the checks.

    if not return_vals:
        # Load the comparison data.

        temperature_mean = np.load(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_temperature.npz"))['temperature_mean']
        temperature_std = np.load(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_temperature.npz"))['temperature_std']
        scattering_mean = np.load(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_scattering.npz"))['scattering_mean']
        scattering_std = np.load(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_scattering.npz"))['scattering_std']
        base_image = xr.open_dataset(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_image.nc"))

        temperature_diff = (model.grid.grid.temperature.numpy() - temperature_mean) / temperature_std
        assert np.all(np.abs(temperature_diff) <= 3.), f"Temperature difference exceeds tolerance: {np.max(np.abs(temperature_diff))}"

        scattering_diff = (model.grid.scattering.numpy() - scattering_mean) / np.where(scattering_std > 0, scattering_std, 1.0)
        assert np.all(np.abs(scattering_diff) < 3.), f"Scattering difference exceeds tolerance: {np.max(np.abs(scattering_diff))}"

        image_diff = (image.intensity - base_image.intensity) / np.where(base_image.unc > 0, base_image.unc, 1.0)
        assert np.all(np.abs(image_diff) < 3.), f"Image difference exceeds tolerance: {np.max(np.abs(image_diff))}"
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
    
    result = [test(grid_class, data[1], return_vals=True) for i in range(5)]

    temperature_list = [r[0] for r in result]
    scattering_list = [r[1] for r in result]
    image_list = [r[2] for r in result]

    temperature_mean = np.mean(temperature_list, axis=0)
    temperature_std = np.std(temperature_list, axis=0)
    scattering_mean = np.mean(scattering_list, axis=0)
    scattering_std = np.std(scattering_list, axis=0)
    image_mean = xr.concat(image_list, dim="iteration").mean(dim="iteration")
    image_std = xr.concat(image_list, dim="iteration").std(dim="iteration")

    image = image_mean.assign(unc=image_std.intensity)

    np.savez(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_temperature.npz"), temperature_mean=temperature_mean, temperature_std=temperature_std)
    np.savez(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_scattering.npz"), scattering_mean=scattering_mean, scattering_std=scattering_std)
    image.to_netcdf(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_image.nc"))
