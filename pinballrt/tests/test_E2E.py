from pinballrt.sources import BlackbodyStar, DiffuseSource, EnergySource, ExternalSource
from pinballrt.grids import UniformCartesianGrid, UniformSphericalGrid, LogUniformSphericalGrid
from pinballrt.model import Model
from pinballrt.utils import calculate_Qvalue
from pinballrt.gas import Gas

import astropy.units as u
from astropy.modeling import models
import numpy as np
import xarray as xr
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

    d = os.path.join(os.path.dirname(__file__), "data/diana_wice.dst")

    # Set up the grid.
    model = Model(grid=grid_class, grid_kwargs=grid_kwargs)

    density = np.ones(model.grid.shape)*1.0e-14 * u.g / u.cm**3
    amax = np.ones(model.grid.shape) * u.cm
    if isinstance(model.grid, UniformCartesianGrid):
        amax[4, 4, 4] = 1.0 * u.micron
    else:
        amax[0, :, :] = 1.0 * u.micron

    if isinstance(model.grid, UniformCartesianGrid):
        vx, vy, vz = np.meshgrid(0.5*(model.grid.grid.w1.numpy()[1:] + model.grid.grid.w1.numpy()[0:-1]), 
                                 0.5*(model.grid.grid.w2.numpy()[1:] + model.grid.grid.w2.numpy()[0:-1]), 
                                 0.5*(model.grid.grid.w3.numpy()[1:] + model.grid.grid.w3.numpy()[0:-1]), indexing='ij')
    else:
        vx, vy, vz = np.meshgrid(0.5*(model.grid.grid.w1.numpy()[1:] + model.grid.grid.w1.numpy()[0:-1]), 
                                 np.zeros(model.grid.grid.n2),
                                 np.zeros(model.grid.grid.n3), indexing='ij')
    velocity = np.concatenate((vx[np.newaxis], vy[np.newaxis], vz[np.newaxis]), axis=0) * (-1.0 * u.km / u.s)

    model.set_physical_properties(density=density, dust=d, amax=amax, p=3.5, gases=[os.path.join(os.path.dirname(__file__), "data/co.dat")], 
                                  abundances=[1.0e-4], microturbulence=0.2 * u.km / u.s, velocity=velocity)
    model.add_sources([BlackbodyStar(),
                       DiffuseSource(model.grid, lambda nu: 4*np.pi**2 * u.steradian * (0.035*u.R_sun)**2 * models.BlackBody(2000.*u.K)(nu), 10.*u.au**-3),
                       EnergySource(model.grid, 0.001*u.L_sun * u.au**-3), 
                       ExternalSource(model.grid, models.BlackBody(2.7*u.K))])

    model.thermal_mc(nphotons=200000, use_ml_step=False, Qthresh=1.045, Delthresh=1.02)

    image = model.make_image(npix=256, pixel_size=0.2*u.arcsec, channels=np.array([1., 1000.])*u.micron, 
                             incl=45.*u.degree, pa=45.*u.degree, distance=1.*u.pc, nphotons=1000000, include_gas=False)

    g = Gas()
    g.set_properties_from_lambda('co.dat')

    cube = model.make_image(npix=256, pixel_size=0.2*u.arcsec, channels=np.linspace(-20., 20., 300)*u.km/u.s, rest_frequency=g.nu[2], 
                            incl=45.*u.degree, pa=45.*u.degree, distance=1.*u.pc, include_dust=False, device='cpu')
    mom0 = cube.sum(dim='lam')

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

        base_mom0 = xr.open_dataset(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_mom0.nc"))
        Q = calculate_Qvalue(mom0.intensity, base_mom0.intensity, percentile=99.0, clip=0.1)
        assert Q < 1.025, f"Mom0 difference exceeds tolerance: {Q}"
    else:
        return model.grid.grid.temperature.numpy(), model.grid.scattering.numpy(), image, mom0

def update_test(test, grid_class):
    found = False
    for data in test_data:
        if data[0] == grid_class:
            found = True
            break

    if not found:
        raise ValueError(f"Grid class {grid_class} not found in test data. Please add it to the test_data list in test_E2E.py.")
    
    temperature, scattering, image, mom0 = test(grid_class, data[1], data[2], return_vals=True)

    np.savez(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_temperature.npz"), temperature=temperature)
    np.savez(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_scattering.npz"), scattering=scattering)
    image.to_netcdf(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_image.nc"))
    mom0.to_netcdf(os.path.join(os.path.dirname(__file__), f"data/{grid_class.__name__}_E2E_mom0.nc"))
