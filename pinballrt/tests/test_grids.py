from pinballrt.sources import BlackbodyStar, DiffuseSource, EnergySource, ExternalSource
from pinballrt.grids import UniformCartesianGrid, UniformSphericalGrid, LogUniformSphericalGrid
from pinballrt.model import Model
from pinballrt.dust import load
import warp as wp
import torch

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

    d = os.path.join(os.path.dirname(__file__), "data/diana.iso.dst")

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
                       DiffuseSource(lambda nu: 4*np.pi**2 * u.steradian * (0.035*u.R_sun)**2 * models.BlackBody(2000.*u.K)(nu), 10.*u.au**-3),
                       EnergySource(0.001*u.L_sun * u.au**-3), 
                       ExternalSource(models.BlackBody(2.7*u.K))])

    result = dill.loads(dill.dumps(model.grid))

def test_grid_physical_properties_shapes():
    """
    Test that the grid shapes are correct.
    """

    # Set up the dust.

    d = os.path.join(os.path.dirname(__file__), "data/diana.iso.dst")

    # Set up the grid.
    model = Model(grid=UniformCartesianGrid, grid_kwargs={"ncells":9, "dx":2.0*u.au})

    density = np.ones(model.grid.shape)*1.0e-16 * u.g / u.cm**3
    
    model.set_physical_properties(density=density, amax=100*u.micron, p=3.5, dust=d)
    
    assert model.grid.grid.dust_density.numpy().shape == model.grid.shape
    assert model.grid.grid.amax.numpy().shape == model.grid.shape
    assert model.grid.grid.p.numpy().shape == model.grid.shape

    model.set_physical_properties(density=density, amax=100, p=3.5, dust=d)

    assert model.grid.grid.dust_density.numpy().shape == model.grid.shape
    assert model.grid.grid.amax.numpy().shape == model.grid.shape
    assert model.grid.grid.p.numpy().shape == model.grid.shape

def test_grid_default_physical_properties():
    """
    Test that the grid default physical properties are correct when unspecified.
    """

    # Set up the dust and gas.

    d = load(os.path.join(os.path.dirname(__file__), "data/diana.iso.dst"))

    # Set up the grid.
    model = Model(grid=UniformCartesianGrid, grid_kwargs={"ncells":9, "dx":2.0*u.au})

    with pytest.raises(ValueError):
        model.grid_list["cpu"].check_physical_properties(include_dust=True, include_gas=True)

    density = np.ones(model.grid.shape)*1.0e-16 * u.g / u.cm**3
    with pytest.raises(ValueError):
        model.set_physical_properties(density=density)
    
    model.set_physical_properties(dust=d)

    with pytest.raises(ValueError):
        model.grid_list["cpu"].check_physical_properties(include_dust=True, include_gas=True)

    density = np.ones(model.grid.shape)*1.0e-16 * u.g / u.cm**3
    model.set_physical_properties(density=density)

    with pytest.raises(ValueError):
        model.grid_list["cpu"].check_physical_properties(include_dust=True, include_gas=True)

    model.set_physical_properties(gases=[os.path.join(os.path.dirname(__file__), "data/co.dat")])
    
    model.grid_list["cpu"].check_physical_properties(include_dust=True, include_gas=True)

    assert np.all(np.abs(model.grid_list["cpu"].grid.amax.numpy() - d.fiducial_values["amax"].to(u.cm).value) < 1e-7)
    assert np.all(np.abs(model.grid_list["cpu"].grid.p.numpy() - d.fiducial_values["p"]) < 1e-7)
    if "abundances" in d.fiducial_values:
        assert np.all([np.all(np.abs(model.grid_list["cpu"].grid.dust_abundances.numpy()[i] - d.fiducial_values["abundances"][i]) < 1e-7) for i in range(len(d.fiducial_values["abundances"]))])

    assert np.all(model.grid_list["cpu"].grid.velocity.numpy() == 0.0)
    assert np.all(model.grid_list["cpu"].grid.microturbulence.numpy() == 0.0)

def test_deposit_energy():
    star = BlackbodyStar()
    
    # Set up the grid.

    for i in range(2):
        grid = UniformSphericalGrid(ncells=1, dr=1.0*u.au, mirror=False, device="cpu")

        d = load(os.path.join(os.path.dirname(__file__), "data/diana_wice.dst"))

        density = np.ones(grid.shape) * 1e-17 * u.g / u.cm**3

        grid.set_physical_properties(density=density, amax=1.0*u.micron, p=3.5, dust=d)
        grid.check_physical_properties(include_dust=True, include_gas=False)
        grid.add_sources(star)

        grid.grid.temperature.numpy()[0,0,0] = 150.

        # Emit the photons

        nphotons = 10000

        photon_list = grid.emit(nphotons, wavelength="random", scattering=False)

        with wp.ScopedDevice(grid.device):
            initial_direction = np.zeros((nphotons, 3), dtype=np.float32)
            initial_direction[:,0] = 1.
            photon_list.direction = wp.array(initial_direction, dtype=wp.vec3)

            photon_list.temperature = wp.array(np.repeat(150., nphotons), dtype=float)

            photon_list.amax = wp.array(np.repeat((1.0*u.micron).to(u.cm), nphotons), dtype=float)
            photon_list.p = wp.array(np.repeat(3.5, nphotons), dtype=float)
            if len(d.abundances) > 0:
                photon_list.dust_abundances = wp.array2d(np.repeat(d.abundances[np.newaxis, :], nphotons, axis=0), dtype=float)

            tau = np.repeat(150., nphotons)
            photon_list.density = wp.array((tau / (d.ml_planck_mean_opacity(wp.to_torch(photon_list.p), 
                                                                            wp.to_torch(photon_list.amax), 
                                                                            wp.to_torch(photon_list.temperature)) * d.kmean.unit * \
                                                                                1.*u.au) * d.kmean).to(1 / u.au), dtype=float)

            grid.grid.dust_density.numpy()[0,0,0] = photon_list.density.numpy()[0]

            photon_list.frequency = star.random_nu(nphotons)

            grid.grid.energy.numpy()[0,0,0] = 0.

        if i == 0:
            grid.propagate_photons(photon_list, learning=True, use_ml_step=False)
            photon_accumulated_energy = np.sum(photon_list.deposited_energy.numpy())
        else:
            grid.propagate_photons(photon_list, learning=False, use_ml_step=False)
            grid_accumulated_energy = grid.grid.energy.numpy()[0,0,0]

    print("Photon accumulated energy:", photon_accumulated_energy)
    print("Grid accumulated energy:", grid_accumulated_energy)

    assert np.isclose(photon_accumulated_energy, grid_accumulated_energy, rtol=0.02)
