from pinball.dust import load
from pinball.sources import Star
from pinball.grids import UniformCartesianGrid, UniformSphericalGrid
from pinball.camera import Camera
from pinball.model import Model

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

import sys

# Set up the dust.

d = load("yso.dst")

# Set up the star.

star = Star()
star.set_blackbody_spectrum(d.nu)

# Set up the grid.

model = Model(grid=UniformCartesianGrid, ncells=9, dx=2.0*u.au)
#model = Model(grid=UniformSphericalGrid, ncells=9, dx=0.5*u.au, mirror=True)

density = np.ones(model.grid.shape)*1.0e-16 * u.g / u.cm**3

model.add_density(density, d)
model.add_star(star)

model.thermal_mc(nphotons=1000000)

for i in range(9):
    plt.imshow(model.grid.grid.temperature[:,:,i], vmin=model.grid.grid.temperature.numpy().min(), vmax=model.grid.grid.temperature.numpy().max())
    plt.colorbar()
    plt.savefig(f"temperature_{i}.png")
    plt.clf()
    plt.close()

image = model.make_image(256, 256, 0.1, np.array([1., 1000.])*u.micron, 45., 45., 1.)

for i in range(9):
    plt.imshow(model.grid.scattering[0,:,:,i], vmin=model.grid.scattering[0].min(), vmax=model.grid.scattering[0].max())
    plt.savefig(f"scattering_{i}.png")
    plt.clf()

plt.imshow(image.intensity[:,:,0])
plt.savefig("image.png")