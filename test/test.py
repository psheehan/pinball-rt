import sys
sys.path.append("../")
from pinball.dust import Dust
from pinball.sources import Star
from pinball.grids import UniformCartesianGrid, UniformSphericalGrid
from pinball.camera import Camera

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Set up the dust.

data = np.loadtxt('dustkappa_yso.inp', skiprows=2)

lam = data[::-1,0].copy() * u.micron
kabs = data[::-1,1].copy() * u.cm**2 / u.g
ksca = data[::-1,2].copy() * u.cm**2 / u.g

d = Dust(lam, kabs, ksca)

d.learn_random_nu()

# Set up the star.

star = Star()
star.set_blackbody_spectrum(d.nu)

# Set up the grid.

grid = UniformCartesianGrid(ncells=9, dx=1.0*u.au)
#grid = UniformSphericalGrid(ncells=9, dr=0.5*u.au, mirror=True)

density = np.ones(grid.shape)*1.0e-16 * u.g / u.cm**3

grid.add_density(density, d)
grid.add_star(star)

grid.thermal_mc(nphotons=1000000)

for i in range(9):
    plt.imshow(grid.grid.temperature[:,:,i], vmin=grid.grid.temperature.numpy().min(), vmax=grid.grid.temperature.numpy().max())
    plt.colorbar()
    plt.savefig(f"temperature_{i}.png")
    plt.clf()
    plt.close()

grid.scattering_mc(100000, np.array([1.0,2.0])*u.micron)

for i in range(9):
    plt.imshow(grid.scattering[0,:,:,i], vmin=grid.scattering[0].min(), vmax=grid.scattering[0].max())
    plt.savefig(f"scattering_{i}.png")
    plt.clf()

camera = Camera(grid)
image = camera.make_image(256, 256, 0.1, np.array([1., 1000.])*u.micron, 45., 45., 1.)

plt.imshow(image.intensity[:,:,0])
plt.savefig("image.png")
