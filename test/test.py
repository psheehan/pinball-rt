import sys
sys.path.append("../")
from pinball.dust import Dust
from pinball.sources import Star
from pinball.grids import CartesianGrid
from pinball.camera import Camera

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

nphotons = 1000000

n1, n2, n3 = 9, 9, 9
w1 = (np.linspace(-4.5, 4.5, n1+1)*u.au).cgs.value
w2 = (np.linspace(-4.5, 4.5, n2+1)*u.au).cgs.value
w3 = (np.linspace(-4.5, 4.5, n3+1)*u.au).cgs.value

density = np.ones((n1,n2,n3))*1.0e-16

# Set up the dust.

data = np.loadtxt('dustkappa_yso.inp', skiprows=2)

lam = data[::-1,0].copy() * 1.0e-4
kabs = data[::-1,1].copy()
ksca = data[::-1,2].copy()

d = Dust(lam, kabs, ksca)

d.learn_random_nu()

# Set up the star.

star = Star()
star.set_blackbody_spectrum(d.nu)

# Set up the grid.

grid = CartesianGrid(w1, w2, w3)
grid.add_density(density, d)
grid.add_star(star)

#grid.thermal_mc(nphotons)

for i in range(9):
    plt.imshow(grid.temperature[:,:,i])
    plt.savefig(f"temperature_{i}.png")

grid.scattering_mc(100000, np.array([1.0,2.0]))

camera = Camera(grid)
image = camera.make_image(256, 256, 0.1, np.array([1., 1000.]), 45., 45., 1.)

print(grid.scattering.max())

for i in range(9):
    plt.imshow(grid.scattering[0,:,:,i])
    plt.savefig(f"scattering_{i}.png")

plt.imshow(image.intensity[:,:,0])
plt.savefig("image.png")
