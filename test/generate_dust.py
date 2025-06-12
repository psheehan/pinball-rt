import sys
sys.path.append("../")
from pinball.dust import Dust
import numpy as np
import astropy.units as u

data = np.loadtxt('dustkappa_yso.inp', skiprows=2)

lam = data[::-1,0].copy() * u.micron
kabs = data[::-1,1].copy() * u.cm**2 / u.g
ksca = data[::-1,2].copy() * u.cm**2 / u.g

d = Dust(lam, kabs, ksca)

d.learn_random_nu(plot=True)

d.write("dust.pkl")