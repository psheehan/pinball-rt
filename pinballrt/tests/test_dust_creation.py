from pinballrt.dust import Dust, load
import numpy as np
import astropy.units as u
import os

def test_Dust():
    """
    Test the Dust class by creating a dust file from input opacity data.
    """

    data = np.load(os.path.join(os.path.dirname(__file__), "data/diana_wice.npy.npz"))

    d = Dust(lam=data["lam"]*u.cm, kabs=data["kabs"]*u.cm**2/u.g, ksca=data["ksca"]*u.cm**2/u.g, amax=data["amax"]*u.cm, p=data["p"], interpolate=0)

    assert d.kmean.value == 2206.6072

    d.save("amax.dst")

def test_learning():
    """
    Test the learn_random_nu method of the Dust class.
    """

    d = load(os.path.join(os.path.dirname(__file__), 'data/amax.dst'))

    # Test the learn_random_nu method.

    n_samples = 1000

    d.learn(model="kabs", nsamples=n_samples, max_epochs=10)
    d.learn(model="ksca", nsamples=n_samples, max_epochs=10, overwrite=True)
    d.learn(model="pmo", nsamples=n_samples, max_epochs=10, overwrite=True)

    d.learn(model="random_nu", nsamples=n_samples, max_epochs=10, overwrite=True)
