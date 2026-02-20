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

    for model in ["kabs", "ksca", "pmo", "random_nu"]:
        d.learn(model=model, nsamples=n_samples, overwrite=True)
        d.fit(epochs=10)
        d.test_model(plot=True)

def test_learn_ml_step():
    """
    Test the learn_ml_step method of the Dust class.
    """
    # Load the dust file.

    d = load(os.path.join(os.path.dirname(__file__), "data/yso.dst"))

    # Run a simple dust simulation to test the machinery.

    d.run_dust_simulation(nphotons=100, tau_range=(0.5, 1.0), nu_range=(d.nu.max()/10, d.nu.max()), use_ml_step=False)

    # Copy the pre-existing sim_results.csv to the current directory for training.

    os.system(f"cp {os.path.join(os.path.dirname(__file__), 'data/sim_results.csv')} sim_results.csv")

    # Test the learn_ml_step method.

    n_samples = 1000
    d.learn(model="ml_step", nsamples=n_samples)
    d.fit(epochs=10)
    d.test_model(plot=True)
