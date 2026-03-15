from pinballrt.dust import Dust, load
import numpy as np
import astropy.units as u
import os

def test_Dust():
    """
    Test the Dust class by creating a dust file from input opacity data.
    """
    # Load the opacity data from a file.
    # The file 'dustkappa_yso.inp' should be in the same directory as this test script.
    # The format of the file is assumed to be:
    # First two lines are headers, then three columns: wavelength (micron), kabs (cm^2/g), ksca (cm^2/g)

    data = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/dustkappa_yso.inp"), skiprows=2)

    lam = data[::-1,0].copy() * u.micron
    kabs = data[::-1,1].copy() * u.cm**2 / u.g
    ksca = data[::-1,2].copy() * u.cm**2 / u.g

    d = Dust(lam, kabs, ksca)

    assert np.abs(d.kmean.value - 2831.6281816798232) < 1e-10

    d.save("tmp.dst")

def test_learn_random_nu():
    """
    Test the learn_random_nu method of the Dust class.
    """

    d = load(os.path.join(os.path.dirname(__file__), "data/yso.dst"))

    # Test the learn_random_nu method.

    n_samples = 1000
    d.learn(model="random_nu", nsamples=n_samples)
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
