from pinballrt.dust import IsotropicDust, HenyeyGreensteinDust, GeneralDust, load, suggest_opacity_sampling
import numpy as np
import astropy.units as u
import pytest
import dill
import os

def test_Dust():
    """
    Test the Dust class by creating a dust file from input opacity data.
    """

    data = np.load(os.path.join(os.path.dirname(__file__), "data/diana.npz"))

    p, amax = np.meshgrid(data["p"], data["amax"], indexing="ij")

    d = IsotropicDust(lam=data["lam"]*u.cm, 
                      kabs=data["kabs"]*u.cm**2/u.g, 
                      ksca=data["ksca"]*u.cm**2/u.g, 
                      amax=amax*u.cm, 
                      p=p)

    assert d.kmean.value == 4574.907442551958

    d.save("amax.dst")

@pytest.mark.parametrize(
    "dust_type,dims",
    [
        pytest.param(IsotropicDust, (), id="IsotropicDust/None"),
        pytest.param(IsotropicDust, ("amax","abundances"), id="IsotropicDust/amax,abundances"),
        pytest.param(IsotropicDust, ("p","amax","abundances"), id="IsotropicDust/p,amax,abundances"),
        pytest.param(HenyeyGreensteinDust, (), id="HenyeyGreensteinDust/None"),
        pytest.param(HenyeyGreensteinDust, ("p","abundances"), id="HenyeyGreensteinDust/p,abundances"),
        pytest.param(HenyeyGreensteinDust, ("p","amax","abundances"), id="HenyeyGreensteinDust/p,amax,abundances"),
        pytest.param(GeneralDust, (), id="GeneralDust/None"),
        pytest.param(GeneralDust, ("p","abundances"), id="GeneralDust/p,abundances"),
        pytest.param(GeneralDust, ("p","amax","abundances"), id="GeneralDust/p,amax,abundances"),
    ]
)
def test_learning(dust_type, dims):
    """
    Test the learn_random_nu method of the Dust class.
    """
    import matplotlib
    matplotlib.use("Agg")
    
    wavelengths = np.logspace(-1, 4, 10) * u.micron

    if "p" in dims:
        p_range = (2.5, 4.5)
    else:
        p_range = None

    if "amax" in dims:
        amax_range = (1*u.micron, 10*u.cm)
    else:
        amax_range = None

    if "abundances" in dims:
        n_dust_subspecies = 3
    else:
        n_dust_subspecies = 1

    samples = suggest_opacity_sampling(10 if len(dims) > 0 else 0, p_range=p_range, amax_range=amax_range, n_dust_subspecies=n_dust_subspecies)
    samples = np.moveaxis(np.repeat(np.expand_dims(samples, 1), wavelengths.size, axis=1), -1, 0)

    index = 0
    if "p" in dims:
        p = samples[index]
        index += 1
    else:
        p = 3.5

    if "amax" in dims:
        amax = samples[index] * u.cm
        index += 1
    else:
        amax = 1.0*u.micron

    if "abundances" in dims:
        silicate_fraction = samples[index]
        index += 1
        water_fraction = samples[index]
    else:
        silicate_fraction = 0.6
        water_fraction = 0.3

    wavelengths = np.repeat(np.expand_dims(wavelengths, axis=0), max(samples.shape[1], 1), axis=0)

    # Define the absorption and scattering opacities (in cm^2/g).
    silicate_feature = 100**silicate_fraction * np.exp(-0.5*(wavelengths - 10*u.micron)**2 / (10.*u.micron)**2) * u.cm**2 / u.g
    water_feature = 100**water_fraction * np.exp(-0.5*(wavelengths - 3.*u.micron)**2 / (2.*u.micron)**2) * u.cm**2 / u.g

    power_law_index = (-2. / (1 + np.exp(-p/3.5 * (-1.5 - np.log10(amax.to(u.cm).value)))))

    kappa_abs = 1.0 * (wavelengths.to(u.micron).value/100.0)**power_law_index * u.cm**2 / u.g + silicate_feature + water_feature
    kappa_scat = 0.5 * (wavelengths.to(u.micron).value/100.0)**power_law_index * u.cm**2 / u.g + silicate_feature + water_feature

    # Create the Dust object.
    dust_kwargs = {"lam":wavelengths[0,:], 
                   "amax":amax[:,0] if "amax" in dims else None, 
                   "p":p[:,0] if "p" in dims else None, 
                   "abundances":(silicate_fraction[:,0], water_fraction[:,0]) if "abundances" in dims else (),
                   "kabs":kappa_abs, 
                   "ksca":kappa_scat,
                   "ntemperatures":10}

    if dust_type == HenyeyGreensteinDust:
        g = np.tanh(p - np.log10(wavelengths.to(u.micron).value))
        dust_kwargs["g"] = g
    elif dust_type == GeneralDust:
        g = np.repeat(np.expand_dims(np.tanh(p - np.log10(wavelengths.to(u.micron).value)), axis=-1), 5, axis=-1)
        theta = np.tile(np.expand_dims(np.linspace(0, 180., 5), axis=(0,1)), (10 if len(dims) > 0 else 1, 10, 1)) * u.deg
        scattering_phase_function = (1 - g**2) / (4 * np.pi * (1 + g**2 - 2*g*np.cos(theta.to(u.rad).value))**(3/2))

        dust_kwargs["scattering_phase_function"] = scattering_phase_function
        dust_kwargs["theta"] = theta[0,0,:]

    d = dust_type(**dust_kwargs)

    # Test the learn_random_nu method.

    models = ["kabs","ksca","pmo","random_nu"]
    if dust_type == HenyeyGreensteinDust:
        models.append("g")
    if dust_type == GeneralDust:
        models.append("scattering_phase_function")
        models.append("random_direction")

    for model in models:
        print('*****************************')
        print(f'{model}')
        print('*****************************')
        d.learn(model=model, overwrite=True)
        if model == "random_nu":
            size = d.kabs.size * d.temperature.size
        elif model == "pmo":
            size = d.kabs.shape[0] * d.temperature.size
        else:
            size = d.kabs.size
        batch_size = int(10.**int(np.log10(size / 10)))
        d.fit(epochs=10, batch_size=batch_size)
        d.test_model(plot=True)

        if model == "random_nu":
            d.plot_random_nu_model(100)

    # Test the learn_ml_step method.

    n_samples = 100
    d.learn(model="ml_step", 
            hidden_units=((32, 32, 32),)*6, 
            nsamples=n_samples, 
            tau_range=(0.5, 1.0), 
            nu_range=(d.nu.max()/10, d.nu.max()), 
            overwrite=True)
    d.fit(epochs=10)
    d.test_model(plot=True)

    os.system("rm -rf sim_results.csv *_logs")

def test_dust_pickle():
    """
    Test that Dust classes can be pickled and unpickled.
    """

    data = np.load(os.path.join(os.path.dirname(__file__), "data/diana.npz"))

    p, amax = np.meshgrid(data["p"], data["amax"], indexing="ij")

    d = IsotropicDust(lam=data["lam"]*u.cm, 
                      kabs=data["kabs"]*u.cm**2/u.g, 
                      ksca=data["ksca"]*u.cm**2/u.g, 
                      amax=amax*u.cm, 
                      p=p)

    result = dill.loads(dill.dumps(d))
