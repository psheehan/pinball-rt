from pinballrt.dust import Dust, load, suggest_opacity_sampling
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

    d = Dust(lam=data["lam"]*u.cm, 
             kabs=data["kabs"]*u.cm**2/u.g, 
             ksca=data["ksca"]*u.cm**2/u.g, 
             amax=amax*u.cm, 
             p=p)

    assert d.kmean.value == 4574.907442551958

    d.save("amax.dst")

@pytest.mark.parametrize(
    "dims",
    [
        pytest.param((), id="None"),
        pytest.param(("amax","abundances"), id="amax,abundances"),
        pytest.param(("p","amax","abundances"), id="p,amax,abundances"),
    ]
)
def test_learning(dims):
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
    d = Dust(lam=wavelengths[0,:], 
             amax=amax[:,0] if "amax" in dims else None, 
             p=p[:,0] if "p" in dims else None, 
             abundances=(silicate_fraction[:,0], water_fraction[:,0]) if "abundances" in dims else (),
             kabs=kappa_abs, 
             ksca=kappa_scat,
             ntemperatures=10)

    # Test the learn_random_nu method.

    for model in ["kabs","ksca","pmo","random_nu"]:
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

    d = Dust(lam=data["lam"]*u.cm, 
             kabs=data["kabs"]*u.cm**2/u.g, 
             ksca=data["ksca"]*u.cm**2/u.g, 
             amax=amax*u.cm, 
             p=p)

    result = dill.loads(dill.dumps(d))


@pytest.mark.parametrize(
    ("include_p", "include_amax", "expected_n_features"),
    [
        pytest.param(False, False, 1, id="frequency-only"),
        pytest.param(True, False, 2, id="p-only"),
        pytest.param(False, True, 2, id="amax-only"),
        pytest.param(True, True, 3, id="p-and-amax"),
    ],
)
def test_ml_opacity_feature_cache(include_p, include_amax, expected_n_features):
    """Test that cached ML opacity features only include active Dust dims plus frequency."""
    import torch
    import warp as wp
    from pinballrt.photons import PhotonList

    wavelengths = np.logspace(-1, 4, 10) * u.micron
    p_vals = np.array([3.0, 3.2, 3.4, 3.6], dtype=np.float32)
    amax_vals = np.array([1e-4, 5e-4, 1e-3, 5e-3], dtype=np.float32) * u.cm
    nu_vals = np.array([1e10, 1e11, 1e12, 1e13], dtype=np.float32)

    kappa_abs = np.ones((p_vals.size, wavelengths.size)) * u.cm**2 / u.g
    kappa_scat = 0.5 * np.ones((p_vals.size, wavelengths.size)) * u.cm**2 / u.g

    d = Dust(lam=wavelengths,
             amax=amax_vals if include_amax else None,
             p=p_vals if include_p else None,
             abundances=(),
             kabs=kappa_abs,
             ksca=kappa_scat)

    assert d.ndims + 1 == expected_n_features

    p_torch = torch.tensor(p_vals, dtype=torch.float32) if include_p else None
    amax_torch = torch.tensor(amax_vals.value, dtype=torch.float32) if include_amax else None
    nu_torch = torch.tensor(nu_vals, dtype=torch.float32)

    samples_builtin = d._get_ml_opacity_samples(p=p_torch, amax=amax_torch, nu=nu_torch, abundances=None)
    assert samples_builtin.shape == (p_vals.size, expected_n_features)

    photon_list = PhotonList()
    photon_list.p = wp.array(p_vals, dtype=float)
    photon_list.amax = wp.array(amax_vals.value, dtype=float)
    photon_list.frequency = wp.array(nu_vals, dtype=float)
    photon_list.dust_abundances = wp.zeros((p_vals.size, 0), dtype=float)

    features_np = np.zeros((p_vals.size, expected_n_features), dtype=np.float32)
    feature_idx = 0
    if include_p:
        features_np[:, feature_idx] = p_vals
        feature_idx += 1
    if include_amax:
        features_np[:, feature_idx] = np.log10(amax_vals.value)
        feature_idx += 1
    features_np[:, feature_idx] = np.log10(nu_vals)

    photon_list.ml_opacity_features = wp.from_torch(torch.tensor(features_np, dtype=torch.float32))

    samples_cached = d._get_ml_opacity_samples(photon_list=photon_list)
    assert samples_cached.shape == (p_vals.size, expected_n_features)
    assert torch.allclose(samples_builtin, samples_cached, rtol=1e-5)


def test_ml_opacity_equivalence():
    """
    Test that ml_kabs returns same values using cached vs non-cached paths.
    """
    import torch
    from pinballrt.photons import PhotonList
    import warp as wp
    
    wavelengths = np.logspace(-1, 4, 10) * u.micron
    p_vals = np.array([3.0, 3.5, 4.0])
    amax_vals = np.array([1e-4, 1e-3, 1e-2]) * u.cm
    
    kappa_abs = np.ones((3, 10)) * u.cm**2 / u.g
    kappa_scat = 0.5 * np.ones((3, 10)) * u.cm**2 / u.g
    
    d = Dust(lam=wavelengths, 
             amax=amax_vals, 
             p=p_vals, 
             abundances=(),
             kabs=kappa_abs, 
             ksca=kappa_scat)
    
    # Train a simple model so ml_kabs is available
    d.learn(model="kabs", overwrite=True, nsamples=100, hidden_units=(16, 16))
    d.fit(epochs=2, batch_size=50)
    
    # Create a photon list without cache
    photon_list_no_cache = PhotonList()
    nphotons = 5
    photon_list_no_cache.p = wp.array(np.array([3.0, 3.2, 3.4, 3.6, 3.8]), dtype=float)
    photon_list_no_cache.amax = wp.array(np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2]), dtype=float)
    photon_list_no_cache.frequency = wp.array(np.array([1e10, 1e11, 1e12, 1e13, 1e14]), dtype=float)
    photon_list_no_cache.dust_abundances = wp.zeros((nphotons, 0), dtype=float)
    
    # Call ml_kabs without cache
    kabs_no_cache = d.ml_kabs(photon_list=photon_list_no_cache)
    
    # Create photon list with cache
    photon_list_with_cache = PhotonList()
    photon_list_with_cache.p = wp.array(np.array([3.0, 3.2, 3.4, 3.6, 3.8]), dtype=float)
    photon_list_with_cache.amax = wp.array(np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2]), dtype=float)
    photon_list_with_cache.frequency = wp.array(np.array([1e10, 1e11, 1e12, 1e13, 1e14]), dtype=float)
    photon_list_with_cache.dust_abundances = wp.zeros((nphotons, 0), dtype=float)
    
    # Fill cache
    n_features = d.ndims + 1
    photon_list_with_cache.ml_opacity_features = wp.zeros((nphotons, n_features), dtype=float)
    features_np = np.zeros((nphotons, n_features))
    features_np[:, 0] = np.array([3.0, 3.2, 3.4, 3.6, 3.8])
    features_np[:, 1] = np.log10(np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2]))
    features_np[:, 2] = np.log10(np.array([1e10, 1e11, 1e12, 1e13, 1e14]))
    photon_list_with_cache.ml_opacity_features = wp.from_torch(torch.tensor(features_np, dtype=torch.float32))
    
    # Call ml_kabs with cache
    kabs_with_cache = d.ml_kabs(photon_list=photon_list_with_cache)
    
    # Results should match
    assert torch.allclose(kabs_no_cache, kabs_with_cache, rtol=1e-5), \
        f"ml_kabs outputs differ: no_cache={kabs_no_cache}, with_cache={kabs_with_cache}"


def test_ml_random_nu_cached_subset_features():
    """Test cached random_nu feature gathering for subset updates and in-kernel ksi bounds."""
    import torch
    import warp as wp
    from pinballrt.photons import PhotonList

    wavelengths = np.logspace(-1, 4, 10) * u.micron
    p_vals = np.array([3.0, 3.2, 3.4, 3.6], dtype=np.float32)
    amax_vals = np.array([1e-4, 5e-4, 1e-3, 5e-3], dtype=np.float32) * u.cm
    temp_vals = np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float32)

    kappa_abs = np.ones((p_vals.size, wavelengths.size)) * u.cm**2 / u.g
    kappa_scat = 0.5 * np.ones((p_vals.size, wavelengths.size)) * u.cm**2 / u.g

    d = Dust(lam=wavelengths,
             amax=amax_vals,
             p=p_vals,
             abundances=(),
             kabs=kappa_abs,
             ksca=kappa_scat)

    photon_list = PhotonList()
    nphotons = p_vals.size
    photon_list.p = wp.array(p_vals, dtype=float)
    photon_list.amax = wp.array(amax_vals.value, dtype=float)
    photon_list.temperature = wp.array(temp_vals, dtype=float)
    photon_list.frequency = wp.zeros(nphotons, dtype=float)
    photon_list.dust_abundances = wp.zeros((nphotons, 0), dtype=float)

    # Cache is allocated at ndims + 2 so random_nu mode can write [log10(T), ksi].
    photon_list.ml_opacity_features = wp.zeros((nphotons, d.ndims + 2), dtype=float)

    subset = np.array([1, 3], dtype=np.int32)
    opacity_update_indices = wp.array(subset, dtype=int)

    samples = d._get_ml_opacity_samples(
        photon_list=photon_list,
        opacity_update_indices=opacity_update_indices,
        n_cached_samples=subset.size,
        sample_mode="random_nu",
    )

    assert samples.shape == (subset.size, d.ndims + 2)

    expected_prefix = torch.tensor(
        np.column_stack([
            p_vals[subset],
            np.log10(amax_vals.value[subset]),
            np.log10(temp_vals[subset]),
        ]),
        dtype=torch.float32,
    )
    assert torch.allclose(samples[:, :d.ndims + 1], expected_prefix, rtol=1e-5)

    ksi = samples[:, -1]
    assert torch.all(torch.isfinite(ksi))
    assert torch.all(ksi >= -8.6643)
    assert torch.all(ksi <= 8.6643)

