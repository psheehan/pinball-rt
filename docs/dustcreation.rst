Creating a dust model
=====================

Before running a radiative transfer simulation, you need to create a dust model that defines the optical properties of 
the dust grains in your simulation. Pinball-rt provides a :class:`~pinballrt.dust.Dust` class that allows you to create 
and manipulate dust models. In practice, there are three more specific dust models available: 
:class:`~pinballrt.dust.IsotropicDust`, :class:`~pinballrt.dust.HenyeyGreensteinDust`, and 
:class:`~pinballrt.dust.GeneralDust` that inherit from :class:`~pinballrt.dust.Dust` and enable more specific control 
over dust scattering properties. To set up a dust model, the absorption and scattering opacities as a function of wavelength 
and grain size distribution parameters (maximum dust grain size, size distribution power-law index, and sub-species 
relative abundances) are needed. At present, these must be obtained from external sources and provided to pinball-rt. 
Here we'll use simple power-law prescription, but in practice you would typically use opacities derived from laboratory 
measurements or Mie theory calculations. Note that the opacities should include astropy units to ensure that there is no 
ambiguity.

To enable maximum flexibility, opacities do not need to be provided on a regular grid. Instead, the opacities should be 
provided at some number (nsamples) of points in the N-dimensional parameter space defined by relevant inputs for determining
the opacity (maximum grain size, size distribution power-law index, sub-species abundances), and for each sample they
should be provided at a specified set of wavelengths. The :class:`~pinballrt.dust.Dust` class will then use machine learning 
to learn the opacities at any point in the parameter space during the simulation. A helper-function, 
:func:`~pinballrt.dust.suggest_opacity_sampling` is provided to help provide efficient sampling of the parameter space, 
but the user is free to provide any set of samples they choose.

.. code-block:: python

   from pinballrt.dust import IsotropicDust, suggest_opacity_sampling
   import numpy as np
   import astropy.units as u

   # Define the wavelength grid (in microns).
   wavelengths = np.logspace(-1, 4, 100) * u.micron

   # Define the dust size distribution properties.
   samples = suggest_opacity_sampling(100, amax_range=(1.*u.micron, 10*u.cm), p_range=(2.5, 4.5))

   # Expand the samples to have a wavelength dimension (nproperties, nsamples, nwavelengths).
   samples = np.moveaxis(np.repeat(np.expand_dims(samples, 1), wavelengths.size, axis=1), -1, 0)

   amax = samples[1] * u.cm
   p = samples[0]

   # Expand the wavelengths to have a sample dimension (nsamples, nwavelengths).
   wavelengths = np.repeat(np.expand_dims(wavelengths, axis=0), max(samples.shape[1], 1), axis=0)

   # Define the absorption and scattering opacities (in cm^2/g).
   power_law_index = (-2. / (1 + np.exp(-p/3.5 * (-1.5 - np.log10(amax.to(u.cm))))))
   kappa_abs = 1.0 * (wavelengths.to(u.micron)/100.0)**power_law_index * u.cm**2 / u.g
   kappa_scat = 0.5 * (wavelengths.to(u.micron)/100.0)**power_law_index * u.cm**2 / u.g

   # Create the IsotropicDust object.
   dust = IsotropicDust(lam=wavelengths[0,:], 
                         amax=amax[:,0], 
                         p=p[:,0], 
                         kabs=kappa_abs, 
                         ksca=kappa_scat)

This creates an :class:`~pinballrt.dust.IsotropicDust` object with the specified opacities, however a few additional steps are needed before the dust model can be used in a 
radiative transfer simulation. The :class:`~pinballrt.dust.IsotropicDust` object uses a machine learning model to produce opacity values during the simulation, as well as to 
randomnly sample photon frequencies emitted by dust grains during the simulation, but these models need to be trained first. To set up the 
training, we use the :meth:`~pinballrt.dust.Dust.learn` method. For example, to set up the model to learn the absorption opacity:

.. code-block:: python

   # Set up the training parameters.
   dust.learn(model="kabs",
              test_fraction=0.1,
              val_fraction=0.1,
              hidden_units=(48, 48, 48))

This example sets up a simple neural network model with three hidden layers of 48 units each to learn the dust absorption opacity. The training 
will use the kabs samples provided above, which had 100 samples across dust properties at 100 wavelengths for 10,000 total samples, with 10% of 
the samples reserved for testing and another 10% for validation. Once the training parameters are set up, we can train the model using the :meth:`~pinballrt.dust.Dust.fit` method:

.. code-block:: python

   # Train the model.
   dust.fit(epochs=50)

Finally, we can evaluate the trained model using the :meth:`~pinballrt.dust.Dust.test_model` method:

.. code-block:: python

   # Test the trained model.
   dust.test_model(plot=True)

This will evaluate the model on the test set and plot the results. A few additional models need to be trained in order to use the dust model in
a simulation. In short:

.. code-block:: python

   for model in ["ksca", "pmo", "random_nu"]:
       if model in ["kabs", "ksca"]:
          d.learn(model=model, hidden_units=(16,)*6, overwrite=True)
       else:
          d.learn(model=model, hidden_units=(48,)*3, overwrite=True)

       d.fit(epochs=300, batch_size=1000)
       d.test_model(plot=True)

This will further create models to produce the scattering opacity, planck mean opacity, and random frequencies sampled from the dust emission spectrum. 
Having to train the dust model before every simulation would be inefficient, so once the model is trained it can be saved to a file using the :meth:`~pinballrt.dust.Dust.save` method, 
and later loaded using the :func:`~pinballrt.dust.load` function:

.. code-block:: python

   # Save the trained model to a file.
   dust.save("dust_model.dst")

   # Later, load the model from the file.
   from pinballrt.dust import load
   dust = load("dust_model.dst")

pinball-rt will search the default directory as well as the ``~/.pinball-rt/data/dust/`` directory for dust model files when loading. Additionally, 
pinball-rt provides a pre-trained dust model that can be used directly without needing to train a new model from scratch:

.. code-block:: python

   from pinballrt.dust import load

   # Load the pre-trained dust model.
   dust = load("yso.dst")

Creating a Henyey-Greenstein dust model follows the same process as above, but with the addition of needing to train a model to produce the scattering asymmetry parameter (g) as a function of wavelength and dust properties:

.. code-block:: python

   from pinballrt.dust import HenyeyGreensteinDust

   g = np.tanh(p - np.log10(wavelengths.to(u.micron).value))

   # Create the Henyey-Greenstein dust model.
   dust = HenyeyGreensteinDust(lam=wavelengths[0,:], 
                               amax=amax[:,0], 
                               p=p[:,0], 
                               kabs=kappa_abs, 
                               ksca=kappa_scat,
                               g=g)

   d.learn(model="g", hidden_units=(16,)*6, overwrite=True)

   d.fit(epochs=300, batch_size=1000)
   d.test_model(plot=True)

Similarly, the most general dust model, :class:`~pinballrt.dust.GeneralDust`, follows the same process but with the addition of needing to train a model to produce the scattering phase function as a function of wavelength, scattering angle and dust properties, and additionally to randomly sample scattering angles during the simulation:

.. code-block:: python

   from pinballrt.dust import GeneralDust

   g = np.repeat(np.expand_dims(np.tanh(p - np.log10(wavelengths.to(u.micron).value)), axis=-1), 5, axis=-1)
   theta = np.tile(np.expand_dims(np.linspace(0, 180., 5), axis=(0,1)), (10 if len(dims) > 0 else 1, 10, 1)) * u.deg
   scattering_phase_function = (1 - g**2) / (4 * np.pi * (1 + g**2 - 2*g*np.cos(theta.to(u.rad).value))**(3/2))

   # Create the General dust model.
   dust = GeneralDust(lam=wavelengths[0,:], 
                      amax=amax[:,0], 
                      p=p[:,0], 
                      kabs=kappa_abs, 
                      ksca=kappa_scat,
                      scattering_phase_function=scattering_phase_function
                      theta=theta[0,0,:])

   for model in ["scattering_phase_function", "random_direction"]:
       d.learn(model=model, hidden_units=(16,)*6, overwrite=True)

       d.fit(epochs=300, batch_size=1000)
       d.test_model(plot=True)

Learning to step through high optical depth regions
---------------------------------------------------

Historically, radiative transfer simulations in regions of high optical depth have been challenging due to the large number of interactions photons undergo 
before escaping. Pinball-rt addresses this issue by implementing a machine learning approach that allows photons to "step through" high optical depth regions 
more efficiently. This is achieved by training a model to predict the output properties of photons traveling through regions of known input optical depths. 
To set this up, we can again use the :meth:`~pinballrt.dust.Dust.learn` method of the Dust class:

.. code-block:: python

   # Set up the training parameters for stepping through high optical depth.
   dust.learn(
       model="ml_step",
       nsamples=1000000,
       test_fraction=0.1,
       val_fraction=0.1,
       hidden_units=(48, 48, 48),
   )

   d.fit(epochs=50)

In practice, running the initial sample of photons through high optical depth regions can be time-consuming, so it may make sense to run it separately and save the results:

.. code-block:: python

   # Save the training data to a file.
   df = dust.run_dust_simulation(nsamples=1000000)
   df.to_csv("sim_results.csv", index=False)

It can also make sense to restrict the range of input proprties used for training to limit how computationally intensive the simulations can get. 
For example, cells with low input frequency and high optical depth can become cells with incredibly high optical depth if the cell is hot and 
photons are reemitted at higher frequencies. To avoid this, we can restrict the training to a specific range of input properties:

.. code-block:: python

   # Set up the training parameters with input property restrictions.
   df = dust.run_dust_simulation(nsamples=1000000, tau_range=(0.5, 2.0), nu_range=(dust.nu.min()*100, dust.nu.max()))

The missing ranges can then be bootstrapped (after fitting the initial model) by running additional simulations focused on those areas:

.. code-block:: python

   # Run additional simulations to cover missing ranges.
   df_high_tau = dust.run_dust_simulation(nsamples=500000, tau_range=(2.0, 4.0), nu_range=(dust.nu.min()*100, dust.nu.max()))

   # Combine the dataframes.
   df = pd.concat([df, df, df_high_tau], ignore_index=True)

Once the model has been trained, it can be tested and saved:

.. code-block:: python

   # Test the trained stepping model.
   dust.test_model(plot=True)

   # Save the trained stepping model to a file.
   dust.save("dust_model_with_stepping.dst")