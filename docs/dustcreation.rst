Creating a dust model
=====================

Before running a radiative transfer simulation, you need to create a dust model that defines the optical properties of 
the dust grains in your simulation. Pinball-rt provides a Dust class that allows you to create and manipulate dust 
models. To set up a dust model, the absorption and scattering opacities as a function of wavelength and grain size 
distribution parameters (maximum dust grain size and size distribution power-law index) are needed. At present, these 
must be obtained from external sources and provided to pinball-rt. Here we'll use simple power-law prescription, but in
practice you would typically use opacities derived from laboratory measurements or Mie theory calculations. Note that 
the opacities should include astropy units to ensure that there is no ambiguity.

.. code-block:: python

   from pinballrt.dust import Dust
   import numpy as np
   import astropy.units as u

   # Define the wavelength grid (in microns).
   wavelengths = np.logspace(-1, 4, 100) * u.micron

   # Define the dust size distribution properties.
   amax = np.logspace(-4., 1., 60.) * u.cm
   p = np.logspace(2.5, 4.5, 11)

   p, amax, wavelengths = np.meshgrid(p, amax, wavelengths, indexing='ij')

   # Define the absorption and scattering opacities (in cm^2/g).
   power_law_index = (-2. / (1 + np.exp(-p/3.5 * (-1.5 - np.log10(amax.to(u.cm))))))
   kappa_abs = 1.0 * (wavelengths.to(u.micron)/100.0)**power_law_index * u.cm**2 / u.g
   kappa_scat = 0.5 * (wavelengths.to(u.micron)/100.0)**power_law_index * u.cm**2 / u.g

   # Create the Dust object.
   dust = Dust(lam=wavelengths[0,0,:], 
               amax=amax[0,:,0], 
               p=p[:,0,0], 
               kabs=kappa_abs, 
               ksca=kappa_scat)

This creates a Dust object with the specified opacities, however a few additional steps are needed before the dust model can be used in a 
radiative transfer simulation. The Dust object uses a machine learning model to produce opacity values during the simulation, as well as to 
randomnly sample photon frequencies emitted by dust grains during the simulation, but these models need to be trained first. To set up the 
training, we use the `learn` method. For example, to set up the model to learn the absorption opacity:

.. code-block:: python

   # Set up the training parameters.
   dust.learn(
       model="kabs",
       nsamples=100000,
       test_fraction=0.1,
       val_fraction=0.1,
       hidden_units=(48, 48, 48),
   )

This example sets up a simple neural network model with three hidden layers of 48 units each to learn the dust absorption opacity. The training 
will use 100,000 samples, with 10% of the samples reserved for testing and another 10% for validation. Once the training parameters are set up, 
we can train the model using the `fit` method:

.. code-block:: python

   # Train the model.
   dust.fit(epochs=50)

Finally, we can evaluate the trained model using the `test_model` method:

.. code-block:: python

   # Test the trained model.
   dust.test_model(plot=True)

This will evaluate the model on the test set and plot the results. A few additional models need to be trained in order to use the dust model in
a simulation. In short:

.. code-block:: python

   for model in ["ksca", "pmo", "random_nu"]:
      if model in ["kabs", "ksca"]:
         d.learn(model=model, nsamples=100000, hidden_units=(16,)*6, overwrite=True)
      else:
         d.learn(model=model, nsamples=10000, hidden_units=(16,)*6, overwrite=True)

   d.fit(epochs=300, batch_size=10000)
   d.test_model(plot=True)

This will further create models to produce the scattering opacity, planck mean opacity, and random frequencies sampled from the dust emission spectrum. 
Having to train the dust model before every simulation would be inefficient, so once the model is trained it can be saved to a file using the `save` method, 
and later loaded using the `load` function:

.. code-block:: python

   # Save the trained model to a file.
   dust.save("dust_model.dst")

   # Later, load the model from the file.
   from pinballrt.dust import load
   dust = load("dust_model.dst")

pinball-rt will search the default directory as well as the ~/.pinball-rt/data/dust/ directory for dust model files when loading. Additionally, 
pinball-rt provides a pre-trained dust model that can be used directly without needing to train a new model from scratch:

.. code-block:: python

   from pinballrt.dust import load

   # Load the pre-trained dust model.
   dust = load("yso.dst")

Learning to step through high optical depth regions
---------------------------------------------------

Historically, radiative transfer simulations in regions of high optical depth have been challenging due to the large number of interactions photons undergo 
before escaping. Pinball-rt addresses this issue by implementing a machine learning approach that allows photons to "step through" high optical depth regions 
more efficiently. This is achieved by training a model to predict the output properties of photons traveling through regions of known input optical depths. 
To set this up, we can again use the `learn` method of the Dust class:

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