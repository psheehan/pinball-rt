Running a model
===============

To run a radiative transfer model with pinball-rt, you first need to set up and import the relevant components: the dust, 
the radiation sources, and the grid. Then you can create a Model instance, add the dust and sources to it, and run the 
thermal Monte Carlo simulation. Start by importing the necessary modules:

.. code-block:: python

   from pinballrt.dust import load
   from pinballrt.sources import Star
   from pinballrt.grids import UniformCartesianGrid
   from pinballrt.model import Model
   import astropy.units as u
   import numpy as np

First lets set up our model using a cartesian grid:

.. code-block:: python

   # Set up the grid.
   model = Model(grid=UniformCartesianGrid, grid_kwargs={"ncells":9, "dx":2.0*u.au})

In this case, we are setting up a 9 x 9 x 9 cartesian grid with a cell size of 2 au. For additional grid geometries that are available, see :doc:`grids`. Next, we need to load the dust properties. We'll do this using
precomputed properties stored in a file that comes with pinball-rt:

.. code-block:: python

   d = load("yso.dst")

We also need to define the density distribution of the dust in our grid. Here, we'll use a uniform density for simplicity:

.. code-block:: python

   density = np.ones(model.grid.shape)*1.0e-16 * u.g / u.cm**3

Note that the density should be given a unit (via astropy.units) so that there is no ambiguity. Now we can add the dust to our model:

.. code-block:: python
    
   model.add_density(density, d)

We also need a source of photons. In this example, we'll use a star with a blackbody spectrum:

.. code-block:: python

   star = Star(temperature=4000*u.K)
   star.set_blackbody_spectrum()
   model.add_star(star)

With the model set up, we can now run the thermal Monte Carlo simulation:

.. code-block:: python

   model.thermal_mc(nphotons=1000000)

If CUDA is available, the code can also be run on the GPU with minimal additional effort:

.. code-block:: python

   model.thermal_mc(nphotons=1000000, device="cuda")