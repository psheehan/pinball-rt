Monte Carlo radiative transfer, at Warp speed!
==============================================

pinball-rt is a modern Monte Carlo radiative transfer code, designed to run on GPUs and to leverage Machine Learning to accelerate the radiative transfer calculations.

Quickstart
----------

Install pinball-rt with pip:

.. code-block:: bash

   pip install pinball-rt

Then set up a model and run:

.. code-block:: python

   from pinballrt.dust import load
   from pinballrt.sources import Star
   from pinballrt.grids import UniformCartesianGrid
   from pinballrt.model import Model
   import astropy.units as u
   import numpy as np

   d = os.path.join(os.path.dirname(__file__), "data/yso.dst")

   # Set up the star.
   star = Star()
   star.set_blackbody_spectrum()

   # Set up the grid.
   model = Model(grid=UniformCartesianGrid, ncells=9, dx=2.0*u.au)

   density = np.ones(model.grid.shape)*1.0e-16 * u.g / u.cm**3

   model.add_density(density, d)
   model.add_star(star)

   model.thermal_mc(nphotons=1000000)

For more information, see the documentation at https://pinball-rt.readthedocs.io (coming soon).