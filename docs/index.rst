.. pinball-rt documentation master file, created by
   sphinx-quickstart on Fri Aug  8 02:31:02 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Monte Carlo radiative transfer, at Warp speed!
==============================================

pinball-rt is a modern Monte Carlo radiative transfer code, designed to run on GPUs and to leverage Machine Learning to accelerate the radiative transfer calculations.

Quickstart
----------

Install pinball-rt with pip:

.. code-block:: bash

   pip install git+https://github.com/psheehan/pinball-rt.git

Then set up a model and run:

.. code-block:: python

   from pinballrt.sources import Star
   from pinballrt.grids import UniformCartesianGrid
   from pinballrt.model import Model
   import astropy.units as u
   import numpy as np

   # Set up the star.
   star = Star()
   star.set_blackbody_spectrum()

   # Set up the grid.
   model = Model(grid=UniformCartesianGrid, grid_kwargs={"ncells":9, "dx":2.0*u.au})

   density = np.ones(model.grid.shape)*1.0e-16 * u.g / u.cm**3

   model.add_density(density, "yso.dst")
   model.add_star(star)

   model.thermal_mc(nphotons=1000000)

Or, click the link below to try it out in Google Colab:

.. raw:: html 

   <a target="_blank" href="https://colab.research.google.com/github/psheehan/pinball-rt/blob/main/examples/pinball-demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Or, Try Me In Colab"/></a>

.. toctree::
   :maxdepth: 1
   :caption: User-Guide

   installation
   running
   dustcreation

.. toctree::
   :maxdepth: 1
   :caption: API:

   model
   dust
   sources
   grids

Acknowledging pinball
---------------------

Love pinball and want to cite it in your paper? Please include the following citations:

<COMING SOON>

Contributing and/or Bugs
------------------------

Want to contribute? Found a bug? Please feel free to open an `issue <https://github.com/psheehan/pinball-warp/issues/new/choose>`_ or `pull request <https://github.com/psheehan/pinball-warp/compare>`_ on GitHub and the pinball team will follow up with you there.
