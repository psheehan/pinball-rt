[![E2E tests](https://github.com/psheehan/pinball-rt/actions/workflows/run_E2E_tests.yml/badge.svg)](https://github.com/psheehan/pinball-rt/actions/workflows/run_E2E_tests.yml)
[![codecov](https://codecov.io/gh/psheehan/pinball-rt/graph/badge.svg?token=980X3QJEOS)](https://codecov.io/gh/psheehan/pinball-rt)
[![Documentation Status](https://readthedocs.org/projects/pinball-rt/badge/?version=latest)](https://pinball-rt.readthedocs.io/en/latest/?badge=latest)
<a target="_blank" href="https://colab.research.google.com/github/psheehan/pinball-rt/blob/main/examples/pinball-demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Try Me In Colab"/>
</a>


Monte Carlo radiative transfer, at Warp speed!
==============================================

pinball-rt is a modern Monte Carlo radiative transfer code, designed to run on GPUs and to leverage Machine Learning to accelerate the radiative transfer calculations.

Quickstart
----------

Install pinball-rt with pip:

```bash
   pip install git+https://github.com/psheehan/pinball-rt.git
```

Then set up a model and run:

```python
   from pinballrt.sources import BlackbodyStar
   from pinballrt.grids import UniformCartesianGrid
   from pinballrt.model import Model
   import astropy.units as u
   import numpy as np

   # Set up the star.
   star = BlackbodyStar()

   # Set up the grid.
   model = Model(grid=UniformCartesianGrid, grid_kwargs={"ncells":9, "dx":2.0*u.au})

   density = np.ones(model.grid.shape)*1.0e-16 * u.g / u.cm**3
   amax = np.ones(model.grid.shape) * u.cm
   amax[4,4,4] = 1. * u.micron

   model.set_physical_properties(density=density, dust="diana_wice.dst", amax=amax)
   model.add_sources(star)

   # Calculate the temperature structure.
   model.thermal_mc(nphotons=100000)

   # Make an image.
   image = model.make_image(npix=256, pixel_size=0.2*u.arcsec, 
                            channels=np.array([1., 1000.])*u.micron, incl=45.*u.degree, 
                            pa=45.*u.degree, distance=1.*u.pc, device='cpu', 
                            include_gas=False, nphotons=1000000)
```

For more information, see the documentation at https://pinball-rt.readthedocs.io.