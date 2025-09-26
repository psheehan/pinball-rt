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
```

For more information, see the documentation at https://pinball-rt.readthedocs.io.