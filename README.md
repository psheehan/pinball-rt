[![E2E tests](https://github.com/psheehan/pinball-rt/actions/workflows/run_E2E_tests.yml/badge.svg)](https://github.com/psheehan/pinball-rt/actions/workflows/run_E2E_tests.yml)
[![codecov](https://codecov.io/gh/psheehan/pinball-rt/graph/badge.svg?token=980X3QJEOS)](https://codecov.io/gh/psheehan/pinball-rt)

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
   model = Model(grid=UniformCartesianGrid, ncells=9, dx=2.0*u.au)

   density = np.ones(model.grid.shape)*1.0e-16 * u.g / u.cm**3

   model.add_density(density, "yso.dst")
   model.add_star(star)

   model.thermal_mc(nphotons=1000000)
```

For more information, see the documentation at https://pinball-rt.readthedocs.io (coming soon).