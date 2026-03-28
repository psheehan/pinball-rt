Running a model
===============

To run a radiative transfer model with pinball-rt, you first need to set up and import the relevant components: the dust, 
the radiation sources, and the grid. Then you can create a Model instance, add the dust and sources to it, and run the 
thermal Monte Carlo simulation. Start by importing the necessary modules:

.. code-block:: python

   from pinballrt.dust import load
   from pinballrt.sources import BlackbodyStar
   from pinballrt.grids import UniformCartesianGrid
   from pinballrt.model import Model
   import astropy.units as u
   import numpy as np

First lets set up our model using a cartesian grid:

.. code-block:: python

   # Set up the grid.
   model = Model(grid=UniformCartesianGrid, grid_kwargs={"ncells":9, "dx":2.0*u.au})

In this case, we are setting up a 9 x 9 x 9 cartesian grid with a cell size of 2 au. For additional grid geometries 
that are available, see :doc:`grids`. Next, we need to load the dust properties. We'll do this using precomputed 
properties stored in a file that comes with pinball-rt:

.. code-block:: python

   d = load("diana_wice.dst")

For further details on how to set up your own dust models, see :doc:`dustcreation`. We also need to define the density 
distribution of the dust in our grid. Here, we'll use a uniform density for simplicity:

.. code-block:: python

   density = np.ones(model.grid.shape)*1.0e-16 * u.g / u.cm**3

Note that the density should be given a unit (via astropy.units) so that there is no ambiguity. Now we can add the 
dust to our model:

.. code-block:: python
    
   model.set_physical_properties(density=density, dust=d, amax=1.0*u.cm, p=3.0)

Note that pinball Dust objects store dust opacities for a range of different properties. In this example we use a 
spatially constant maximum dust grain size and the grain size distribution power law index, but these properties can 
also be specified as arrays with varying values. 

We can also add gas species to the model in order to make spectral line images:

.. code-block:: python

   vx, vy, vz = np.meshgrid(0.5*(model.grid.grid.w1.numpy()[1:] + model.grid.grid.w1.numpy()[0:-1]), 
                       0.5*(model.grid.grid.w2.numpy()[1:] + model.grid.grid.w2.numpy()[0:-1]), 
                       0.5*(model.grid.grid.w3.numpy()[1:] + model.grid.grid.w3.numpy()[0:-1]), indexing='ij')
   velocity = np.concatenate((vx[np.newaxis], vy[np.newaxis], vz[np.newaxis]), axis=0) * (-1.0 * u.km / u.s)

   model.set_physical_properties(gases=["co.dat"], abundances=[1e-4], velocity=velocity, microturbulence=0.1*u.km/u.s)

Gas properties must come from files in the style of the 
`Leiden Atomic and Molecular Database (LAMDA) <https://home.strw.leidenuniv.nl/~moldata/>`_, and will be downloaded
automatically if they are not already present on your system and are found in the LAMDA database.

We also need a source of photons. In this example, we'll use a star with a blackbody spectrum:

.. code-block:: python

   star = BlackbodyStar(temperature=4000*u.K)
   model.add_sources(star)

but pinball-rt provides several different types of sources that can be used. See :doc:`sources` for more information on 
how to set up these alternative types of sources.

.. code-block:: python

   model.thermal_mc(nphotons=100000)

If CUDA is available, the code can also be run on the GPU with minimal additional effort:

.. code-block:: python

   model.thermal_mc(nphotons=100000, device="cuda")

Additionally, pinball-rt can run in parallel using multiple CPU cores by setting the ``ncores`` parameter, 
using either the ``multiprocessing`` backend or MPI.:

.. code-block:: python

   model = Model(grid=UniformCartesianGrid, 
                 grid_kwargs={"ncells":9, "dx":2.0*u.au}, 
                 ncores=4)            # multiprocessing
   model = Model(grid=UniformCartesianGrid, 
                 grid_kwargs={"ncells":9, "dx":2.0*u.au}, 
                 ncores=4, mpi=True)  # MPI

Note that to run with MPI, you need to launch the script using the ``mpirun`` or ``mpiexec`` command, or start an 
interactive ipython session using ``ipyparallel``, and have ``mpi4py`` installed. Both modules must be installed 
separately. pinball-rt can scale to multiple GPUs on one ore more nodes by using a combination of the ``device`` 
parameter and the ``ncores`` parameter.

To make images from the model, we can use the ``make_image`` method:

.. code-block:: python

   image = model.make_image(npix=256, pixel_size=0.2*u.arcsec, 
                            channels=np.array([1., 1000.])*u.micron, incl=45.*u.degree, 
                            pa=45.*u.degree, distance=1.*u.pc, device='cpu', 
                            include_gas=False, nphotons=1000000)

Channels can be specified in units of wavelength, frequency, or velocity.

