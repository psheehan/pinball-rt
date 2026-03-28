import astropy.constants as const
import astropy.units as u

from .sources import DiffuseSource, EnergySource
from .grids import Grid
from .dust import load, Dust
from .gas import Gas
from .camera import Camera
from schwimmbad import SerialPool, MultiPool
import xarray as xr
import numpy as np
import warp as wp
import torch
from tqdm.auto import tqdm
from numpy.random import SeedSequence, seed
import time

#wp.config.quiet = True

def initializer(arg):
    import warp as wp
    #wp.config.quiet = True
    wp.init()
    tqdm.set_lock(arg)

class Model:
    def __init__(self, grid: Grid, grid_kwargs={}, ncores=1, mpi=False):
        """
        Initialize the Model with a grid and optional parameters.

        Parameters
        ----------
        grid : Grid
            The grid to use for the model.
        grid_kwargs : dict, optional
            Additional keyword arguments to pass to the grid constructor (default is an empty dictionary).
        ncores : int, optional
            The number of CPU cores to use (default is 1).
        mpi : bool, optional
            Whether to use mpi for parallel processing.
        """
        self.grid_list = {"cpu":grid(**grid_kwargs, device='cpu')}
        if wp.get_cuda_device_count() > 0:
            self.grid_list["cuda"] = grid(**grid_kwargs, device=d)
            self.grid = self.grid_list["cuda"]
        else:
            self.grid = self.grid_list["cpu"]

        self.camera_list = {}
        for device in self.grid_list:
            self.camera_list[device] = Camera(self.grid_list[device])
        if "cuda" in self.camera_list:
            self.camera = self.camera_list["cuda"]
        else:
            self.camera = self.camera_list["cpu"]
            
        self.ncores = ncores
        if self.ncores > 1:
            if mpi:
                from mpi4py.futures import MPIPoolExecutor
                self.pool = MPIPoolExecutor(self.ncores)
            else:
                self.pool = MultiPool(self.ncores, initializer=initializer, initargs=(tqdm.get_lock(),))
        else:
            self.pool = SerialPool()

    def set_physical_properties(self, density=None, dust=None, amax=None, p=None, gases=None, abundances=None, 
                                velocity=None, microturbulence=None):
        """
        Set the physical properties of the grid.
        
        Parameters
        ----------
        density : astropy.units.Quantity, optional
            The density to add to the grid.
        dusttogasratio : float, optional
            The dust-to-gas mass ratio. Default is 0.01.
        dust : Dust, optional
            The dust distribution to use.
        amax : astropy.units.Quantity, optional
            The maximum dust size of the dust in the grid. Can be specified as a single value to be constant over
            the grid, or as an array with a spatially varying value.
        p : float or array-like, optional
            The dust grain size distribution power-law slope. Can be specified as a single value to be constant over
            the grid, or as an array with a spatially varying value.
        gases : Gas, optional
            List of gas species to include in the grid.
        abundances : dict, optional
            The abundances of the gas species.
        velocity : astropy.units.Quantity, optional
            The velocity field of the gas.
        microturbulence : astropy.units.Quantity, optional
            The microturbulent velocity of the gas.
        """
        for device in self.grid_list:
            self.grid_list[device].set_physical_properties(density=density, dusttogasratio=dusttogasratio,
                                                           dust=load(dust) if isinstance(dust, str) else dust,
                                                           amax=amax, p=p, gases=gases, abundances=abundances,
                                                           velocity=velocity, microturbulence=microturbulence)

    def add_sources(self, sources):
        """
        Add sources to the grid.

        Parameters
        ----------
        sources : list of Source objects
            The sources to add to the grid.
        """
        for device in self.grid_list:
            self.grid_list[device].add_sources(sources)

    def thermal_mc(self, nphotons, use_ml_step=False, Qthresh=2.0, Delthresh=1.1, p=99., device="cpu", 
                   return_timing=False, nbatch=1):
        """
        Perform a thermal Monte Carlo simulation.

        Parameters
        ----------
        nphotons : int
            The number of photons to simulate.
        Qthresh : float, optional
            The threshold for the quality factor (default is 2.0).
        Delthresh : float, optional
            The threshold for the temperature change (default is 1.1).
        p : float, optional
            The percentile to use for the temperature adjustment (default is 99.).
        device : str, optional
            The device to use for the simulation (default is "cpu").
        """
        for source in self.grid_list[device].sources:
            if isinstance(source, DiffuseSource):
                source.initialize_luminosity_array(wavelength="random")

        told = self.grid.grid.temperature.numpy().copy()

        timing = {}
        count = 0
        while count < 10:
            iter_timing = {}

            print("Iteration", count)
            treallyold = told.copy()
            told = self.grid.grid.temperature.numpy().copy()

            njobs = self.ncores * nbatch

            t1 = time.time()
            result = self.pool.map(thermal_mc_task, 
                                   zip([self.grid_list[device]]*njobs, 
                                        range(njobs), 
                                        SeedSequence(np.random.randint(10000)).spawn(njobs),
                                        [nphotons]*njobs,
                                        [njobs]*njobs,
                                        [use_ml_step]*njobs))
            total_energy = [r[0] for r in result]
            time.sleep(0.1)
            t2 = time.time()
            iter_timing["Total Time"] = t2 - t1

            total_energy = np.mean(np.array(total_energy), axis=0)
            with wp.ScopedDevice(self.grid.device):
                self.grid.grid.energy = wp.array3d(total_energy, dtype=float)

            t1 = time.time()
            self.grid.update_grid(timing=iter_timing)
            t2 = time.time()
            iter_timing["Update grid temperature time"] = t2 - t1

            for dev in self.grid_list:
                with wp.ScopedDevice(self.grid_list[dev].device):
                        self.grid_list[dev].grid.temperature = wp.array3d(self.grid.grid.temperature.numpy(), dtype=float)
                        self.grid_list[dev].grid.energy = wp.zeros(self.grid.shape, dtype=float)

            if count > 1:
                R = np.maximum(told/self.grid.grid.temperature.numpy(), self.grid.grid.temperature.numpy()/told)
                Rold = np.maximum(told/treallyold, treallyold/told)

                Q = np.percentile(R, p)
                Qold = np.percentile(Rold, p)

                Del = max(Q/Qold, Qold/Q)

                print(count, Q, Del)
                if Q < Qthresh and Del < Delthresh:
                    break
            else:
                print(count)

            timing[str(count)] = iter_timing
            count += 1

        if return_timing:
            return timing

    def scattering_mc(self, nphotons, wavelengths, device="cpu", return_timing=False, nbatch=1):
        """
        Perform a scattering Monte Carlo simulation.

        Parameters
        ----------
        nphotons : int
            The number of photons to simulate.
        wavelengths : array-like
            The wavelengths to simulate.
        device : str, optional
            The device to use for the simulation (default is "cpu").
        """
        for dev in self.grid_list:
            with wp.ScopedDevice(self.grid_list[dev].device):
                self.grid_list[dev].scattering = torch.zeros((len(wavelengths),)+self.grid_list[dev].shape, 
                                                             dtype=torch.float32, 
                                                             device=wp.device_to_torch(wp.get_device()))

        timing = {}
        for i, wavelength in enumerate(wavelengths):
            iter_timing = {}

            for source in self.grid_list[device].sources + [self.grid_list[device].grid_source]:
                if isinstance(source, DiffuseSource):
                    source.initialize_luminosity_array(wavelength=wavelength)

            njobs = self.ncores * nbatch

            t1 = time.time()
            result = self.pool.map(scattering_mc_task, 
                                   zip([self.grid_list[device]]*njobs, 
                                       range(njobs), 
                                       SeedSequence(np.random.randint(10000)).spawn(njobs),
                                       [nphotons]*njobs,
                                       [njobs]*njobs,
                                       [wavelength]*njobs,
                                       [i]*njobs))
            total_scattering = [r[0] for r in result]
            time.sleep(0.1)
            t2 = time.time()
            iter_timing["Total Time"] = t2 - t1

            total_scattering = torch.mean(torch.cat([torch.unsqueeze(total_scattering[i], 0) for i in range(njobs)]), 
                                                    axis=0) / (4.*np.pi * self.grid.volume.to(device))

            for source in self.grid_list[device].sources:
                if isinstance(source, DiffuseSource) and not isinstance(source, EnergySource):
                    total_scattering[i] += torch.tensor((source.luminosity * (self.grid.distance_unit**2 * u.Jy) * \
                                                         source.density / (4.*np.pi * u.steradian * \
                                                                           (self.grid.grid.dust_density.numpy() * \
                                                                            self.grid.dust.interpolate_kext(
                                                                                self.grid.grid.p.numpy(), 
                                                                                self.grid.grid.amax.numpy(), 
                                                                                np.ones(self.grid.shape)*wavelength) * \
                                                         self.grid.distance_unit**-1))).value, 
                                                         device=device)

            for dev in self.grid_list:
                self.grid_list[dev].scattering[i] = total_scattering[i].clone().to(wp.device_to_torch(self.grid_list[dev].device))

            timing[str(i)] = iter_timing

        if return_timing:
            return timing

    def make_image(self, npix=100, pixel_size=None, channels=None, rest_frequency=None, incl=0, pa=0, distance=1*u.pc, 
                   include_dust=True, include_gas=True, include_sources=True, nphotons=100000, device="cpu"):
        """
        Create an image from the dust distribution.

        Parameters
        ----------
        npix : int or tuple, optional
            The number of pixels in the image.
        pixel_size : Quantity
            The size of each pixel in the image. If none, will set the pixel size such that the image is
            25% larger than the grid.
        channels : array-like Quantity
            The wavelengths, frequencies, or velocities to simulate.
        incl : Quantity
            The inclination angle of the image.
        pa : Quantity
            The position angle of the image.
        distance : Quantity
            The distance to the image plane.
        include_dust : bool, optional
            Whether to include dust in the image (default is True).
        include_gas : bool, optional
            Whether to include gas in the image (default is True).
        include_sources : bool, optional
            Whether to include sources in the image (default is True).
        nphotons : int, optional
            The number of photons to use in the Monte Carlo simulation (default is 100000).
        device : str, optional
            The device to use for the simulation (default is "cpu").
        """

        if isinstance(npix, int):
            nx, ny = npix, npix
        elif isinstance(npix, (list, tuple, np.ndarray)):
            nx, ny = npix

        if pixel_size is None:
            pixel_size = ((1.25*self.grid.grid_size()*self.grid.distance_unit / distance).decompose()*
                          u.radian).to(u.arcsec) / npix

        self.grid.grid.include_dust = include_dust
        self.grid.grid.include_gas = include_gas

        # Check whether spectral is wavelength or frequency

        if channels.unit.is_equivalent(u.micron):
            lam = channels.to(u.micron)
            nu = (const.c / channels).to(u.GHz)
        elif channels.unit.is_equivalent(u.GHz):
            nu = channels.to(u.GHz)
            lam = (const.c / nu).to(u.micron)
        elif channels.unit.is_equivalent(u.km / u.s):
            if rest_frequency is None:
                raise ValueError("rest_frequency must be provided when channels are in velocity units.")
            nu = (rest_frequency * (1 - channels / const.c)).to(u.GHz)
            lam = (const.c / nu).to(u.micron)
        else:
            raise ValueError("Either lam or nu must be provided.")

        # Check which lines from the gas should be included

        if include_gas:
            self.grid.select_lines(lam)

        self.grid.grid.include_dust = include_dust
        self.grid.grid.include_gas = include_gas

        # Check whether spectral is wavelength or frequency

        if channels.unit.is_equivalent(u.micron):
            lam = channels.to(u.micron)
            nu = (const.c / channels).to(u.GHz)
        elif channels.unit.is_equivalent(u.GHz):
            nu = channels.to(u.GHz)
            lam = (const.c / nu).to(u.micron)
        elif channels.unit.is_equivalent(u.km / u.s):
            if rest_frequency is None:
                raise ValueError("rest_frequency must be provided when channels are in velocity units.")
            nu = (rest_frequency * (1 - channels / const.c)).to(u.GHz)
            lam = (const.c / nu).to(u.micron)
        else:
            raise ValueError("Either lam or nu must be provided.")

        # Check which lines from the gas should be included

        if include_gas:
            self.grid.select_lines(lam)

        # First, run a scattering simulation to get the scattering phase function

        if include_dust:
            self.scattering_mc(nphotons, lam, device=device)

        # Now set up the image proper.

        for dev in self.camera_list:
            self.camera_list[dev].set_orientation(incl, pa, distance)

        physical_pixel_size = (pixel_size*distance).to(self.grid.distance_unit, equivalencies=u.dimensionless_angles()).value

        image = xr.Dataset(
            #data_vars={
            #    "intensity": (["x", "y", "lam"], np.zeros((nx, ny, lam.size))),},
            coords={
                "x": ("x", (np.arange(nx) - nx / 2)*pixel_size),
                "y": ("y", (np.arange(ny) - ny / 2)*pixel_size),
                "lam": ("lam", lam),
                "nu": ("lam", (const.c / lam).to(u.GHz)),},
            attrs={
                "pixel_size": pixel_size,})

        new_x, new_y = xr.broadcast(image.x, image.y)
        new_x, new_y = new_x.values.flatten(), new_y.values.flatten()

        njobs = self.ncores

        intensity = np.array(list(self.pool.map(make_image_raytracing_task, 
                                                zip([self.camera_list[device]]*njobs, 
                                                     np.array_split(new_x, self.ncores), 
                                                     np.array_split(new_y, self.ncores),
                                                     [nx]*njobs,
                                                     [ny]*njobs,
                                                     [physical_pixel_size]*njobs,
                                                     [image.nu]*njobs)
                                                ))).sum(axis=0) * (u.Jy / u.steradian)

        if include_sources:
            source_intensity = np.array(list(self.pool.map(make_image_source_task, 
                                                        zip([self.camera_list[device]]*njobs, 
                                                                SeedSequence(np.random.randint(10000)).spawn(njobs),
                                                                [image]*njobs,
                                                                [nx]*njobs,
                                                                [ny]*njobs,
                                                                [physical_pixel_size]*njobs,
                                                                [njobs]*njobs,)
                                                        ))).mean(axis=0) * u.Jy/u.steradian

        intensity += source_intensity

        image = image.assign(intensity=(("x","y","lam"), intensity.to(u.Jy / u.steradian)))

        return image

    def make_spectrum(self, lam=np.array([1.])*u.micron, incl=0, pa=0, distance=1*u.pc, nphotons=10000, device="cpu"):
        """
        Raytrace to make a spectrum.

        Parameters
        ----------
        lam : array-like Quantity
            The wavelengths to simulate.
        incl : Quantity
            The inclination angle of the image.
        pa : Quantity
            The position angle of the image.
        distance : Quantity
            The distance to the image plane in parsecs.
        nphotons : int
            The number of photons to simulate, per wavelength.
        device : str, optional
            The device to use for the simulation (default is "cpu").
        Returns
        -------
        spectrum : xarray.Dataset
            The resulting spectrum.
        """
        image = self.make_image(lam=lam, incl=incl, pa=pa, distance=distance, nphotons=nphotons, device=device)

        spectrum = image.sum(dim=["x","y"])
        spectrum = spectrum.assign(intensity=(("lam",), (spectrum.intensity.data * image.pixel_size**2).to(u.Jy)))

        return spectrum
    
def thermal_mc_task(args):
    grid, position, s, nphotons, njobs, use_ml_step = args
    seed(s.generate_state(1)[0])
    iter_timing = {}
    photon_list = grid.emit(int(nphotons / njobs), timing=iter_timing)
    grid.propagate_photons(photon_list, use_ml_step=use_ml_step, timing=iter_timing, position=position)

    return grid.grid.energy.numpy(), iter_timing

def scattering_mc_task(args):
    grid, position, s, nphotons, njobs, wavelength, i = args
    seed(s.generate_state(1)[0])
    iter_timing = {}
    photon_list = grid.emit(int(nphotons / njobs), wavelength, scattering=True, timing=iter_timing)
    grid.propagate_photons_scattering(photon_list, i, timing=iter_timing, position=position)

    return grid.scattering, iter_timing

def make_image_raytracing_task(args): 
    camera, x, y, nx, ny, physical_pixel_size, nu = args
    return camera.raytrace(x, y, nx, ny, physical_pixel_size, nu).numpy()

def make_image_source_task(args):
    camera, s, image, nx, ny, physical_pixel_size, njobs = args
    seed(s.generate_state(1)[0])
    return camera.raytrace_sources(image.x, image.y, nx, ny, image.nu, physical_pixel_size*camera.grid.distance_unit, 
                nrays=int(1000/njobs)).numpy()