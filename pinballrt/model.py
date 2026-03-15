import astropy.constants as const
import astropy.units as u
from .grids import Grid
from .dust import load, Dust
from .camera import Camera
from schwimmbad import SerialPool
import xarray as xr
import numpy as np
import warp as wp
import torch
import time

class Model:
    def __init__(self, grid: Grid, grid_kwargs={}, ncores = 1, pool = SerialPool()):
        """
        Initialize the Model with a grid and optional parameters.

        Parameters
        ----------
        grid : Grid
            The grid to use for the model.
        ncells : int or list-like, optional
            The number of cells in the grid. Can be specified either as an integer, in which case each dimension will 
            have the same number of cells, or a 3 element tuple/list/array that specifies the number of cells separately 
            for each dimension (default is 9).
        dx : astropy.units.Quantity, optional
            The cell size (default is 1.0 * u.au).
        mirror : bool, optional
            Whether to use a mirrored grid (default is False).
        ncores : int, optional
            The number of CPU cores to use (default is 1).
        pool : schwimmbad.Pool, optional
            The pool to use for parallel processing (default is SerialPool()).
        """
        self.grid_list = {"cpu":[grid(**grid_kwargs, device='cpu') for _ in range(ncores)]}
        if wp.get_cuda_device_count() > 0:
            self.grid_list["cuda"] = [grid(**grid_kwargs, device=d) for d in wp.get_cuda_devices()]
            self.grid = self.grid_list["cuda"][0]
        else:
            self.grid = self.grid_list["cpu"][0]

        self.camera_list = {}
        for device in self.grid_list:
            self.camera_list[device] = [Camera(grid) for grid in self.grid_list[device]]
        if "cuda" in self.camera_list:
            self.camera = self.camera_list["cuda"][0]
        else:
            self.camera = self.camera_list["cpu"][0]
            
        self.ncores = ncores
        self.pool = pool

    def add_density(self, density: u.Quantity, dust):
        """
        Add density to the grid.
        
        Parameters
        ----------
        density : astropy.units.Quantity
            The density to add to the grid.
        dust : Dust
            The dust distribution to use.
        """
        for device in self.grid_list:
            for grid in self.grid_list[device]:
                grid.add_density(density, load(dust) if isinstance(dust, str) else dust)

    def add_star(self, star):
        """
        Add a star to the grid.

        Parameters
        ----------
        star : Star
            The star to add to the grid.
        """
        for device in self.grid_list:
            for grid in self.grid_list[device]:
                grid.add_star(star)

    def thermal_mc(self, nphotons, use_ml_step=False, Qthresh=2.0, Delthresh=1.1, p=99., device="cpu", return_timing=False, nbatch=1):
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
        told = self.grid.grid.temperature.numpy().copy()

        timing = {}
        count = 0
        while count < 10:
            iter_timing = {}

            print("Iteration", count)
            treallyold = told.copy()
            told = self.grid.grid.temperature.numpy().copy()

            t1 = time.time()
            result = self.pool.map(lambda grid: grid.propagate_photons(
                    grid.emit(int(nphotons / self.ncores / nbatch), timing=iter_timing), 
                    use_ml_step=use_ml_step, timing=iter_timing), self.grid_list[device]*nbatch)
            success = [r for r in result]
            t2 = time.time()
            iter_timing["Total Time"] = t2 - t1

            total_energy = np.mean(np.array([grid.grid.energy.numpy() for grid in self.grid_list[device]]), axis=0)
            with wp.ScopedDevice(self.grid.device):
                self.grid.grid.energy = wp.array3d(total_energy, dtype=float)

            t1 = time.time()
            self.grid.update_grid(timing=iter_timing)
            t2 = time.time()
            iter_timing["Update grid temperature time"] = t2 - t1

            for dev in self.grid_list:
                for grid in self.grid_list[dev]:
                    with wp.ScopedDevice(grid.device):
                        grid.grid.temperature = wp.array3d(self.grid.grid.temperature.numpy(), dtype=float)
                        grid.grid.energy = wp.zeros(self.grid.shape, dtype=float)

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

    def scattering_mc(self, nphotons, wavelengths, device="cpu", return_timing=False):
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
            for grid in self.grid_list[dev]:
                with wp.ScopedDevice(grid.device):
                    grid.scattering = torch.zeros((len(wavelengths),)+grid.shape, dtype=torch.float32, device=wp.device_to_torch(wp.get_device()))

        timing = {}
        for i, wavelength in enumerate(wavelengths):
            iter_timing = {}

            for grid in self.grid_list[device]:
                grid.initialize_luminosity_array(wavelength=wavelength)

            t1 = time.time()
            result = self.pool.map(lambda grid: grid.propagate_photons_scattering(grid.emit(int(nphotons / self.ncores), wavelength, scattering=True, timing=iter_timing), i, timing=iter_timing), self.grid_list[device])
            success = [r for r in result]
            t2 = time.time()
            iter_timing["Total Time"] = t2 - t1

            total_scattering = torch.mean(torch.cat([torch.unsqueeze(grid.scattering, 0) for grid in self.grid_list[device]]), axis=0) / (4.*np.pi * self.grid.volume.to(device))
            for dev in self.grid_list:
                for grid in self.grid_list[dev]:
                    grid.scattering[i] = total_scattering[i].clone().to(wp.device_to_torch(grid.device))

            timing[str(i)] = iter_timing

        if return_timing:
            return timing

    def make_image(self, npix=100, pixel_size=None, lam=np.array([1.])*u.micron, incl=0, pa=0, distance=1*u.pc, nphotons=100000, device="cpu"):
        """
        Create an image from the dust distribution.

        Parameters
        ----------
        npix : int or tuple, optional
            The number of pixels in the image.
        pixel_size : Quantity
            The size of each pixel in the image. If none, will set the pixel size such that the image is
            25% larger than the grid.
        lam : array-like Quantity
            The wavelengths to simulate.
        incl : Quantity
            The inclination angle of the image.
        pa : Quantity
            The position angle of the image.
        dpc : Quantity
            The distance to the image plane in parsecs.
        device : str, optional
            The device to use for the simulation (default is "cpu").
        """

        if isinstance(npix, int):
            nx, ny = npix, npix
        elif isinstance(npix, (list, tuple, np.ndarray)):
            nx, ny = npix

        if pixel_size is None:
            pixel_size = ((1.25*self.grid.grid_size()*self.grid.distance_unit / distance).decompose()*u.radian).to(u.arcsec) / npix

        # First, run a scattering simulation to get the scattering phase function

        self.scattering_mc(nphotons, lam, device=device)

        # Now set up the image proper.

        for dev in self.camera_list:
            for camera in self.camera_list[dev]:
                camera.set_orientation(incl, pa, distance)

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

        intensity = np.array(list(self.pool.map(lambda x: x[0].raytrace(x[1], x[2], nx, ny, physical_pixel_size, image.nu).numpy(), 
                        zip(self.camera_list[device], np.array_split(new_x, self.ncores), np.array_split(new_y, self.ncores))))).sum(axis=0) * (u.Jy / u.steradian)

        source_intensity = np.array(list(self.pool.map(lambda camera: camera.raytrace_sources(image.x, image.y, nx, ny, image.nu, physical_pixel_size*self.grid.distance_unit, 
                        nrays=int(1000/self.ncores)).numpy(), self.camera_list[device]))).mean(axis=0) * u.Jy/u.steradian

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