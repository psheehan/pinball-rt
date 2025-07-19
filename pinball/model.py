import astropy.constants as const
import astropy.units as u
from .grids import Grid
from .camera import Camera
from schwimmbad import SerialPool
import xarray as xr
import numpy as np
import warp as wp
import time

class Model:
    def __init__(self, grid: Grid, ncells: int = 9, dx: u.Quantity = 1.0 * u.au, mirror: bool = False, ncores = 1, pool = SerialPool()):
        """Initialize the Model with a grid and optional parameters."""
        self.grid_list = [grid(ncells=ncells, dx=dx) for _ in range(ncores)]
        self.grid = self.grid_list[0]
        self.camera = Camera(self.grid)
        self.camera_list = [Camera(grid) for grid in self.grid_list]
        self.ncores = ncores
        self.pool = pool

    def add_density(self, density: u.Quantity, dust):
        """Add density to the grid."""
        for grid in self.grid_list:
            grid.add_density(density, dust)

    def add_star(self, star):
        """Add a star to the grid."""
        for grid in self.grid_list:
            grid.add_star(star)

    def thermal_mc(self, nphotons, Qthresh=2.0, Delthresh=1.1, p=99.):
        told = self.grid.grid.temperature.numpy().copy()

        count = 0
        while count < 10:
            print("Iteration", count)
            treallyold = told.copy()
            told = self.grid.grid.temperature.numpy().copy()

            t1 = time.time()
            result = self.pool.map(lambda grid: grid.propagate_photons(grid.emit(int(nphotons / self.ncores))), self.grid_list)
            success = [r for r in result]
            t2 = time.time()
            print("Time:", t2 - t1)

            total_energy = np.mean(np.array([grid.grid.energy.numpy() for grid in self.grid_list]), axis=0)
            self.grid.grid.energy = wp.array3d(total_energy, dtype=float)

            t1 = time.time()
            self.grid.update_grid()
            t2 = time.time()
            print("Update grid temperature time:", t2 - t1)

            for grid in self.grid_list:
                grid.grid.temperature = self.grid.grid.temperature
                grid.grid.energy = self.grid.grid.energy

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

            count += 1

    def scattering_mc(self, nphotons, wavelengths):
        for grid in self.grid_list:
            grid.scattering = np.zeros((len(wavelengths),)+grid.shape, dtype=np.float32)

        for i, wavelength in enumerate(wavelengths):
            for grid in self.grid_list:
                grid.initialize_luminosity_array(wavelength=wavelength)

            t1 = time.time()
            result = self.pool.map(lambda grid: grid.propagate_photons_scattering(grid.emit(int(nphotons / self.ncores), wavelength, scattering=True), i), self.grid_list)
            success = [r for r in result]
            t2 = time.time()
            print("Time:", t2 - t1)

            total_scattering = np.mean(np.array([grid.scattering for grid in self.grid_list]), axis=0) / (4.*np.pi * self.grid.volume)
            for grid in self.grid_list:
                grid.scattering[i] = total_scattering[i]

    def make_image(self, nx, ny, pixel_size, lam, incl, pa, dpc):
        # First, run a scattering simulation to get the scattering phase function

        self.scattering_mc(100000, lam)

        # Now set up the image proper.

        for camera in self.camera_list:
            camera.set_orientation(incl, pa, dpc)

        pixel_size = (pixel_size*u.arcsecond*dpc*u.pc).cgs.value

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

        intensity = np.array(list(self.pool.map(lambda x: x[0].raytrace(x[1], x[2], nx, ny, image.pixel_size, image.nu).numpy(), 
                        zip(self.camera_list, np.array_split(new_x, self.ncores), np.array_split(new_y, self.ncores))))).sum(axis=0)

        intensity += np.array(list(self.pool.map(lambda camera: camera.raytrace_sources(image.x, image.y, nx, ny, image.nu, dpc, 
                        nrays=int(1000/self.ncores)).numpy(), self.camera_list))).mean(axis=0)

        image = image.assign(intensity=(("x","y","lam"), intensity))

        return image