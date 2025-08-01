import astropy.constants as const
import astropy.units as u
from .grids import Grid
from .dust import load
from .camera import Camera
from schwimmbad import SerialPool
import xarray as xr
import numpy as np
import warp as wp
import torch
import time

class Model:
    def __init__(self, grid: Grid, ncells: int = 9, dx: u.Quantity = 1.0 * u.au, mirror: bool = False, ncores = 1, pool = SerialPool()):
        """Initialize the Model with a grid and optional parameters."""
        self.grid_list = {"cpu":[grid(ncells=ncells, dx=dx, device='cpu') for _ in range(ncores)]}
        if wp.get_cuda_device_count() > 0:
            self.grid_list["cuda"] = [grid(ncells=ncells, dx=dx, device=d) for d in wp.get_cuda_devices()]
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
        """Add density to the grid."""
        for device in self.grid_list:
            for grid in self.grid_list[device]:
                with wp.ScopedDevice(grid.device):
                    d = load(dust)
                grid.add_density(density, d)

    def add_star(self, star):
        """Add a star to the grid."""
        for device in self.grid_list:
            for grid in self.grid_list[device]:
                grid.add_star(star)

    def thermal_mc(self, nphotons, Qthresh=2.0, Delthresh=1.1, p=99., device="cpu"):
        told = self.grid.grid.temperature.numpy().copy()

        count = 0
        while count < 10:
            print("Iteration", count)
            treallyold = told.copy()
            told = self.grid.grid.temperature.numpy().copy()

            t1 = time.time()
            result = self.pool.map(lambda grid: grid.propagate_photons(grid.emit(int(nphotons / self.ncores))), self.grid_list[device])
            success = [r for r in result]
            t2 = time.time()
            print("Time:", t2 - t1)

            total_energy = np.mean(np.array([grid.grid.energy.numpy() for grid in self.grid_list[device]]), axis=0)
            with wp.ScopedDevice(self.grid.device):
                self.grid.grid.energy = wp.array3d(total_energy, dtype=float)

            t1 = time.time()
            self.grid.update_grid()
            t2 = time.time()
            print("Update grid temperature time:", t2 - t1)

            for dev in self.grid_list:
                for grid in self.grid_list[dev]:
                    with wp.ScopedDevice(grid.device):
                        grid.grid.temperature = wp.array3d(self.grid.grid.temperature)
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

            count += 1

    def scattering_mc(self, nphotons, wavelengths, device="cpu"):
        for dev in self.grid_list:
            for grid in self.grid_list[dev]:
                with wp.ScopedDevice(grid.device):
                    grid.scattering = torch.zeros((len(wavelengths),)+grid.shape, dtype=torch.float32, device=wp.device_to_torch(wp.get_device()))

        for i, wavelength in enumerate(wavelengths):
            for grid in self.grid_list[device]:
                grid.initialize_luminosity_array(wavelength=wavelength)

            t1 = time.time()
            result = self.pool.map(lambda grid: grid.propagate_photons_scattering(grid.emit(int(nphotons / self.ncores), wavelength, scattering=True), i), self.grid_list[device])
            success = [r for r in result]
            t2 = time.time()
            print("Time:", t2 - t1)

            total_scattering = torch.mean(torch.cat([torch.unsqueeze(grid.scattering, 0) for grid in self.grid_list[device]]), axis=0) / (4.*np.pi * self.grid.volume)
            for dev in self.grid_list:
                for grid in self.grid_list[dev]:
                    grid.scattering[i] = total_scattering[i].clone().to(wp.device_to_torch(grid.device))

    def make_image(self, nx, ny, pixel_size, lam, incl, pa, dpc, device="cpu"):
        # First, run a scattering simulation to get the scattering phase function

        self.scattering_mc(100000, lam, device=device)

        # Now set up the image proper.

        for dev in self.camera_list:
            for camera in self.camera_list[dev]:
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
                        zip(self.camera_list[device], np.array_split(new_x, self.ncores), np.array_split(new_y, self.ncores))))).sum(axis=0)

        intensity += np.array(list(self.pool.map(lambda camera: camera.raytrace_sources(image.x, image.y, nx, ny, image.nu, dpc, 
                        nrays=int(1000/self.ncores)).numpy(), self.camera_list[device]))).mean(axis=0)

        image = image.assign(intensity=(("x","y","lam"), intensity))

        return image