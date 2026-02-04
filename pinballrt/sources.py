from pinballrt.utils import EPSILON
from .photons import PhotonList
from astropy.modeling import models
import astropy.units as u
import astropy.constants as const
import scipy.integrate
import warp as wp
import numpy as np
import torch
import time

class SphericalSource:
    def __init__(self, luminosity, frequency, intensity, x=0., y=0., z=0.):
        self.luminosity = luminosity
        self.x = x
        self.y = y
        self.z = z

        self.nu = frequency
        if hasattr(intensity, '__call__'):
            self.intensity = intensity
        else:
            self.log10_intensity_func = np.interp1d(np.log10(frequency.to(u.GHz).value), np.log10(intensity.value), kind='linear')
            self.intensity = lambda nu: 10**self.log10_intensity_func(np.log10(nu.to(u.GHz).value)) * intensity.unit

        self.radius = (self.luminosity / (4.*np.pi**2*u.steradian*scipy.integrate.trapezoid(self.intensity(self.nu), self.nu)))**0.5

        self.random_nu_CPD = scipy.integrate.cumulative_trapezoid(self.intensity(self.nu), self.nu, initial=0.)
        self.random_nu_CPD /= self.random_nu_CPD[-1]

    def emit(self, nphotons, distance_unit, wavelength="random", simulation="thermal", device="cpu", timing={}):
        theta = np.pi*np.random.rand(nphotons)
        phi = 2*np.pi*np.random.rand(nphotons)

        position = np.hstack(((self.radius.to(distance_unit).value*np.sin(theta)*np.cos(phi))[:,np.newaxis],
                             (self.radius.to(distance_unit).value*np.sin(theta)*np.sin(phi))[:,np.newaxis],
                             (self.radius.to(distance_unit).value*np.cos(theta))[:,np.newaxis]))

        r_hat = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).T
        theta_hat = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)]).T
        phi_hat = np.array([-np.sin(phi), np.cos(phi), np.zeros(nphotons)]).T

        cost = np.random.rand(nphotons)
        sint = np.sqrt(1-cost**2)
        phi = 2*np.pi*np.random.rand(nphotons)

        direction = cost[:,np.newaxis]*r_hat + (sint*np.cos(phi))[:,np.newaxis]*phi_hat + (sint*np.sin(phi))[:,np.newaxis]*theta_hat

        if wavelength == "random":
            t1 = time.time()
            frequency = self.random_nu(nphotons, device=device)
            t2 = time.time()
            timing["Random frequency generation"] = t2 - t1
        else:
            frequency = np.repeat((const.c / wavelength).to(u.GHz), nphotons).value

        if simulation == "thermal":
            photon_energy = np.repeat(self.luminosity.to(u.L_sun).value / nphotons, nphotons)
        elif simulation == "scattering":
            photon_energy = np.repeat((4.*np.pi**2*u.steradian*self.radius**2*self.intensity(frequency[0]*u.GHz)).to(distance_unit**2 * u.Jy).value / nphotons, nphotons)

        with wp.ScopedDevice(device):
            photon_list = PhotonList()
            photon_list.position = wp.array(position, dtype=wp.vec3)
            photon_list.direction = wp.array(direction, dtype=wp.vec3)
            photon_list.frequency = wp.array(frequency, dtype=float)
            photon_list.energy = wp.array(photon_energy, dtype=float)
            photon_list.in_grid = wp.ones(nphotons, dtype=bool)

        return photon_list
    
    def emit_rays(self, nu, distance_unit, ez, nrays, physical_pixel_size, device="cpu"):
        theta = np.pi*np.random.rand(nrays)
        phi = 2*np.pi*np.random.rand(nrays)

        position = np.hstack(((self.radius.to(distance_unit).value*np.sin(theta)*np.cos(phi))[:,np.newaxis],
                             (self.radius.to(distance_unit).value*np.sin(theta)*np.sin(phi))[:,np.newaxis],
                             (self.radius.to(distance_unit).value*np.cos(theta))[:,np.newaxis]))
        
        direction = np.tile(ez, (nrays, 1))

        intensity = (np.tile(self.intensity(nu.data)*np.pi, (nrays, 1)) / nrays).to(u.Jy / u.steradian).value * ((self.radius / physical_pixel_size).decompose()**2).value
        tau_intensity = np.zeros((nrays, nu.size), dtype=float)

        with wp.ScopedDevice(device):
            ray_list = PhotonList()
            ray_list.position = wp.array(position, dtype=wp.vec3)
            ray_list.direction = wp.array(direction, dtype=wp.vec3)
            ray_list.indices = wp.zeros(position.shape, dtype=int)
            ray_list.intensity = wp.array2d(intensity, dtype=float)
            ray_list.tau_intensity = wp.array2d(tau_intensity, dtype=float)
            ray_list.pixel_too_large = wp.zeros(nrays, dtype=bool)

            ray_list.radius = wp.array(np.zeros(nrays), dtype=float)
            ray_list.logradius = wp.array(np.zeros(nrays), dtype=float)
            ray_list.theta = wp.zeros(nrays, dtype=float)
            ray_list.phi = wp.zeros(nrays, dtype=float)
            ray_list.sin_theta = wp.zeros(nrays, dtype=float)
            ray_list.cos_theta = wp.zeros(nrays, dtype=float)
            ray_list.phi = wp.zeros(nrays, dtype=float)
            ray_list.sin_phi = wp.zeros(nrays, dtype=float)
            ray_list.cos_phi = wp.zeros(nrays, dtype=float)

        return ray_list

    def random_nu(self, nphotons, device="cpu"):
        ksi = np.random.rand(nphotons)

        with wp.ScopedDevice(device):
            random_nu = wp.zeros(nphotons, dtype=float)
            wp.launch(random_nu_kernel,
                      dim=(nphotons,),
                      inputs=[wp.array(ksi, dtype=float), wp.array(self.random_nu_CPD, dtype=float), wp.array(self.nu.value, dtype=float), random_nu, wp.array(np.arange(len(self.random_nu_CPD)), dtype=int), np.random.randint(0, 100000)])

        return random_nu
        
class BlackbodyStar(SphericalSource):
    def __init__(self, temperature=4000.*u.K, luminosity=1.0*const.L_sun, x=0., y=0., z=0., 
                 nu=np.logspace(0.5, 6.45, 1000)*u.GHz):
        self.temperature = temperature
        self.luminosity = luminosity
        self.radius = (self.luminosity / (4.*np.pi*const.sigma_sb*self.temperature**4))**0.5
        self.x = x
        self.y = y
        self.z = z

        self.nu = np.logspace(np.log10(nu.value.min()), np.log10(nu.value.max()), 1000) * nu.unit
        self.intensity = models.BlackBody(temperature=self.temperature)
        
        self.random_nu_CPD = scipy.integrate.cumulative_trapezoid(self.intensity(self.nu), self.nu, initial=0.)
        self.random_nu_CPD /= self.random_nu_CPD[-1]

class ExternalSource(SphericalSource):
    def __init__(self, grid, intensity, frequency=None):
        radius = grid.grid_size()*grid.distance_unit / 2.

        if frequency is None:
            frequency = np.logspace(np.log10(grid.dust.nu.value.min()), np.log10(grid.dust.nu.value.max()), 1000) * grid.dust.nu.unit
        self.grid = grid

        super().__init__(luminosity=4.*np.pi**2*u.steradian*radius**2*scipy.integrate.trapezoid(intensity(frequency), frequency),
                         frequency=frequency,
                         intensity=intensity,
                         x=0., y=0., z=0.)

    def emit(self, nphotons, distance_unit, wavelength="random", simulation="thermal", device="cpu", timing={}):
        photon_list = super().emit(nphotons, distance_unit, wavelength, simulation, device, timing)

        # Flip directions to point inward
        photon_list.direction = wp.array2d(-photon_list.direction.numpy(), dtype=wp.vec3)

        # Check the distance to the outer wall of the grid and move photons just inside
        s = wp.zeros(nphotons, dtype=float)

        wp.launch(kernel=self.grid.outer_wall_distance,
                dim=(nphotons,),
                inputs=[photon_list, self.grid.grid, s])
        s = wp.to_torch(s)
        will_be_in_grid = s < torch.inf
        iwill_be_in_grid = torch.arange(nphotons, dtype=torch.int32, device=wp.device_to_torch(wp.get_device()))[will_be_in_grid]
        wp.launch(kernel=self.grid.move,
                    dim=iwill_be_in_grid.shape,
                    inputs=[photon_list, s, iwill_be_in_grid])

        with wp.ScopedDevice(device):
            photon_list.position = wp.array(wp.to_torch(photon_list.position), dtype=wp.vec3)
            photon_list.direction = wp.array(wp.to_torch(photon_list.direction), dtype=wp.vec3)
            photon_list.frequency = wp.array(wp.to_torch(photon_list.frequency), dtype=float)
            photon_list.energy = wp.array(wp.to_torch(photon_list.energy), dtype=float)
            photon_list.in_grid = wp.from_torch(will_be_in_grid)

        return photon_list

class DiffuseSource:
    def __init__(self, grid, spectrum, density, frequency=None):
        self.grid = grid
        if density.ndim == 3:
            self.density = density
        else:
            self.density = np.tile(density, self.grid.shape)

        if callable(spectrum):
            if frequency is None:
                self.frequency = np.logspace(np.log10(self.grid.dust.nu.value.min()), np.log10(self.grid.dust.nu.value.max()), 1000) * self.grid.dust.nu.unit
            else:
                self.frequency = frequency
            self.spectrum = spectrum(self.frequency)
            self.intensity = spectrum
        else:
            if frequency is None:
                raise ValueError("Frequency array must be provided if spectrum is not callable.")
            self.frequency = frequency
            self.spectrum = spectrum
            self.log10_intensity_func = np.interp1d(np.log10(self.frequency.to(u.GHz).value), np.log10(self.spectrum.value), kind='linear')
            self.intensity = lambda nu: 10**self.log10_intensity_func(np.log10(nu.to(u.GHz).value)) * self.spectrum.unit

        self.total_luminosity = ((self.grid.volume*self.grid.distance_unit**3 * density).sum() *scipy.integrate.trapezoid(self.intensity(self.frequency), self.frequency)).to(u.L_sun)

        self.random_nu_CPD = scipy.integrate.cumulative_trapezoid(self.intensity(self.frequency), self.frequency, initial=0.)
        self.random_nu_CPD /= self.random_nu_CPD[-1]

    def initialize_luminosity_array(self, wavelength):
        if wavelength == "random":
            self.luminosity = self.total_luminosity.to(u.L_sun) * self.density / self.density.sum()
            self.total_lum = self.total_luminosity.to(u.L_sun)
        else:
            frequency = (const.c / wavelength).to(u.GHz)
            self.luminosity = (self.density * self.grid.volume * self.grid.distance_unit**3 * self.intensity(frequency)).to(self.grid.distance_unit**2 * u.Jy).value
            self.total_lum = self.luminosity.sum()

    def emit(self, nphotons, distance_unit, wavelength="random", simulation="thermal", device="cpu", timing={}):
        ksi = np.random.rand(nphotons)
        if self.luminosity.sum() == 0:
            self.luminosity += EPSILON
        cum_lum = np.cumsum(self.luminosity.flatten()).reshape(self.grid.shape) / self.luminosity.sum()

        cell_coords = []
        for i in range(nphotons):
            cell_coords += [np.where(cum_lum[cum_lum > ksi[i]].min() == cum_lum)]
        
        with wp.ScopedDevice(device):
            photon_list = PhotonList()

            cell_coords = wp.array2d(np.array(cell_coords)[:,:,0], dtype=int)
            
            photon_list.position = wp.array(np.zeros((nphotons,3)), dtype=wp.vec3)
            wp.launch(kernel=self.grid.random_location_in_cell, 
                    dim=(nphotons,), 
                    inputs=[photon_list.position, cell_coords, self.grid.grid, np.random.randint(0, 100000)])

            photon_list.direction = wp.array(np.zeros((nphotons, 3)), dtype=wp.vec3)
            wp.launch(kernel=self.grid.random_direction,
                        dim=(nphotons,),
                        inputs=[photon_list.direction, torch.arange(nphotons, dtype=torch.int32, device=wp.device_to_torch(wp.get_device())), np.random.randint(0, 100000)])
            
            if wavelength == "random":
                t1 = time.time()
                photon_list.frequency = self.random_nu(nphotons, cell_coords)
                t2 = time.time()
                timing["Random frequency generation"] = t2 - t1
            else:
                photon_list.frequency = wp.array(np.repeat((const.c / wavelength).to(u.GHz).value, nphotons), dtype=float)
            
            photon_list.energy = wp.array(np.repeat(self.total_lum/nphotons, nphotons).astype(np.float32), dtype=float)
            photon_list.in_grid = wp.ones(nphotons, dtype=bool)

        return photon_list

    def random_nu(self, nphotons, cell_coords):
        ksi = np.random.rand(nphotons)

        random_nu = wp.zeros(nphotons, dtype=float)
        wp.launch(random_nu_kernel,
                    dim=(nphotons,),
                    inputs=[wp.array(ksi, dtype=float), wp.array(self.random_nu_CPD, dtype=float), wp.array(self.frequency.value, dtype=float), random_nu, wp.array(np.arange(len(self.random_nu_CPD)), dtype=int), np.random.randint(0, 100000)])

        return random_nu

class GridSource(DiffuseSource):
    def __init__(self, grid):
        self.grid = grid

    def initialize_luminosity_array(self, wavelength):
        if wavelength == "random":
            self.luminosity = (4*const.sigma_sb.cgs*self.grid.dust.planck_mean_opacity(self.grid.grid.temperature.numpy(), self.grid)*u.cm**2/u.g * \
                    self.mass * self.grid.grid.temperature.numpy()*u.K**4).to(u.L_sun)
        else:
            nu = (const.c / wavelength).to(u.GHz)
            self.luminosity = np.zeros(self.grid.shape)

            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    for k in range(self.grid.shape[2]):
                        self.luminosity[i,j,k] = (4*np.pi*u.steradian*self.grid.grid.density.numpy()[i,j,k]*self.grid.volume.cpu().numpy()[i,j,k]*self.grid.dust.interpolate_kabs(nu)*self.grid.distance_unit**2*models.BlackBody(temperature=self.grid.grid.temperature.numpy()[i,j,k]*u.K)(nu)).to(u.au**2 * u.Jy).value

        self.total_lum = self.luminosity.sum()

    def random_nu(self, nphotons, cell_coords):
        photon_list = PhotonList()

        photon_list.indices = cell_coords

        photon_list.density = wp.zeros(nphotons, dtype=float)
        photon_list.temperature = wp.zeros(nphotons, dtype=float)
        photon_list.amax = wp.zeros(nphotons, dtype=float)
        photon_list.p = wp.zeros(nphotons, dtype=float)
        wp.launch(kernel=self.grid.photon_cell_properties,
                    dim=(nphotons,),
                    inputs=[photon_list, self.grid.grid, wp.array(np.arange(nphotons), dtype=int)])

        return self.grid.dust.random_nu(photon_list)

class EnergySource(GridSource):
    def __init__(self, grid, energy_density):
        super().__init__(grid)
        self.energy_density = energy_density
        self.luminosity = energy_density * self.grid.volume * self.grid.distance_unit**3
        self.total_lum = self.luminosity.sum()

    def initialize_luminosity_array(self, wavelength):
        return

    def emit(self, nphotons, distance_unit, wavelength="random", simulation="thermal", device="cpu", timing={}):
        photon_list = super().emit(nphotons, distance_unit, wavelength, simulation, device, timing)

        self.grid.grid.energy = wp.array3d(self.luminosity.to(u.L_sun).value + self.grid.grid.energy.numpy(), dtype=float)

        return photon_list
    
@wp.kernel
def random_nu_kernel(ksi: wp.array(dtype=float),
                        random_nu_CPD: wp.array(dtype=float),
                        nu: wp.array(dtype=float),
                        random_nu: wp.array(dtype=float),
                        iCPD: wp.array(dtype=int),
                        seed: int): # pragma: no cover
    ip = wp.tid()
    rng = wp.rand_init(seed, ip)
    
    index = len(random_nu_CPD) - 1
    # Find the index where ksi[ip] is less than random_nu_CPD[index]
    for i in range(len(random_nu_CPD)):
        if ksi[ip] < random_nu_CPD[i]:
            index = i
            break

    dCPD = random_nu_CPD[index] - random_nu_CPD[index-1]

    if dCPD < EPSILON:
        random_nu[ip] = (nu[index] - nu[index-1]) * wp.randf(rng) + nu[index-1]
    else:
        random_nu[ip] = (ksi[ip] - random_nu_CPD[index-1]) * (nu[index] - nu[index-1]) / \
                dCPD + nu[index-1]