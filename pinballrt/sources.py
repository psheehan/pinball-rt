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
        
class BlackbodyStar:
    x: float
    y: float
    z: float
    temperature: float
    luminosity: float
    radius: float

    def __init__(self, temperature=4000.*u.K, luminosity=1.0*const.L_sun, x=0., y=0., z=0., 
                 nu=np.logspace(0.5, 6.45, 1000)*u.GHz):
        self.temperature = temperature
        self.luminosity = luminosity
        self.radius = (self.luminosity / (4.*np.pi*const.sigma_sb*self.temperature**4))**0.5
        self.x = x
        self.y = y
        self.z = z

        self.nu = np.logspace(np.log10(nu.value.min()), np.log10(nu.value.max()), 1000) * nu.unit

        self.flux = models.BlackBody(temperature=self.temperature)
        
        self.Bnu = self.flux(self.nu)

        self.random_nu_CPD = scipy.integrate.cumulative_trapezoid(self.Bnu, self.nu, initial=0.)
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
            photon_energy = np.repeat((4.*np.pi**2*u.steradian*self.radius**2*models.BlackBody(temperature=self.temperature)(frequency[0]*u.GHz)).to(distance_unit**2 * u.Jy).value / nphotons, nphotons)

        with wp.ScopedDevice(device):
            photon_list = PhotonList()
            photon_list.position = wp.array(position, dtype=wp.vec3)
            photon_list.direction = wp.array(direction, dtype=wp.vec3)
            photon_list.frequency = wp.array(frequency, dtype=float)
            photon_list.energy = wp.array(photon_energy, dtype=float)

        return photon_list
    
    def emit_rays(self, nu, distance_unit, ez, nrays, physical_pixel_size, device="cpu"):
        theta = np.pi*np.random.rand(nrays)
        phi = 2*np.pi*np.random.rand(nrays)

        position = np.hstack(((self.radius.to(distance_unit).value*np.sin(theta)*np.cos(phi))[:,np.newaxis],
                             (self.radius.to(distance_unit).value*np.sin(theta)*np.sin(phi))[:,np.newaxis],
                             (self.radius.to(distance_unit).value*np.cos(theta))[:,np.newaxis]))
        
        direction = np.tile(ez, (nrays, 1))

        intensity = (np.tile(self.flux(nu.data)*np.pi, (nrays, 1)) / nrays).to(u.Jy / u.steradian).value * ((self.radius / physical_pixel_size).decompose()**2).value
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
            wp.launch(self.random_nu_kernel,
                      dim=(nphotons,),
                      inputs=[wp.array(ksi, dtype=float), wp.array(self.random_nu_CPD, dtype=float), wp.array(self.nu.value, dtype=float), random_nu, wp.array(np.arange(len(self.random_nu_CPD)), dtype=int), np.random.randint(0, 100000)])

        return random_nu
    
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

class GridSource:
    def __init__(self, grid):
        self.grid = grid

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

        return photon_list

    def initialize_luminosity_array(self, wavelength):
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

class DistributionOfBlackbodyStars(GridSource):
    def __init__(self, grid, temperature=2000.*u.K, total_luminosity=1.0*const.L_sun, number_density=10*u.au**-3):
        self.grid = grid
        self.temperature = temperature
        self.total_luminosity = total_luminosity
        if number_density.size == 1:
            self.number_density = np.tile(number_density, self.grid.shape)
        else:
            self.number_density = number_density
            
        self.nstars = (self.number_density * (self.grid.volume*self.grid.distance_unit**3).to(u.pc**3)).decompose()

        luminosity_per_star = self.total_luminosity / self.nstars.sum()
        self.stellar_radius = (luminosity_per_star / (4.*np.pi*const.sigma_sb*self.temperature**4))**0.5

        self.nu = np.logspace(np.log10(self.grid.dust.nu.value.min()), np.log10(self.grid.dust.nu.value.max()), 1000) * self.grid.dust.nu.unit

        self.flux = models.BlackBody(temperature=self.temperature)
        
        self.Bnu = self.flux(self.nu)

        self.random_nu_CPD = scipy.integrate.cumulative_trapezoid(self.Bnu, self.nu, initial=0.)
        self.random_nu_CPD /= self.random_nu_CPD[-1]

    def initialize_luminosity_array(self, wavelength):
        if wavelength == "random":
            self.luminosity = self.total_luminosity.to(u.L_sun) * self.number_density / self.number_density.sum()
            self.total_lum = self.total_luminosity.to(u.L_sun)
        else:
            frequency = (const.c / wavelength).to(u.GHz)
            self.luminosity = self.nstars * (4.*np.pi**2*u.steradian*self.stellar_radius**2*models.BlackBody(temperature=self.temperature)(frequency)).to(self.grid.distance_unit**2 * u.Jy).value
            self.total_lum = self.luminosity.sum()

    def random_nu(self, nphotons, cell_coords):
        ksi = np.random.rand(nphotons)

        random_nu = wp.zeros(nphotons, dtype=float)
        wp.launch(self.random_nu_kernel,
                    dim=(nphotons,),
                    inputs=[wp.array(ksi, dtype=float), wp.array(self.random_nu_CPD, dtype=float), wp.array(self.nu.value, dtype=float), random_nu, wp.array(np.arange(len(self.random_nu_CPD)), dtype=int), np.random.randint(0, 100000)])

        return random_nu
    
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