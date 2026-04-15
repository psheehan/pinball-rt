"""
One critical component of Monte Carlo radiative transfer are sources of photons to be propagated through the model. Sources are added by
initializing a source object and adding it to the model using the :meth:`add_sources<pinballrt.model.Model.add_sources>` method. For example:

.. code-block:: python

   from pinballrt import Model
   from pinballrt.sources import BlackbodyStar
   
   star = BlackbodyStar()

   model = Model()
   model.add_sources(star)

Multiple sources can be added in a single call by providing a list of sources: :code:`model.add_sources([star1, star2])`. 
pinball-rt implements four different types of sources (and one special instance of those types), which are described in 
the following sections.
"""

from pinballrt.utils import EPSILON
from .photons import PhotonList
from .utils import GridStruct, random_direction
from astropy.modeling import models
import astropy.units as u
import astropy.constants as const
import scipy.integrate
import warp as wp
import numpy as np
import torch
import time

class SphericalSource:
    r"""
    Sphercal sources, e.g. stars, emit photons from the surface of a sphere in a random outward direction. They can be 
    created using the `SphericalSource` class, which takes the total luminosity of the source as well as the spectrum of the source
    as input. The spectrum must be specified with astropy units that can be converted to units of `u.Jy / u.steradian`. For example:

    .. code-block:: python

        from pinballrt.sources import SphericalSource
        import astropy.units as u
        from astropy.modeling import models

        frequency = np.logspace(9, 15, 100) * u.Hz
        spectrum = models.BlackBody(temperature=5000*u.K)(frequency)

        star = SphericalSource(luminosity=1.0e4*u.Lsun, frequency, spectrum)

    The radius of the spherical source is determined by solving

    .. math:: L = 4 \pi^2 R^2 \int I_{\nu} d\nu

    where :math:`L` is the luminosity of the source, :math:`R` is the radius of the source, and :math:`I_\nu` is the specific intensity of 
    the source at frequency :math:`\nu`, and is provided as the spectrum above.
    """
    def __init__(self, luminosity, frequency, intensity, x=0., y=0., z=0.):
        """
        Parameters
        ----------
        luminosity : `astropy.units.Quantity`
            The total luminosity of the source.
        frequency : `astropy.units.Quantity`
            The frequency array over which the intensity is defined.
        intensity : `astropy.units.Quantity` or callable
            The intensity as a function of frequency. If callable, it should take a frequency array as input and return the intensity at those frequencies in units that are compatible with Jy / steradian.
        x, y, z : float
            The position of the center of the source in the simulation grid.
        """
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
        direction_frame = cost[:,np.newaxis]*r_hat + (sint*np.cos(phi))[:,np.newaxis]*phi_hat + (sint*np.sin(phi))[:,np.newaxis]*theta_hat

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
            photon_list.direction_frame = wp.array(direction_frame, dtype=wp.vec3)
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
            ray_list.direction_frame = wp.array(direction, dtype=wp.vec3)
            ray_list.indices = wp.zeros(position.shape, dtype=int)
            ray_list.intensity = wp.array2d(intensity, dtype=float)
            ray_list.tau_intensity = wp.array2d(tau_intensity, dtype=float)
            ray_list.pixel_too_large = wp.zeros(nrays, dtype=bool)

            ray_list.density = wp.zeros(nrays, dtype=float)
            ray_list.temperature = wp.zeros(nrays, dtype=float)
            ray_list.amax = wp.zeros(nrays, dtype=float)
            ray_list.p = wp.zeros(nrays, dtype=float)

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
        r"""
        A spherical blackbody star emitting photons isotropically from its surface. BlackbodyStar is a special case of 
        a :meth:`SphericalSource<pinballrt.sources.SphericalSource>` where the intensity is given by the Planck function 
        for a specified temperature. The radius of the star is determined by the luminosity and temperature as described 
        in the :meth:`SphericalSource` class, but using the known solution that
        
        .. math:: \int B_{\nu} d\nu = \sigma_{SB} T^4 / \pi.

        Parameters
        ----------
        temperature : `astropy.units.Quantity`
            The temperature of the blackbody star.
        luminosity : `astropy.units.Quantity`
            The total luminosity of the star.
        x, y, z : float
            The position of the center of the star in the simulation grid.
        nu : `astropy.units.Quantity`
            The frequency array over which the blackbody intensity is defined.
        """
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
        """
        An external isotropic radiation source surrounding the simulation grid. External sources emit photons inward from 
        a sphere just beyond the outer boundary of the grid. It can be specified in terms of the specific intensity as a 
        function of frequency, which can be expressed either as a function, e.g.

        .. code-block:: python

            from astropy.modeling import models
            from pinballrt.sources import ExternalSource
            import astropy.units as u

            source = ExternalSource(grid, intensity=models.BlackBody(temperature=10*u.K))

        (where the :meth:`models.BlackBody` class is callable and returns the intensity at the specified frequencies) or 
        as an array of intensities at specified frequencies, e.g.

        .. code-block:: python

            from astropy.modeling import models
            from pinballrt.sources import ExternalSource
            import astropy.units as u
            import numpy as np

            frequency = np.logspace(9, 15, 100) * u.Hz
            intensity = models.BlackBody(temperature=10*u.K)(frequency)

            source = ExternalSource(grid, intensity, frequency)

        Parameters
        ----------
        grid : `pinballrt.sources.Grid`
            The simulation grid.
        intensity : `astropy.units.Quantity` or callable
            The intensity as a function of frequency. If callable, it should take a frequency array as input and return the intensity at those frequencies in units that are compatible with Jy / steradian.
        frequency : `astropy.units.Quantity`, optional
            The frequency array over which the intensity is defined. If not provided, it will be generated based on the grid's dust properties.
        """
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
        r"""
        A diffuse source emitting photons from within the simulation grid. Diffuse sources emit photons from random locations 
        withing the grid, with a probability of emission from each cell proportional to the luminosity of that cell. 
        The luminosity of each cell is determined by the product of the density, the spectrum, and the cell volume. 
        The spectrum can be specified as a function or as an array of values at specified frequencies, similar to the 
        `ExternalSource` class, and should be in units such that when multiplied by the density and cell volume, the 
        result can be converted to ergs / s / Hz. For example, a uniform distribution of single-temperature blackbody stars could
        be created with:

        .. code-block:: python

            from pinballrt.sources import DiffuseSource
            import astropy.units as u
            import numpy as np

            spectrum = lambda nu: 4*np.pi**2 * u.steradian * (0.035*u.R_sun)**2 * models.BlackBody(2000.*u.K)(nu)
            density = u.g / u.cm**3

            diffuse_source = DiffuseSource(model.grid, spectrum, density)

        One could, of course, specify a more complex spectrum, for example a distribution of blackbody stars with some 
        mass function would mathematically be:

        .. math:: I_{\nu} = \int 4 \pi^2 R(M)^2 B_{\nu}(T(M)) \frac{dN}{dM} dM

        Parameters
        ----------
        grid : `pinballrt.sources.Grid`
            The simulation grid.
        spectrum : `astropy.units.Quantity` or callable
            The spectrum as a function of frequency. If callable, it should take a frequency array as input and return the spectrum at those frequencies in units such that when
            multiplied by the density and cell volume that are compatible with ergs / s / Hz.
        density : `astropy.units.Quantity`
            The density distribution of the diffuse source within the grid. It should be specified in units such that when the spectrum, density, and cell volume are multiplied, the result is in ergs / s / Hz.
        frequency : `astropy.units.Quantity`, optional
            The frequency array over which the spectrum is defined. If not provided, it will be generated based on the grid's dust properties.
        """
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

        self.total_luminosity = ((self.grid.volume.cpu().numpy()*self.grid.distance_unit**3 * density).sum() *scipy.integrate.trapezoid(self.intensity(self.frequency), self.frequency)).to(u.L_sun)

        self.random_nu_CPD = scipy.integrate.cumulative_trapezoid(self.intensity(self.frequency), self.frequency, initial=0.)
        self.random_nu_CPD /= self.random_nu_CPD[-1]

    def initialize_luminosity_array(self, wavelength):
        if wavelength == "random":
            self.luminosity = self.total_luminosity.to(u.L_sun) * self.density / self.density.sum()
            self.total_lum = self.total_luminosity.to(u.L_sun)
        else:
            frequency = (const.c / wavelength).to(u.GHz)
            self.luminosity = (self.density * self.grid.volume.cpu().numpy() * self.grid.distance_unit**3 * self.intensity(frequency)).to(self.grid.distance_unit**2 * u.Jy).value
            self.total_lum = self.luminosity.sum()

    @wp.kernel
    def random_cell_from_cum_lum(grid: GridStruct,
                                 cum_lum: wp.array3d(dtype=float),
                                 cell_coords: wp.array2d(dtype=int),
                                 seed: int):
        
        ip = wp.tid()

        rng = wp.rand_init(seed, ip)

        ksi = wp.randf(rng)

        for i in range(grid.n1):
            for j in range(grid.n2):
                for k in range(grid.n3):
                    if ksi < cum_lum[i, j, k]:
                        cell_coords[ip][0] = i
                        cell_coords[ip][1] = j
                        cell_coords[ip][2] = k
                        break
                if ksi < cum_lum[i, j, k]:
                    break
            if ksi < cum_lum[i, j, k]:
                break

    def emit(self, nphotons, distance_unit, wavelength="random", simulation="thermal", device="cpu", timing={}):
        if self.luminosity.sum() == 0:
            self.luminosity += EPSILON
        cum_lum = np.cumsum(self.luminosity.flatten()).reshape(self.grid.shape) / self.luminosity.sum()
        
        with wp.ScopedDevice(device):
            photon_list = PhotonList()

            cell_coords = wp.array2d(np.zeros((nphotons, 3)), dtype=int)
            wp.launch(kernel=self.random_cell_from_cum_lum,
                        dim=(nphotons,),
                        inputs=[self.grid.grid, wp.array3d(cum_lum, dtype=float), cell_coords, np.random.randint(0, 100000)])
            
            photon_list.position = wp.array(np.zeros((nphotons,3)), dtype=wp.vec3)
            wp.launch(kernel=self.grid.random_location_in_cell, 
                    dim=(nphotons,), 
                    inputs=[photon_list.position, cell_coords, self.grid.grid, np.random.randint(0, 100000)])

            photon_list.direction = wp.array(np.zeros((nphotons, 3)), dtype=wp.vec3)
            photon_list.direction_frame = wp.array(np.zeros((nphotons, 3)), dtype=wp.vec3)
            wp.launch(kernel=random_direction,
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

            with wp.ScopedDevice(self.grid.device):
                self.luminosity = (4*np.pi*u.steradian*\
                                   self.grid.grid.dust_density.numpy()*\
                                   self.grid.volume.cpu().numpy()*\
                                   self.grid.dust.ml_kabs(wp.to_torch(self.grid.grid.p).flatten(), 
                                                          wp.to_torch(self.grid.grid.amax).flatten(), 
                                                          torch.tensor(nu.value, dtype=torch.float32, device=wp.device_to_torch(wp.get_device())).expand(np.prod(self.grid.shape))).cpu().numpy().reshape(self.grid.shape)*\
                                   self.grid.distance_unit**2*models.BlackBody(temperature=self.grid.grid.temperature.numpy()*u.K)(nu)).to(u.au**2 * u.Jy).value

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
        """
        A diffuse energy source that directly injects energy into the grid based on a specified energy density, and then the grid
        reradiates that energy away based on the temperature and dust properties in the cell. They should be specified in terms of
        the luminosity per volume that is being injected into the grid, in units convertible to ergs / s / cm^3. For example, a 
        uniform energy density could be created with:

        .. code-block:: python

            from pinballrt.sources import EnergySource
            import astropy.units as u

            energy_density = 1e-15 * u.erg / u.s / u.cm**3

            energy_source = EnergySource(model.grid, energy_density)

        Parameters
        ----------
        grid : `pinballrt.sources.Grid`
            The simulation grid.
        energy_density : `astropy.units.Quantity`
            The energy density distribution of the source within the grid. It should be specified in units such that when multiplied by the cell volume, the result is compatible with ergs / s.
        """
        super().__init__(grid)
        self.energy_density = energy_density
        self.luminosity = energy_density * self.grid.volume.cpu().numpy() * self.grid.distance_unit**3
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