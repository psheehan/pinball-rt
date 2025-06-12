from .photons import PhotonList
from astropy.modeling import models
import astropy.units as u
import astropy.constants as const
import warp as wp
import numpy as np
import time

from .utils import EPSILON, equal, equal_zero, planck_function

@wp.struct
class GridStruct:
    w1: wp.array(dtype=float)
    w2: wp.array(dtype=float)
    w3: wp.array(dtype=float)
    n1: int
    n2: int
    n3: int

    sin_w2: wp.array(dtype=float)
    cos_w2: wp.array(dtype=float)
    tan_w2: wp.array(dtype=float)
    neg_mu: wp.array(dtype=float)
    sin_tol_w2: wp.array(dtype=float)
    cos_tol_w2: wp.array(dtype=float)
    sin_w3: wp.array(dtype=float)
    cos_w3: wp.array(dtype=float)

    mirror_symmetry: bool

    density: wp.array3d(dtype=float)
    temperature: wp.array3d(dtype=float)
    energy: wp.array3d(dtype=float)

class Grid:
    def __init__(self, _w1, _w2, _w3):
        self.grid = GridStruct()
        self.grid.w1 = wp.array(_w1, dtype=float)
        self.grid.w2 = wp.array(_w2, dtype=float)
        self.grid.w3 = wp.array(_w3, dtype=float)
        self.grid.n1 = _w1.size-1
        self.grid.n2 = _w2.size-1
        self.grid.n3 = _w3.size-1

        self.shape = (self.grid.n1, self.grid.n2, self.grid.n3)

    def add_density(self, _density, dust):
        self.grid.density = wp.array3d((_density*dust.kmean).to(1./self.distance_unit).value, dtype=float)

        self.grid.energy = wp.zeros(_density.shape, dtype=float)
        self.grid.temperature = wp.array3d(np.ones(_density.shape)*0.1, dtype=float)
        self.mass = (_density * self.volume * self.distance_unit**3).decompose()

        self.dust = dust

    def add_star(self, star):
        self.star = star

    def base_emit(self, nphotons, wavelength="random", scattering=False):
        if scattering:
            nphotons_per_source = int(nphotons / 2)
        else:
            nphotons_per_source = nphotons

        photon_list = self.star.emit(nphotons_per_source, self.distance_unit, wavelength, simulation="scattering" if scattering else "thermal")

        cell_coords = []
        if scattering:
            ksi = np.random.rand(nphotons_per_source)
            cum_lum = np.cumsum(self.luminosity.flatten()).reshape(self.shape) / self.total_lum

            for i in range(nphotons_per_source):
                cell_coords += [np.where(cum_lum[cum_lum > ksi[i]].min() == cum_lum)]
            cell_coords = np.array(cell_coords)[:,:,0].astype(np.int32)

            new_position = wp.array(np.zeros((nphotons_per_source,3)), dtype=wp.vec3)
            wp.launch(kernel=self.random_location_in_cell, 
                      dim=(nphotons_per_source,), 
                      inputs=[new_position, cell_coords, self.grid], 
                      device='cpu')
            
            photon_list.position = wp.array(np.concatenate((photon_list.position.numpy(), new_position), axis=0), dtype=wp.vec3)
            
            new_direction = wp.array(np.zeros((nphotons_per_source, 3)), dtype=wp.vec3)
            wp.launch(kernel=self.random_direction,
                      dim=(nphotons_per_source,),
                      inputs=[new_direction, np.arange(nphotons_per_source, dtype=np.int32)],
                      device='cpu')
            photon_list.direction = wp.array(np.concatenate((photon_list.direction.numpy(), new_direction), axis=0), dtype=wp.vec3)

            photon_list.frequency = wp.array(np.concatenate([photon_list.frequency, np.repeat((const.c / wavelength).to(u.GHz).value, nphotons_per_source)]), dtype=float)
            photon_list.energy = wp.array(np.concatenate([photon_list.energy, np.repeat(self.total_lum/nphotons_per_source, nphotons_per_source).astype(np.float32)]), dtype=float)

        return photon_list
    
    @wp.kernel
    def tau_distance_absorption(grid: GridStruct,
                     photon_list: PhotonList, 
                     distances: wp.array(dtype=float),
                     iphotons: wp.array(dtype=int)):

        ip = iphotons[wp.tid()]

        ix, iy, iz = photon_list.indices[ip][0], photon_list.indices[ip][1], photon_list.indices[ip][2]

        photon_list.alpha[ip] = grid.density[ix,iy,iz] * photon_list.kabs[ip]
        distances[ip] = photon_list.tau[ip] / photon_list.alpha[ip]

    @wp.kernel
    def tau_distance_scattering(grid: GridStruct,
                     photon_list: PhotonList,
                     distances: wp.array(dtype=float),
                     iphotons: wp.array(dtype=int)):

        ip = iphotons[wp.tid()]

        ix, iy, iz = photon_list.indices[ip][0], photon_list.indices[ip][1], photon_list.indices[ip][2]

        photon_list.alpha[ip] = grid.density[ix,iy,iz] * photon_list.ksca[ip]
        distances[ip] = photon_list.tau[ip] / photon_list.alpha[ip]

    @wp.kernel
    def move(photon_list: PhotonList,
             distances: wp.array(dtype=float),
             iray: wp.array(dtype=int)):
    
        ip = iray[wp.tid()]

        photon_list.position[ip][0] = photon_list.position[ip][0] + distances[ip] * photon_list.direction[ip][0]
        photon_list.position[ip][1] = photon_list.position[ip][1] + distances[ip] * photon_list.direction[ip][1]
        photon_list.position[ip][2] = photon_list.position[ip][2] + distances[ip] * photon_list.direction[ip][2]

    @wp.kernel
    def deposit_energy(photon_list: PhotonList,
                       grid: GridStruct,
                       distances: wp.array(dtype=float),
                       iphotons: wp.array(dtype=int)):
        """
        Deposit energy in the grid based on the absorption of photons.

        Parameters
        ----------
        indices : wp.array2d(dtype=int)
            Indices of the grid cells where the photons are located.
        """

        ip = iphotons[wp.tid()]

        ix, iy, iz = photon_list.indices[ip][0], photon_list.indices[ip][1], photon_list.indices[ip][2]

        if photon_list.absorb[ip]:
            grid.energy[ix,iy,iz] += 10.**(wp.log10(photon_list.energy[ip]) + wp.log10(distances[ip]) + wp.log10(photon_list.kabs[ip]) + wp.log10(grid.density[ix,iy,iz]))

    @wp.kernel
    def reduce_tau(photon_list: PhotonList,
                   distances: wp.array(dtype=float),
                   iphotons: wp.array(dtype=int)):
        """
        Reduce the optical depth of the photons based on the distances they have traveled.
        """
        ip = iphotons[wp.tid()]

        photon_list.tau[ip] -= distances[ip] * photon_list.alpha[ip]

    @wp.kernel
    def deposit_scattering(photon_list: PhotonList,
                       distances: wp.array(dtype=float),
                       scattering: wp.array3d(dtype=float),
                       iphotons: wp.array(dtype=int)):

        ip = iphotons[wp.tid()]

        ix, iy, iz = photon_list.indices[ip][0], photon_list.indices[ip][1], photon_list.indices[ip][2]

        tau_abs = distances[ip] * photon_list.alpha[ip] * photon_list.kabs[ip] / photon_list.ksca[ip]
        tau_scat = distances[ip] * photon_list.alpha[ip]

        if tau_abs < EPSILON:
            average_energy = (1. - 0.5*tau_abs) * photon_list.energy[ip]
        else:
            average_energy = (1.0 - np.exp(-tau_abs)) / tau_abs * photon_list.energy[ip]

        scattering[ix,iy,iz] += average_energy * distances[ip]

        photon_list.energy[ip] = photon_list.energy[ip] * wp.exp(-tau_abs)
        photon_list.tau[ip] -= tau_scat
        photon_list.total_tau_abs[ip] += tau_abs

    @wp.kernel
    def photon_temperature(photon_list: PhotonList,
                           temperature: wp.array3d(dtype=float),
                           photon_temperature: wp.array(dtype=float),
                           iphotons: wp.array(dtype=int)):
        itemp = wp.tid()
        ip = iphotons[itemp]

        ix, iy, iz = photon_list.indices[ip][0], photon_list.indices[ip][1], photon_list.indices[ip][2]

        photon_temperature[itemp] = temperature[ix, iy, iz]

    @wp.kernel
    def update_frequency(photon_list: PhotonList,
                         frequency: wp.array(dtype=float),
                         kabs: wp.array(dtype=float),
                         ksca: wp.array(dtype=float),
                         iphotons: wp.array(dtype=int)):
        
        i = wp.tid()
        ip = iphotons[i]

        photon_list.frequency[ip] = frequency[i]
        photon_list.kabs[ip] = kabs[i]
        photon_list.ksca[ip] = ksca[i]
        photon_list.albedo[ip] = ksca[i] / (kabs[i] + ksca[i])

    @wp.kernel
    def random_direction(direction: wp.array(dtype=wp.vec3),
                         iphotons: wp.array(dtype=int)):
        i = wp.tid()
        ip = iphotons[i]

        rng = wp.rand_init(4321, i)

        cost = -1. + 2.*wp.randf(rng)
        sint = wp.sqrt(1.-cost**2.)
        phi = 2.*np.pi*wp.randf(rng)

        direction[ip][0] = sint*np.cos(phi)
        direction[ip][1] = sint*np.sin(phi)
        direction[ip][2] = cost

    @wp.kernel
    def random_tau(photon_list: PhotonList,
                   iphotons: wp.array(dtype=int)):
        i = wp.tid()
        ip = iphotons[i]

        rng = wp.rand_init(1234, i)

        photon_list.tau[ip] = -wp.log(1. - wp.randf(rng))

    @wp.kernel
    def random_absorb(photon_list: PhotonList,
                      iphotons: wp.array(dtype=int)):
        i = wp.tid()
        ip = iphotons[i]

        rng = wp.rand_init(1234, i)

        photon_list.absorb[ip] = wp.randf(rng) < photon_list.albedo[ip]

    def interact(self, photon_list: PhotonList, absorb, iabsorb, interact, iphotons, scattering=False):
        nphotons = iphotons.size

        wp.launch(kernel=self.random_direction,
                  dim=(nphotons,),
                  inputs=[photon_list.direction, iphotons],
                  device='cpu')

        t1 = time.time()
        nabsorb = iabsorb.size
        if not scattering and nabsorb > 0:
            photon_temperature = wp.zeros(nabsorb, dtype=float)
            wp.launch(kernel=self.photon_temperature,
                      dim=(nabsorb,),
                      inputs=[photon_list, self.grid.temperature, photon_temperature, iabsorb],
                      device='cpu')
        t2 = time.time()
        photon_temperature_time = t2 - t1

        if not scattering and nabsorb > 0:
            new_frequency = self.dust.random_nu(photon_temperature.numpy())

        t1 = time.time()
        if not scattering and nabsorb > 0:
            wp.launch(kernel=self.update_frequency,
                      dim=(nabsorb,),
                      inputs=[photon_list, new_frequency, self.dust.interpolate_kabs(new_frequency*u.GHz).astype(np.float32), self.dust.interpolate_ksca(new_frequency*u.GHz).astype(np.float32), iabsorb],
                      device='cpu')
        t2 = time.time()
        dust_interpolation_time = t2 - t1
    
        wp.launch(kernel=self.random_tau, dim=(nphotons,), inputs=[photon_list, iphotons], device='cpu')
        if not scattering:
            wp.launch(kernel=self.random_absorb, dim=(nphotons,), inputs=[photon_list, iphotons], device='cpu')

        t1 = time.time()
        wp.launch(kernel=self.photon_loc,
                  dim=(interact.sum(),),
                  inputs=[photon_list, self.grid, iphotons],
                  device='cpu')
        t2 = time.time()
        photon_loc_time = t2 - t1

        return photon_temperature_time, dust_interpolation_time, photon_loc_time

    def update_grid(self):
        total_energy = self.grid.energy.numpy()
        temperature = self.grid.temperature.numpy().copy()

        converged = False
        while not converged:
            old_temperature = temperature.copy()

            temperature = ((total_energy*u.L_sun).cgs.value / (4*const.sigma_T.cgs.value*\
                    self.dust.planck_mean_opacity(old_temperature)*\
                    self.mass.cgs.value))**0.25

            temperature[temperature < 0.1] = 0.1

            if (np.abs(old_temperature - temperature) / old_temperature).max() < 1.0e-2:
                converged = True

        self.grid.temperature = wp.array3d(temperature, dtype=float)

    def initialize_luminosity_array(self, wavelength):
        nu = (const.c / wavelength).to(u.GHz)
        self.luminosity = np.zeros(self.shape)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    self.luminosity[i,j,k] = (4*np.pi*u.steradian*self.grid.density.numpy()[i,j,k]*self.volume[i,j,k]*self.dust.interpolate_kabs(nu)*self.distance_unit**2*models.BlackBody(temperature=self.grid.temperature.numpy()[i,j,k]*u.K)(nu)).to(u.au**2 * u.Jy).value

        self.total_lum = self.luminosity.sum()

    def propagate_photons(self, photon_list: PhotonList, debug=False):
        nphotons = photon_list.position.numpy().shape[0]
        iphotons = np.arange(nphotons, dtype=np.int32)
        iphotons_original = iphotons.copy()

        photon_list.tau = wp.array(-np.log(1. - np.random.rand(nphotons)), dtype=float)

        s1 = wp.zeros(nphotons, dtype=float)
        s2 = wp.zeros(nphotons, dtype=float)
        photon_list.alpha = wp.zeros(nphotons, dtype=float)
        photon_list.in_grid = wp.zeros(nphotons, dtype=bool)

        next_wall_time = 0.
        dust_interpolation_time = 0.
        tau_distance_time = 0.
        move_time = 0.
        deposit_energy_time = 0.
        photon_loc_time = 0.
        in_grid_time = 0.
        removing_photons_time = 0.
        absorb_time = 0.
        
        t1 = time.time()
        photon_list.kabs = wp.array(self.dust.interpolate_kabs(photon_list.frequency*u.GHz), dtype=float)
        photon_list.ksca = wp.array(self.dust.interpolate_ksca(photon_list.frequency*u.GHz), dtype=float)
        photon_list.albedo = wp.array(photon_list.ksca.numpy() / (photon_list.kabs.numpy() + photon_list.ksca.numpy()), dtype=float)
        t2 = time.time()
        dust_interpolation_time += t2 - t1

        photon_list.absorb = wp.array(np.random.rand(nphotons) > photon_list.albedo.numpy(), dtype=bool)

        count = 0
        while nphotons > 0:
            print(nphotons)
            count += 1

            t1 = time.time()
            wp.launch(kernel=self.next_wall_distance,
                      dim=(nphotons,),
                      inputs=[photon_list, self.grid, s1, iphotons],
                      device='cpu')
            t2 = time.time()
            next_wall_time += t2 - t1
        
            t1 = time.time()
            wp.launch(kernel=self.tau_distance_absorption,
                      dim=(nphotons,),
                      inputs=[self.grid, photon_list, s2, iphotons],
                      device='cpu')
            t2 = time.time()
            tau_distance_time += t2 - t1
        
            s = np.minimum(s1.numpy(), s2.numpy())

            t1 = time.time()
            wp.launch(kernel=self.move,
                      dim=(nphotons,),
                      inputs=[photon_list, s, iphotons],
                      device='cpu')
            t2 = time.time()
            move_time += t2 - t1
        
            t1 = time.time()
            wp.launch(kernel=self.deposit_energy,
                      dim=(nphotons,),
                      inputs=[photon_list, self.grid, s, iphotons],
                      device='cpu')
            t2 = time.time()
            deposit_energy_time += t2 - t1

            wp.launch(kernel=self.reduce_tau,
                       dim=(nphotons,),
                       inputs=[photon_list, s, iphotons],
                       device='cpu')
            
            t1 = time.time()
            wp.launch(kernel=self.photon_loc,
                      dim=(nphotons,),
                      inputs=[photon_list, self.grid, iphotons],
                      device='cpu')
            t2 = time.time()
            photon_loc_time += t2 - t1

            t1 = time.time()
            wp.launch(kernel=self.check_in_grid,
                      dim=(nphotons,),
                      inputs=[photon_list, self.grid, iphotons],
                      device='cpu')
            t2 = time.time()
            in_grid_time += t2 - t1
        
            t1 = time.time()
            iphotons = iphotons_original[photon_list.in_grid]
            nphotons = iphotons.size
            t2 = time.time()
            removing_photons_time += t2 - t1

            t1 = time.time()
            interaction = np.logical_and(photon_list.tau.numpy() <= 1e-10, photon_list.in_grid)
            interaction_indices = iphotons_original[interaction]
            absorb = np.logical_and(interaction, photon_list.absorb.numpy())
            absorb_indices = iphotons_original[absorb]
            tmp_photon_temp_time, tmp_dust_interpolation_time, tmp_photon_loc_time = self.interact(photon_list, absorb, absorb_indices, interaction, interaction_indices)
            t2 = time.time()
            absorb_time += t2 - t1 - tmp_dust_interpolation_time - tmp_photon_loc_time
            #absorb_time += tmp_time
            dust_interpolation_time += tmp_dust_interpolation_time
            photon_loc_time += tmp_photon_loc_time

        print(next_wall_time)
        print(dust_interpolation_time)
        print(tau_distance_time)
        print(move_time)
        print(deposit_energy_time)
        print(photon_loc_time)
        print(in_grid_time)
        print(removing_photons_time)
        print(absorb_time)

    def propagate_photons_scattering(self, photon_list: PhotonList, inu: int, debug=False):
        nphotons = photon_list.position.numpy().shape[0]
        iphotons = np.arange(nphotons, dtype=np.int32)
        iphotons_original = iphotons.copy()

        photon_list.tau = wp.array(-np.log(1. - np.random.rand(nphotons)), dtype=float)

        s1 = wp.zeros(nphotons, dtype=float)
        s2 = wp.zeros(nphotons, dtype=float)
        photon_list.alpha = wp.zeros(nphotons, dtype=float)
        photon_list.in_grid = wp.zeros(nphotons, dtype=bool)

        next_wall_time = 0.
        dust_interpolation_time = 0.
        tau_distance_time = 0.
        move_time = 0.
        deposit_energy_time = 0.
        photon_loc_time = 0.
        in_grid_time = 0.
        removing_photons_time = 0.
        absorb_time = 0.
        
        t1 = time.time()
        photon_list.kabs = wp.array(self.dust.interpolate_kabs(photon_list.frequency*u.GHz), dtype=float)
        photon_list.ksca = wp.array(self.dust.interpolate_ksca(photon_list.frequency*u.GHz), dtype=float)
        photon_list.albedo = wp.array(photon_list.ksca.numpy() / (photon_list.kabs.numpy() + photon_list.ksca.numpy()), dtype=float)
        t2 = time.time()
        dust_interpolation_time += t2 - t1

        photon_list.absorb = wp.array(np.repeat(False, nphotons), dtype=bool)

        photon_list.total_tau_abs = wp.zeros(nphotons, dtype=float)

        count = 0
        while nphotons > 0:
            print(nphotons)
            count += 1

            t1 = time.time()
            wp.launch(kernel=self.next_wall_distance,
                      dim=(nphotons,),
                      inputs=[photon_list, self.grid, s1, iphotons],
                      device='cpu')
            t2 = time.time()
            next_wall_time += t2 - t1
        
            t1 = time.time()
            wp.launch(kernel=self.tau_distance_scattering,
                      dim=(nphotons,),
                      inputs=[self.grid, photon_list, s2, iphotons],
                      device='cpu')
            t2 = time.time()
            tau_distance_time += t2 - t1
        
            s = np.minimum(s1.numpy(), s2.numpy())

            t1 = time.time()
            wp.launch(kernel=self.deposit_scattering,
                      dim=(nphotons,),
                      inputs=[photon_list, s, self.scattering[inu], iphotons],
                      device='cpu')
            t2 = time.time()
            deposit_energy_time += t2 - t1

            t1 = time.time()
            wp.launch(kernel=self.move,
                      dim=(nphotons,),
                      inputs=[photon_list, s, iphotons],
                      device='cpu')
            t2 = time.time()
            move_time += t2 - t1
        
            t1 = time.time()
            wp.launch(kernel=self.photon_loc,
                      dim=(nphotons,),
                      inputs=[photon_list, self.grid, iphotons],
                      device='cpu')
            t2 = time.time()
            photon_loc_time += t2 - t1
        
            t1 = time.time()
            wp.launch(kernel=self.check_in_grid,
                      dim=(nphotons,),
                      inputs=[photon_list, self.grid, iphotons],
                      device='cpu')
            t2 = time.time()
            in_grid_time += t2 - t1
        
            t1 = time.time()
            iphotons = iphotons_original[np.logical_and(photon_list.in_grid, photon_list.total_tau_abs.numpy() < 30.)]
            nphotons = iphotons.size
            t2 = time.time()
            removing_photons_time += t2 - t1

            t1 = time.time()
            interaction = np.logical_and(photon_list.tau.numpy() <= 1e-10, photon_list.in_grid)
            interaction_indices = iphotons_original[interaction]
            absorb = np.logical_and(interaction, photon_list.absorb)
            absorb_indices = iphotons_original[absorb]
            tmp_photon_temp_time, tmp_dust_interpolation_time, tmp_photon_loc_time = self.interact(photon_list, absorb, absorb_indices, interaction, interaction_indices, scattering=True)
            t2 = time.time()
            absorb_time += t2 - t1 - tmp_photon_loc_time
            #absorb_time += tmp_time

        print(next_wall_time)
        print(dust_interpolation_time)
        print(tau_distance_time)
        print(move_time)
        print(deposit_energy_time)
        print(photon_loc_time)
        print(in_grid_time)
        print(removing_photons_time)
        print(absorb_time)

    @wp.kernel
    def check_pixel_too_large(photon_list: PhotonList,
                              pixel_size: float,
                              cell_size: wp.array3d(dtype=float),
                              irays: wp.array(dtype=int)):
        
        iray = wp.tid()
        ir = irays[iray]

        ix, iy, iz = photon_list.indices[ir][0], photon_list.indices[ir][1], photon_list.indices[ir][2]

        photon_list.pixel_too_large[ir] = pixel_size > cell_size[ix,iy,iz]

    @wp.kernel
    def add_intensity(ray_list: PhotonList,
                      s: wp.array(dtype=float),
                      grid: GridStruct,
                      scattering: wp.array4d(dtype=float),
                      irays: wp.array(dtype=int)):

        iray, inu = wp.tid()
        ir = irays[iray]

        ix, iy, iz = ray_list.indices[ir][0], ray_list.indices[ir][1], ray_list.indices[ir][2]

        tau_cell = 0.
        intensity_abs = 0.
        alpha_ext = 0.
        alpha_sca = 0.

        tau_cell = tau_cell + s[ir]*ray_list.kext[inu]*grid.density[ix,iy,iz]
        alpha_ext = alpha_ext + ray_list.kext[inu]*grid.density[ix,iy,iz]
        alpha_sca = alpha_sca + ray_list.kext[inu]*ray_list.albedo[inu]*grid.density[ix,iy,iz]
        intensity_abs = intensity_abs + ray_list.kext[inu] * (1. - ray_list.albedo[inu]) * \
                grid.density[ix,iy,iz] * planck_function(ray_list.frequency[inu] / 1e9, grid.temperature[ix,iy,iz])

        albedo_total = alpha_sca / alpha_ext

        if alpha_ext > 0.:
            intensity_abs = intensity_abs * (1.0 - wp.exp(-tau_cell)) / alpha_ext

        intensity_sca = (1.0 - wp.exp(-tau_cell)) * albedo_total * scattering[inu,ix,iy,iz]

        intensity_cell = intensity_abs

        ray_list.intensity[ir,inu] = ray_list.intensity[ir,inu] + intensity_cell * wp.exp(-ray_list.tau_intensity[ir,inu])
        ray_list.tau_intensity[ir,inu] = ray_list.tau_intensity[ir,inu] + tau_cell

    def propagate_rays(self, ray_list: PhotonList, frequency, pixel_size):
        nrays = ray_list.position.numpy().shape[0]
        iray = np.arange(nrays, dtype=np.int32)
        iray_original = iray.copy()
        nnu = frequency.size
        ray_list.in_grid = wp.zeros(nrays, dtype=bool)

        ray_list.kext = wp.array(self.dust.interpolate_kext(frequency*u.GHz), dtype=float)
        ray_list.albedo = wp.array(self.dust.interpolate_albedo(frequency*u.GHz), dtype=float)

        ray_list.frequency = wp.array(frequency, dtype=float)

        s = wp.zeros(nrays, dtype=float)
        
        while nrays > 0:
            wp.launch(kernel=self.check_pixel_too_large,
                      dim=(nrays,),
                      inputs=[ray_list, pixel_size, (self.volume**(1./3)).astype(np.float32), iray],
                      device='cpu')

            wp.launch(kernel=self.next_wall_distance,
                      dim=(nrays,),
                      inputs=[ray_list, self.grid, s, iray],
                      device='cpu')

            wp.launch(kernel=self.add_intensity,
                      dim=(nrays, nnu),
                      inputs=[ray_list, s, self.grid, self.scattering, iray],
                      device='cpu')
        
            wp.launch(kernel=self.move,
                      dim=(nrays,),
                      inputs=[ray_list, wp.array(s), iray],
                      device='cpu')

            wp.launch(kernel=self.photon_loc,
                      dim=(nrays,),
                      inputs=[ray_list, self.grid, iray],
                      device='cpu')

            wp.launch(kernel=self.check_in_grid,
                      dim=(nrays,),
                      inputs=[ray_list, self.grid, iray],
                      device='cpu')

            iray = iray_original[np.logical_and(ray_list.in_grid, np.logical_not(ray_list.pixel_too_large))]
            nrays = iray.size

    def thermal_mc(self, nphotons, Qthresh=2.0, Delthresh=1.1, p=99.):
        told = self.grid.temperature.numpy().copy()

        count = 0
        while count < 10:
            print("Iteration", count)
            treallyold = told.copy()
            told = self.grid.temperature.numpy().copy()

            photon_list = self.emit(nphotons)

            t1 = time.time()
            self.propagate_photons(photon_list)
            t2 = time.time()
            print("Time:", t2 - t1)

            self.update_grid()

            if count > 1:
                R = np.maximum(told/self.grid.temperature.numpy(), self.grid.temperature.numpy()/told)
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
        self.scattering = np.zeros((len(wavelengths),)+self.shape, dtype=np.float32)
        
        for i, wavelength in enumerate(wavelengths):
            self.initialize_luminosity_array(wavelength=wavelength)

            photon_list = self.emit(nphotons, wavelength, scattering=True)

            t1 = time.time()
            self.propagate_photons_scattering(photon_list, i)
            t2 = time.time()
            print("Time:", t2 - t1)

            self.scattering[i] /= (4.*np.pi * self.volume)

class UniformCartesianGrid(Grid):
    def __init__(self, ncells=9, dx=1.0):
        if type(ncells) == int:
            n1, n2, n3 = ncells, ncells, ncells
        elif type(ncells) == tuple:
            n1, n2, n3 = ncells
            
        if type(dx) == tuple:
            dx, dy, dz = dx
        else:
            dx, dy, dz = dx, dx, dx

        self.distance_unit = dx.unit

        _w1 = np.linspace(-0.5*n1*dx.value, 0.5*n1*dx.value, n1+1)
        _w2 = np.linspace(-0.5*n2*dy.value, 0.5*n2*dy.value, n2+1)
        _w3 = np.linspace(-0.5*n3*dz.value, 0.5*n3*dz.value, n3+1)

        super().__init__(_w1, _w2, _w3)

        self.volume = np.ones((self.grid.n1, self.grid.n2, self.grid.n3)) * (dx.value * dy.value * dz.value)

    def emit(self, nphotons, wavelength="random", scattering=False):
        photon_list = self.base_emit(nphotons, wavelength=wavelength, scattering=scattering)

        photon_list.indices = wp.zeros((nphotons, 3), dtype=int)
        wp.launch(kernel=self.photon_loc,
                  dim=(nphotons,),
                  inputs=[photon_list, self.grid, np.arange(nphotons, dtype=np.int32)],
                  device='cpu')

        return photon_list

    @wp.kernel
    def random_location_in_cell(position: wp.array(dtype=wp.vec3),
                                coords: wp.array2d(dtype=int),
                                grid: GridStruct):
        ip = wp.tid()

        ix, iy, iz = coords[ip][0], coords[ip][1], coords[ip][2]

        rng = wp.rand_init(1234, ip)

        position[ip][0] = grid.w1[ix] + (grid.w1[ix+1] - grid.w1[ix])*wp.randf(rng)
        position[ip][1] = grid.w2[iy] + (grid.w2[iy+1] - grid.w2[iy])*wp.randf(rng)
        position[ip][2] = grid.w3[iz] + (grid.w3[iz+1] - grid.w3[iz])*wp.randf(rng)

    @wp.kernel
    def next_wall_distance(photon_list: PhotonList,
                           grid: GridStruct,
                           distances: wp.array(dtype=float),
                           irays: wp.array(dtype=int)):
    
        ip = irays[wp.tid()]
        #print(ip)

        iw1, iw2, iw3 = photon_list.indices[ip][0], photon_list.indices[ip][1], photon_list.indices[ip][2]

        s = wp.inf
    
        sx1 = (grid.w1[iw1] - photon_list.position[ip][0]) / photon_list.direction[ip][0]
        if sx1 > 0: s = wp.min(s, sx1)
        sx2 = (grid.w1[iw1+1] - photon_list.position[ip][0]) / photon_list.direction[ip][0]
        if sx2 > 0: s = wp.min(s, sx2)

        sy1 = (grid.w2[iw2] - photon_list.position[ip][1]) / photon_list.direction[ip][1]
        if sy1 > 0: s = wp.min(s, sy1)
        sy2 = (grid.w2[iw2+1] - photon_list.position[ip][1]) / photon_list.direction[ip][1]
        if sy2 > 0: s = wp.min(s, sy2)

        sz1 = (grid.w3[iw3] - photon_list.position[ip][2]) / photon_list.direction[ip][2]
        if sz1 > 0: s = wp.min(s, sz1)
        sz2 = (grid.w3[iw3+1] - photon_list.position[ip][2]) / photon_list.direction[ip][2]
        if sz2 > 0: s = wp.min(s, sz2)
    
        distances[ip] = s

    @wp.kernel
    def outer_wall_distance(photon_list: PhotonList,
                           grid: GridStruct,
                           distances: wp.array(dtype=float)):

        ip = wp.tid()

        s = 0.

        if photon_list.direction[ip][0] != 0:
            if photon_list.position[ip][0] <= grid.w1[0]:
                sx = (grid.w1[0] - photon_list.position[ip][0]) / photon_list.direction[ip][0]
                if sx > s:
                    s = sx
            elif photon_list.position[ip][0] >= grid.w1[grid.n1]:
                sx = (grid.w1[grid.n1] - photon_list.position[ip][0]) / photon_list.direction[ip][0]
                if sx > s:
                    s = sx

        if photon_list.direction[ip][1] != 0:
            if photon_list.position[ip][1] <= grid.w2[0]:
                sy = (grid.w2[0] - photon_list.position[ip][1]) / photon_list.direction[ip][1]
                if sy > s:
                    s = sy
            elif photon_list.position[ip][1] >= grid.w2[grid.n2]:
                sy = (grid.w2[grid.n2] - photon_list.position[ip][1]) / photon_list.direction[ip][1]
                if sy > s:
                    s = sy

        if photon_list.direction[ip][2] != 0:
            if photon_list.position[ip][2] <= grid.w3[0]:
                sz = (grid.w3[0] - photon_list.position[ip][2]) / photon_list.direction[ip][2]
                if sz > s:
                    s = sz
            elif photon_list.position[ip][2] >= grid.w3[grid.n3]:
                sz = (grid.w3[grid.n3] - photon_list.position[ip][2]) / photon_list.direction[ip][2]
                if sz > s:
                    s = sz

        new_position = photon_list.position[ip] + s*photon_list.direction[ip]

        if equal(new_position[0],grid.w1[0],EPSILON):
            new_position[0] = grid.w1[0]
        elif equal(new_position[0],grid.w1[grid.n1],EPSILON):
            new_position[0] = grid.w1[grid.n1]

        if equal(new_position[1],grid.w2[0],EPSILON):
            new_position[1] = grid.w2[0]
        elif equal(new_position[1],grid.w2[grid.n2],EPSILON):
            new_position[1] = grid.w2[grid.n2]

        if equal(new_position[2],grid.w3[0],EPSILON):
            new_position[2] = grid.w3[0]
        elif equal(new_position[2],grid.w3[grid.n3],EPSILON):
            new_position[2] = grid.w3[grid.n3]

        if ((new_position[0] < grid.w1[0]) or (new_position[0] > grid.w1[grid.n1]) or (new_position[1] < grid.w2[0]) or
                (new_position[1] > grid.w2[grid.n2]) or (new_position[2] < grid.w3[0]) or (new_position[2] > grid.w3[grid.n3])):
            s = np.inf

        distances[ip] = s

    def grid_size(self):
        return 2*np.sqrt(np.abs(self.grid.w1).max()**2 + np.abs(self.grid.w2).max()**2 + np.abs(self.grid.w3).max()**2)

    @wp.kernel
    def check_in_grid(photon_list: PhotonList,
                grid: GridStruct,
                irays: wp.array(dtype=int)):
    
        ip = irays[wp.tid()]

        if (photon_list.indices[ip,0] >= grid.n1 or photon_list.indices[ip,0] < 0 or \
                photon_list.indices[ip,1] >= grid.n2 or photon_list.indices[ip,1] < 0 or \
                photon_list.indices[ip,2] >= grid.n3 or photon_list.indices[ip,2] < 0):
            photon_list.in_grid[ip] = False
        else:
            photon_list.in_grid[ip] = True
    
    @wp.kernel
    def photon_loc(photon_list: PhotonList,
                   grid: GridStruct,
                   iray: wp.array(dtype=int)):

        ip = iray[wp.tid()]

        if photon_list.position[ip][0] >= grid.w1[grid.n1]:
            i1 = grid.n1-1
        elif photon_list.position[ip][0] <= grid.w1[0]:
            i1 = 0
        else:
            i1 = wp.int(wp.floor((photon_list.position[ip][0] - grid.w1[0]) / (grid.w1[1] - grid.w1[0])))

        if equal(photon_list.position[ip][0], grid.w1[i1], EPSILON):
            photon_list.position[ip][0] = grid.w1[i1]
        elif equal(photon_list.position[ip][0], grid.w1[i1+1], EPSILON):
            photon_list.position[ip][0] = grid.w1[i1+1]

        if photon_list.position[ip][0] == grid.w1[i1] and photon_list.direction[ip][0] < 0:
            i1 -= 1
        elif photon_list.position[ip][0] == grid.w1[i1+1] and photon_list.direction[ip][0] > 0:
            i1 += 1
        photon_list.indices[ip][0] = i1

        if photon_list.position[ip][1] >= grid.w2[grid.n2]:
            i2 = grid.n2-1
        elif photon_list.position[ip][1] <= grid.w2[0]:
            i2 = 0
        else:
            i2 = wp.int(wp.floor((photon_list.position[ip][1] - grid.w2[0]) / (grid.w2[1] - grid.w2[0])))

        if equal(photon_list.position[ip][1], grid.w2[i2], EPSILON):
            photon_list.position[ip][1] = grid.w2[i2]
        elif equal(photon_list.position[ip][1], grid.w2[i2+1], EPSILON):
            photon_list.position[ip][1] = grid.w2[i2+1]

        if photon_list.position[ip][1] == grid.w2[i2] and photon_list.direction[ip][1] < 0:
            i2 -= 1
        elif photon_list.position[ip][1] == grid.w2[i2+1] and photon_list.direction[ip][1] > 0:
            i2 += 1
        photon_list.indices[ip][1] = i2

        if photon_list.position[ip][2] >= grid.w3[grid.n3]:
            i3 = grid.n3-1
        elif photon_list.position[ip][2] <= grid.w3[0]:
            i3 = 0
        else:
            i3 = wp.int(wp.floor((photon_list.position[ip][2] - grid.w3[0]) / (grid.w3[1] - grid.w3[0])))

        if equal(photon_list.position[ip][2], grid.w3[i3], EPSILON):
            photon_list.position[ip][2] = grid.w3[i3]
        elif equal(photon_list.position[ip][2], grid.w3[i3+1], EPSILON):
            photon_list.position[ip][2] = grid.w3[i3+1]

        if photon_list.position[ip][2] == grid.w3[i3] and photon_list.direction[ip][2] < 0:
            i3 -= 1
        elif photon_list.position[ip][2] == grid.w3[i3+1] and photon_list.direction[ip][2] > 0:
            i3 += 1
        photon_list.indices[ip][2] = i3

class UniformSphericalGrid(Grid):
    def __init__(self, ncells=9, dr=1.0, mirror=True):
        if type(ncells) == int:
            n1, n2, n3 = ncells, ncells, ncells
        elif type(ncells) == tuple:
            n1, n2, n3 = ncells

        self.distance_unit = dr.unit

        _w1 = np.linspace(0, n1*dr.value, n1+1)
        if mirror:
            _w2_max = np.pi / 2
        else:
            _w2_max = np.pi
        _w2 = np.linspace(0, _w2_max, n2+1)
        _w3 = np.linspace(0, 2*np.pi, n3+1)

        super().__init__(_w1, _w2, _w3)

        self.grid.sin_w2 = wp.array(np.sin(_w2), dtype=float)
        self.grid.cos_w2 = wp.array(np.cos(_w2), dtype=float)
        self.grid.tan_w2 = wp.array(np.tan(_w2), dtype=float)
        self.grid.neg_mu = wp.array(-self.grid.cos_w2.numpy(), dtype=float)
        self.grid.sin_tol_w2 = wp.array(np.abs(np.sin(_w2.astype(np.float32) * (1.0 - EPSILON)) - np.sin(_w2.astype(np.float32))), dtype=float)
        self.grid.cos_tol_w2 = wp.array(np.abs(np.cos(_w2.astype(np.float32) * (1.0 - EPSILON)) - np.cos(_w2.astype(np.float32))), dtype=float)

        self.grid.sin_w3 = wp.array(np.sin(_w3), dtype=float)
        self.grid.cos_w3 = wp.array(np.cos(_w3), dtype=float)

        if np.abs(self.grid.cos_w2.numpy()[-1]) < EPSILON:
            self.grid.mirror_symmetry = True
            self.volume_scale = 2
        else:
            self.grid.mirror_symmetry = False
            self.volume_scale = 1

        self.volume = (self.grid.w1.numpy().astype(np.float64)[1:]**3 - self.grid.w1.numpy().astype(np.float64)[0:-1]**3)[:,np.newaxis,np.newaxis] * \
                (self.grid.cos_w2.numpy().astype(np.float64)[0:-1] - self.grid.cos_w2.numpy().astype(np.float64)[1:])[np.newaxis,:,np.newaxis] * \
                (self.grid.w3.numpy().astype(np.float64)[1:] - self.grid.w3.numpy().astype(np.float64)[0:-1]) / 3 * self.volume_scale

    def emit(self, nphotons, wavelength="random", scattering=False):
        photon_list = self.base_emit(nphotons, wavelength=wavelength, scattering=scattering)
        
        photon_list.radius = wp.array(np.zeros(nphotons), dtype=float)
        photon_list.theta = wp.zeros(nphotons, dtype=float)
        photon_list.phi = wp.zeros(nphotons, dtype=float)
        photon_list.sin_theta = wp.zeros(nphotons, dtype=float)
        photon_list.cos_theta = wp.zeros(nphotons, dtype=float)
        photon_list.phi = wp.zeros(nphotons, dtype=float)
        photon_list.sin_phi = wp.zeros(nphotons, dtype=float)
        photon_list.cos_phi = wp.zeros(nphotons, dtype=float)

        photon_list.indices = wp.zeros((nphotons, 3), dtype=int)
        wp.launch(kernel=self.photon_loc,
                  dim=(nphotons,),
                  inputs=[photon_list, self.grid, np.arange(nphotons, dtype=np.int32)],
                  device='cpu')

        return photon_list

    @wp.kernel
    def random_location_in_cell(position: wp.array(dtype=wp.vec3),
                                coords: wp.array2d(dtype=int),
                                grid: GridStruct):
        ip = wp.tid()

        ix, iy, iz = coords[ip][0], coords[ip][1], coords[ip][2]

        rng = wp.rand_init(1234, ip)

        r = grid.w1[ix] + wp.randf(rng) * (grid.w1[ix+1] - grid.w1[ix])
        theta = grid.w2[iy] + wp.randf(rng) * (grid.w2[iy+1] - grid.w2[iy])
        phi = grid.w3[iz] + wp.randf(rng) * (grid.w3[iz+1] - grid.w3[iz])

        position[ip][0] = r * wp.sin(theta) * wp.cos(phi)
        position[ip][1] = r * wp.sin(theta) * wp.sin(phi)
        position[ip][2] = r * wp.cos(theta)

    @wp.kernel
    def next_wall_distance(photon_list: PhotonList,
                           grid: GridStruct,
                           distances: wp.array(dtype=float),
                           irays: wp.array(dtype=int)):

        ip = irays[wp.tid()]
        #print(ip)

        iw1, iw2, iw3 = photon_list.indices[ip][0], photon_list.indices[ip][1], photon_list.indices[ip][2]

        s = float(wp.inf)

        #r = photon_list.radius[ip]

        # Calculate the distance to the intersection with the next radial wall.

        b = wp.dot(photon_list.position[ip], photon_list.direction[ip])

        for i in range(iw1, iw1+2):
            if photon_list.radius[ip] == grid.w1[i]:
                sr1 = -b + wp.abs(b)
                #if (sr1 < s) and (sr1 > 0) and not equal_zero(sr1/
                #        (photon_list.radius[ip]*(grid.w2[iw2+1]-grid.w2[iw2])),EPSILON):
                if (sr1 < s) and (sr1 > 0):
                    s = sr1
                sr2 = -b - wp.abs(b)
                #if (sr2 < s) and (sr2 > 0) and not equal_zero(sr2/
                #        (photon_list.radius[ip]*(grid.w2[iw2+1]-grid.w2[iw2])),EPSILON):
                if (sr2 < s) and (sr2 > 0):
                    s = sr2
            else:
                c = photon_list.radius[ip]*photon_list.radius[ip] - grid.w1[i]*grid.w1[i]
                d = b*b - c

                if (d >= 0):
                    sr1 = -b + wp.sqrt(d)
                    if (sr1 < s) and (sr1 > 0):
                        s = sr1
                    sr2 = -b - wp.sqrt(d)
                    if (sr2 < s) and (sr2 > 0):
                        s = sr2

        # Calculate the distance to the intersection with the next theta wall.

        if grid.n2 != 2:
            for i in range(iw2, iw2+2):
                if equal_zero(grid.cos_w2[i], EPSILON):
                    st1 = -photon_list.position[ip][2] / photon_list.direction[ip][2]
                    #if equal_zero(st1 / (photon_list.radius[ip]*(grid.w2[iw2+1]-grid.w2[iw2])), EPSILON):
                    #    st1 = 0.
                    if (st1 < s) and (st1 > 0):
                        s = st1
                else:
                    a = photon_list.direction[ip][0]*photon_list.direction[ip][0]+photon_list.direction[ip][1]*photon_list.direction[ip][1]-photon_list.direction[ip][2]*photon_list.direction[ip][2]*grid.tan_w2[i]*grid.tan_w2[i]
                    b = 2.*(photon_list.position[ip][0]*photon_list.direction[ip][0]+photon_list.position[ip][1]*photon_list.direction[ip][1]-photon_list.position[ip][2]*photon_list.direction[ip][2]*grid.tan_w2[i]*grid.tan_w2[i])

                    if equal(photon_list.sin_theta[ip], grid.sin_w2[i], grid.sin_tol_w2[i]):
                        st1 = (-b + wp.abs(b))/(2.*a)
                        if (st1 < s) and (st1 > 0):
                            s = st1
                        st2 = (-b - wp.abs(b))/(2.*a)
                        if (st2 < s) and (st2 > 0):
                            s = st2
                    else:
                        c = photon_list.position[ip][0]*photon_list.position[ip][0]+photon_list.position[ip][1]*photon_list.position[ip][1]-photon_list.position[ip][2]*photon_list.position[ip][2]*grid.tan_w2[i]*grid.tan_w2[i]
                        d = b*b-4.*a*c

                        if d >= 0:
                            st1 = (-b + wp.sqrt(d))/(2.*a)
                            if (st1 < s) and (st1 > 0):
                                s = st1
                            st2 = (-b - wp.sqrt(d))/(2.*a)
                            if (st2 < s) and (st2 > 0):
                                s = st2

        # Calculate the distance to intersection with the nearest phi wall.

        if grid.n3 != 2:
            for i in range(iw3, iw3+3):
                if photon_list.phi[ip] != grid.w3[i]:
                    c = photon_list.position[ip][0]*grid.sin_w3[i]-photon_list.position[ip][1]*grid.cos_w3[i]
                    d = photon_list.direction[ip][0]*grid.sin_w3[i]-photon_list.direction[ip][1]*grid.cos_w3[i]

                    sp = -c/d

                    if (sp < s) and (sp > 0):
                        s = sp

        distances[ip] = s

    @wp.kernel
    def outer_wall_distance(photon_list: PhotonList,
                           grid: GridStruct,
                           distances: wp.array(dtype=float)):
        """
        Calculate the distance to the outermost radial wall for a photon in spherical coordinates.
    
        Parameters
        ----------
        position : array-like, shape (3,)
            The current position vector of the photon.
        direction : array-like, shape (3,)
            The current direction vector of the photon.
    
        Returns
        -------
        s : float
            The distance to the outer radial wall, or np.inf if no intersection.
        """

        ip = wp.tid()

        r = wp.sqrt(photon_list.position[ip][0]**2. + photon_list.position[ip][1]**2. + photon_list.position[ip][2]**2.)
        s = wp.inf

        b = wp.dot(photon_list.position[ip], photon_list.direction[ip])
        c = r*r - grid.w1[grid.n1-1]*grid.w1[grid.n1-1]
        d = b*b - c
    
        if d >= 0:
            sr1 = -b + wp.sqrt(d)
            if (sr1 < s) and (sr1 > 0):
                s = sr1
            sr2 = -b - wp.sqrt(d)
            if (sr2 < s) and (sr2 > 0):
                s = sr2
    
        distances[ip] = s

    def grid_size(self):
        return 2 * self.grid.w1.numpy()[self.grid.n1]

    @wp.kernel
    def check_in_grid(photon_list: PhotonList,
                      grid: GridStruct,
                      irays: wp.array(dtype=int)):
    
        ip = irays[wp.tid()]

        if (photon_list.indices[ip][0] >= grid.n1) or (photon_list.indices[ip][0] < 0):
            photon_list.in_grid[ip] = False
        else:
            photon_list.in_grid[ip] = True

    @wp.kernel
    def photon_loc(photon_list: PhotonList,
                   grid: GridStruct,
                   iray: wp.array(dtype=int)):
        """
        #Given a photon's position and direction, return its cell indices in the spherical grid.
        #Optionally, prev_indices can be provided for efficient searching.
        #Returns: l (np.array of shape (3,))
        """

        ip = iray[wp.tid()]
        
        EPS = 1e-5

        photon_list.radius[ip] = wp.sqrt(photon_list.position[ip][0]**2. + photon_list.position[ip][1]**2. + photon_list.position[ip][2]**2.)

        # Handle r == 0 case
        if photon_list.radius[ip] == 0:
            photon_list.cos_theta[ip] *= -1.0
            photon_list.theta[ip] = np.pi - photon_list.theta[ip]
            if grid.n3 != 2:
                photon_list.phi[ip] = wp.mod(photon_list.phi[ip] + np.pi, 2.*np.pi)
        else:
            R = wp.sqrt(photon_list.position[ip][0]**2. + photon_list.position[ip][1]**2.)
            
            photon_list.cos_theta[ip] = photon_list.position[ip][2] / photon_list.radius[ip]
            photon_list.sin_theta[ip] = R / photon_list.radius[ip]
            photon_list.theta[ip] = wp.acos(photon_list.cos_theta[ip])
            
            if grid.n3 != 2:
                photon_list.phi[ip] = wp.mod(wp.atan2(photon_list.position[ip][1], photon_list.position[ip][0]) + 2.*np.pi, 2.*np.pi)
            if R == 0:
                photon_list.cos_phi[ip] = 1.0
                photon_list.sin_phi[ip] = 0.0
            else:
                photon_list.cos_phi[ip] = photon_list.position[ip][0] / R
                photon_list.sin_phi[ip] = photon_list.position[ip][1] / R

        if grid.mirror_symmetry:
            if photon_list.cos_theta[ip] < 0:
                photon_list.theta[ip] = np.pi - photon_list.theta[ip]
                photon_list.direction[ip][2] *= -1.
                photon_list.cos_theta[ip] *= -1.

            if equal_zero(photon_list.cos_theta[ip], EPSILON) and photon_list.direction[ip][2] < 0:
                photon_list.direction[ip][2] *= -1.

        # --- Radial index ---
        if photon_list.radius[ip] >= grid.w1[grid.n1-1]:
            i1 = grid.n1-1
        elif photon_list.radius[ip] <= grid.w1[0]:
            i1 = 0
        else:
            i1 = wp.int(wp.floor((photon_list.radius[ip] - grid.w1[0]) / (grid.w1[1] - grid.w1[0])))

        # Snap to wall if needed
        if equal(photon_list.radius[ip], grid.w1[i1], EPS):
            photon_list.radius[ip] = grid.w1[i1]
        elif equal(photon_list.radius[ip], grid.w1[i1+1], EPS):
            photon_list.radius[ip] = grid.w1[i1+1]

        # Direction-based update
        gnx = photon_list.sin_theta[ip] * photon_list.cos_phi[ip]
        gny = photon_list.sin_theta[ip] * photon_list.sin_phi[ip]
        gnz = photon_list.cos_theta[ip]
        if (photon_list.radius[ip] == grid.w1[i1]) and (wp.dot(photon_list.direction[ip], wp.vec3(gnx, gny, gnz)) < 0):
            i1 -= 1
        elif (photon_list.radius[ip] == grid.w1[i1+1]) and (wp.dot(photon_list.direction[ip], wp.vec3(gnx, gny, gnz)) >= 0):
            i1 += 1

        if photon_list.radius[ip] > grid.w1[grid.n1] * (1. + EPSILON):
            i1 = grid.n1

        # --- Theta index ---
        if grid.n2 == 1:
            i2 = 0
        else:
            if -photon_list.cos_theta[ip] >= grid.neg_mu[grid.n2-1]:
                i2 = grid.n2-1
            elif -photon_list.cos_theta[ip] <= grid.neg_mu[0]:
                i2 = 0
            else:
                i2 = wp.int(wp.floor((photon_list.theta[ip] - grid.w2[0]) / (grid.w2[1] - grid.w2[0])))

            # Snap to wall if needed
            #if equal(photon_list.cos_theta[ip], grid.cos_w2[i2], grid.cos_tol_w2[i2]):
            if equal(photon_list.theta[ip], grid.w2[i2], EPS):
                photon_list.theta[ip] = grid.w2[i2]
                photon_list.cos_theta[ip] = grid.cos_w2[i2]
                photon_list.sin_theta[ip] = grid.sin_w2[i2]
            #elif equal(photon_list.cos_theta[ip], grid.cos_w2[i2+1], grid.cos_tol_w2[i2+1]):
            if equal(photon_list.theta[ip], grid.w2[i2+1], EPS):
                photon_list.theta[ip] = grid.w2[i2+1]
                photon_list.cos_theta[ip] = grid.cos_w2[i2+1]
                photon_list.sin_theta[ip] = grid.sin_w2[i2+1]

            # Direction-based update
            gnx = photon_list.cos_theta[ip] * photon_list.cos_phi[ip]
            gny = photon_list.cos_theta[ip] * photon_list.sin_phi[ip]
            gnz = -photon_list.sin_theta[ip]
            if (photon_list.cos_theta[ip] == grid.cos_w2[i2]) and (wp.dot(photon_list.direction[ip], wp.vec3(gnx, gny, gnz)) < 0):
                i2 -= 1
            elif (photon_list.cos_theta[ip] == grid.cos_w2[i2+1]) and (wp.dot(photon_list.direction[ip], wp.vec3(gnx, gny, gnz)) >= 0):
                i2 += 1

            # Clamp
            if i2 == -1: i2 = 0
            if i2 == grid.n2: i2 = grid.n2-1

        # --- Phi index ---
        if grid.n3 == 2:
            i3 = 0
        else:
            i3 = wp.int(wp.floor((photon_list.phi[ip] - grid.w3[0]) / (grid.w3[1] - grid.w3[0])))

            # Snap to wall if needed
            if equal(photon_list.phi[ip], grid.w3[i3], EPS):
                photon_list.phi[ip] = grid.w3[i3]
                photon_list.sin_phi[ip] = grid.sin_w3[i3]
                photon_list.cos_phi[ip] = grid.cos_w3[i3]
            elif equal(photon_list.phi[ip], grid.w3[i3+1], EPS):
                photon_list.phi[ip] = grid.w3[i3+1]
                photon_list.sin_phi[ip] = grid.sin_w3[i3+1]
                photon_list.cos_phi[ip] = grid.cos_w3[i3+1]

            # Direction-based update
            gnx = -photon_list.sin_phi[ip]
            gny = photon_list.cos_phi[ip]
            gnz = 0.0
            if (photon_list.phi[ip] == grid.w3[i3]) and (wp.dot(photon_list.direction[ip], wp.vec3(gnx, gny, gnz)) <= 0):
                i3 -= 1
            elif (photon_list.phi[ip] == grid.w3[i3+1]) and (wp.dot(photon_list.direction[ip], wp.vec3(gnx, gny, gnz)) >= 0):
                i3 += 1
            i3 = (i3 + grid.n3) % grid.n3

            # Special case for phi=0 and negative direction
            if (photon_list.phi[ip] == 0) and (i3 == grid.n3-1):
                photon_list.phi[ip] = grid.w3[i3+1]

        # Return indices

        photon_list.indices[ip][0] = i1
        photon_list.indices[ip][1] = i2
        photon_list.indices[ip][2] = i3

        #Since we may have updated r, theta and phi to be exactly on the grid 
        # cell walls, change the photon position slightly to reflect this. */

        photon_list.position[ip][0] = photon_list.radius[ip] * photon_list.sin_theta[ip] * photon_list.cos_phi[ip]
        photon_list.position[ip][1] = photon_list.radius[ip] * photon_list.sin_theta[ip] * photon_list.sin_phi[ip]
        photon_list.position[ip][2] = photon_list.radius[ip] * photon_list.cos_theta[ip]