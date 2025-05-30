from .photons import PhotonList
from astropy.modeling import models
import astropy.units as u
import astropy.constants as const
import warp as wp
import numpy as np
import time

from .utils import EPSILON, equal, planck_function

@wp.struct
class GridStruct:
    w1: wp.array(dtype=float)
    w2: wp.array(dtype=float)
    w3: wp.array(dtype=float)
    n1: int
    n2: int
    n3: int

class Grid:
    density: wp.array3d(dtype=float)
    temperature: wp.array3d(dtype=float)
    energy: wp.array3d(dtype=float)

    def __init__(self, _w1, _w2, _w3):
        self.grid = GridStruct()
        self.grid.w1 = wp.array(_w1, dtype=float)
        self.grid.w2 = wp.array(_w2, dtype=float)
        self.grid.w3 = wp.array(_w3, dtype=float)
        self.grid.n1 = _w1.size-1
        self.grid.n2 = _w2.size-1
        self.grid.n3 = _w3.size-1

    def add_density(self, _density, dust):
        self.density = _density.astype(np.float32)

        self.energy = wp.zeros(_density.shape, dtype=float)
        self.temperature = np.ones(_density.shape, dtype=np.float32)*0.1
        self.mass = self.density * self.volume

        self.dust = dust

    def add_star(self, star):
        self.star = star

    def emit(self, nphotons, wavelength="random", scattering=False):
        if scattering:
            nphotons_per_source = int(nphotons / 2)
        else:
            nphotons_per_source = nphotons

        photon_list = self.star.emit(nphotons_per_source, wavelength)

        cell_coords = []
        if scattering:
            ksi = np.random.rand(nphotons_per_source)
            cum_lum = np.cumsum(self.luminosity.flatten()).reshape(self.density.shape) / self.total_lum

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

            photon_list.frequency = wp.array(np.concatenate([photon_list.frequency, np.repeat((const.c / (wavelength*u.micron)).to(u.Hz).value, nphotons_per_source)]), dtype=float)
            photon_list.energy = wp.array(np.concatenate([photon_list.energy, np.repeat(self.total_lum/nphotons_per_source, nphotons_per_source).astype(np.float32)]), dtype=float)

        photon_list.indices = wp.zeros((nphotons, 3), dtype=int)
        wp.launch(kernel=self.photon_loc,
                  dim=(nphotons,),
                  inputs=[photon_list, self.grid, np.arange(nphotons, dtype=np.int32)],
                  device='cpu')

        return photon_list
    
    @wp.kernel
    def tau_distance(tau: wp.array(dtype=float),
                     density: wp.array3d(dtype=float),
                     indices: wp.array2d(dtype=int),
                     kabs: wp.array(dtype=float),
                     distances: wp.array(dtype=float),
                     alpha:wp.array(dtype=float),
                     iphotons:wp.array(dtype=int)):
    
        ip = iphotons[wp.tid()]
    
        ix, iy, iz = indices[ip][0], indices[ip][1], indices[ip][2]
    
        alpha[ip] = density[ix,iy,iz] * kabs[ip]
        distances[ip] = tau[ip] / alpha[ip]
    
    @wp.kernel
    def move(photon_list: PhotonList,
             distances: wp.array(dtype=float),
             iray: wp.array(dtype=int)):
    
        ip = iray[wp.tid()]

        photon_list.position[ip][0] += distances[ip] * photon_list.direction[ip][0]
        photon_list.position[ip][1] += distances[ip] * photon_list.direction[ip][1]
        photon_list.position[ip][2] += distances[ip] * photon_list.direction[ip][2]

    @wp.kernel
    def deposit_energy(indices: wp.array2d(dtype=int),
                       distances: wp.array(dtype=float),
                       kabs: wp.array(dtype=float),
                       density: wp.array3d(dtype=float),
                       energy: wp.array3d(dtype=float),
                       photon_energy: wp.array(dtype=float),
                       absorb_photon: wp.array(dtype=bool),
                       iphotons: wp.array(dtype=int)):

        ip = iphotons[wp.tid()]

        ix, iy, iz = indices[ip][0], indices[ip][1], indices[ip][2]

        if absorb_photon[ip]:
            energy[ix,iy,iz] += 10.**(wp.log10(photon_energy[ip]) + wp.log10(distances[ip]) + wp.log10(kabs[ip]) + wp.log10(density[ix,iy,iz]))

    @wp.kernel
    def deposit_scattering(indices: wp.array2d(dtype=int),
                       distances: wp.array(dtype=float),
                       scattering: wp.array3d(dtype=float),
                       average_energy: wp.array(dtype=float),
                       iphotons: wp.array(dtype=int)):

        ip = iphotons[wp.tid()]

        ix, iy, iz = indices[ip][0], indices[ip][1], indices[ip][2]

        scattering[ix,iy,iz] += average_energy[ip] * distances[ip]

    @wp.kernel
    def photon_temperature(indices: wp.array2d(dtype=int),
                           temperature: wp.array3d(dtype=float),
                           photon_temperature: wp.array(dtype=float),
                           iphotons: wp.array(dtype=int)):
        itemp = wp.tid()
        ip = iphotons[itemp]

        ix, iy, iz = indices[ip][0], indices[ip][1], indices[ip][2]

        photon_temperature[itemp] = temperature[ix, iy, iz]

    @wp.kernel
    def update_frequency(photon_list: PhotonList,
                            frequency: wp.array(dtype=float),
                            iphotons: wp.array(dtype=int)):
        i = wp.tid()
        ip = iphotons[i]
        photon_list.frequency[ip] = frequency[i]

    @wp.kernel
    def random_direction(direction: wp.array(dtype=wp.vec3),
                         iphotons: wp.array(dtype=int)):
        i = wp.tid()
        ip = iphotons[i]

        rng = wp.rand_init(1234, i)

        cost = -1. + 2.*wp.randf(rng)
        sint = wp.sqrt(1.-cost**2.)
        phi = 2.*np.pi*wp.randf(rng)

        direction[ip][0] = sint*np.cos(phi)
        direction[ip][1] = sint*np.sin(phi)
        direction[ip][2] = cost

    def interact(self, photon_list: PhotonList, absorb, iabsorb, iphotons):
        nphotons = iphotons.size

        wp.launch(kernel=self.random_direction,
                  dim=(nphotons,),
                  inputs=[photon_list.direction, iphotons],
                  device='cpu')

        t1 = time.time()
        nabsorb = iabsorb.size
        photon_temperature = wp.zeros(nabsorb, dtype=float)
        wp.launch(kernel=self.photon_temperature,
                  dim=(nabsorb,),
                  inputs=[photon_list.indices, self.temperature, photon_temperature, iabsorb],
                  device='cpu')
        t2 = time.time()

        return_val = self.dust.random_nu(photon_temperature.numpy())

        wp.launch(kernel=self.update_frequency,
                  dim=(nabsorb,),
                  inputs=[photon_list, return_val, iabsorb],
                  device='cpu')

        return t2-t1

    def update_grid(self):
        total_energy = self.energy.numpy()
        temperature = self.temperature.copy()

        converged = False
        while not converged:
            old_temperature = temperature.copy()

            temperature = (total_energy / (4*const.sigma_T.cgs.value*\
                    self.dust.planck_mean_opacity(old_temperature)*\
                    self.mass))**0.25

            temperature[temperature < 0.1] = 0.1

            if (np.abs(old_temperature - temperature) / old_temperature).max() < 1.0e-2:
                converged = True

        self.temperature = temperature.astype(np.float32)

    def initialize_luminosity_array(self, wavelength):
        nu = const.c.cgs.value / wavelength
        self.luminosity = np.zeros(self.density.shape)

        for i in range(self.temperature.shape[0]):
            for j in range(self.temperature.shape[1]):
                for k in range(self.temperature.shape[2]):
                    self.luminosity[i,j,k] = 4*np.pi*self.density[i,j,k]*self.volume[i,j,k]*self.dust.interpolate_kabs(nu)*models.BlackBody(temperature=self.temperature[i,j,k]*u.K)(nu*u.Hz).cgs.value

        self.total_lum = self.luminosity.sum()

    def propagate_photons(self, photon_list: PhotonList, debug=False):
        nphotons = photon_list.position.numpy().shape[0]
        iphotons = np.arange(nphotons, dtype=np.int32)
        iphotons_original = iphotons.copy()

        tau = -np.log(1. - np.random.rand(nphotons)).astype(np.float32)

        s1 = wp.zeros(nphotons, dtype=float)
        s2 = wp.zeros(nphotons, dtype=float)
        alpha = wp.zeros(nphotons, dtype=float)
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
        kabs = self.dust.interpolate_kabs(photon_list.frequency).astype(np.float32)
        ksca = self.dust.interpolate_ksca(photon_list.frequency).astype(np.float32)
        albedo = ksca / (kabs + ksca)
        t2 = time.time()
        dust_interpolation_time += t2 - t1

        absorb_photon = np.random.rand(nphotons) > albedo

        count = 0
        while nphotons > 0:
            count += 1

            t1 = time.time()
            wp.launch(kernel=self.next_wall_distance,
                      dim=(nphotons,),
                      inputs=[photon_list, self.grid, s1, iphotons],
                      device='cpu')
            t2 = time.time()
            next_wall_time += t2 - t1
        
            t1 = time.time()
            wp.launch(kernel=self.tau_distance,
                      dim=(nphotons,),
                      inputs=[tau, self.density, photon_list.indices, kabs, s2, alpha, iphotons],
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
                      inputs=[photon_list.indices, s, kabs, self.density, self.energy, photon_list.energy, absorb_photon, iphotons],
                      device='cpu')
            t2 = time.time()
            deposit_energy_time += t2 - t1
        
            tau -= s*alpha.numpy()
        
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
            interaction = np.logical_and(tau <= 0, photon_list.in_grid)
            interaction_indices = iphotons_original[interaction]
            absorb = np.logical_and(interaction, absorb_photon)
            absorb_indices = iphotons_original[absorb]
            tmp_time = self.interact(photon_list, absorb, absorb_indices, interaction_indices)
            t2 = time.time()
            absorb_time += t2 - t1
            #absorb_time += tmp_time
        
            t1 = time.time()
            kabs[absorb] = self.dust.interpolate_kabs(photon_list.frequency.numpy()[absorb])
            ksca[absorb] = self.dust.interpolate_ksca(photon_list.frequency.numpy()[absorb])
            albedo = ksca / (kabs + ksca)
            t2 = time.time()
            dust_interpolation_time += t2 - t1
        
            tau[interaction] = -np.log(1. - np.random.rand(interaction.sum()))
            absorb_photon[interaction] = np.random.rand(interaction.sum()) > albedo[interaction]

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

        tau = -np.log(1. - np.random.rand(nphotons)).astype(np.float32)

        s1 = wp.zeros(nphotons, dtype=float)
        s2 = wp.zeros(nphotons, dtype=float)
        alpha_scat = wp.zeros(nphotons, dtype=float)
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
        kabs = self.dust.interpolate_kabs(photon_list.frequency).astype(np.float32)
        ksca = self.dust.interpolate_ksca(photon_list.frequency).astype(np.float32)
        albedo = ksca / (kabs + ksca)
        t2 = time.time()
        dust_interpolation_time += t2 - t1

        absorb_photon = np.repeat(False, nphotons)

        total_tau_abs = np.zeros(nphotons)

        count = 0
        while nphotons > 0:
            count += 1

            t1 = time.time()
            wp.launch(kernel=self.next_wall_distance,
                      dim=(nphotons,),
                      inputs=[photon_list, self.grid, s1, iphotons],
                      device='cpu')
            t2 = time.time()
            next_wall_time += t2 - t1
        
            t1 = time.time()
            wp.launch(kernel=self.tau_distance,
                      dim=(nphotons,),
                      inputs=[tau, self.density, photon_list.indices, ksca, s2, alpha_scat, iphotons],
                      device='cpu')
            t2 = time.time()
            tau_distance_time += t2 - t1
        
            s = np.minimum(s1.numpy(), s2.numpy())

            tau_abs = s * alpha_scat * kabs / ksca
            tau_scat = s * alpha_scat

            average_energy = np.where(tau_abs < EPSILON, (1. - 0.5*tau_abs) * photon_list.energy, (1.0 - np.exp(-tau_abs)) / tau_abs * photon_list.energy)

            t1 = time.time()
            wp.launch(kernel=self.deposit_scattering,
                      dim=(nphotons,),
                      inputs=[photon_list.indices, s, self.scattering[inu], average_energy, iphotons],
                      device='cpu')
            t2 = time.time()
            deposit_energy_time += t2 - t1

            photon_list.energy = wp.array(photon_list.energy * np.exp(-tau_abs), dtype=float)
            tau -= tau_scat
            total_tau_abs += tau_abs
        
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
            iphotons = iphotons_original[np.logical_and(photon_list.in_grid, total_tau_abs < 30.)]
            nphotons = iphotons.size
            t2 = time.time()
            removing_photons_time += t2 - t1

            t1 = time.time()
            interaction = np.logical_and(tau <= 0, photon_list.in_grid)
            interaction_indices = iphotons_original[interaction]
            absorb = np.logical_and(interaction, absorb_photon)
            absorb_indices = iphotons_original[absorb]
            tmp_time = self.interact(photon_list, absorb, absorb_indices, interaction_indices)
            t2 = time.time()
            absorb_time += t2 - t1
            #absorb_time += tmp_time
        
            tau[interaction] = -np.log(1. - np.random.rand(interaction.sum()))

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
    def check_pixel_too_large(indices: wp.array2d(dtype=int),
                              pixel_size: float,
                              pixel_too_large: wp.array(dtype=bool),
                              cell_size: wp.array3d(dtype=float),
                              irays: wp.array(dtype=int)):
        
        iray = wp.tid()
        ir = irays[iray]

        ix, iy, iz = indices[ir][0], indices[ir][1], indices[ir][2]

        pixel_too_large[ir] = pixel_size > cell_size[ix,iy,iz]

    @wp.kernel
    def add_intensity(s: wp.array(dtype=float),
                      intensity: wp.array2d(dtype=float),
                      tau: wp.array2d(dtype=float),
                      nu: wp.array(dtype=float),
                      kext: wp.array(dtype=float),
                      albedo: wp.array(dtype=float),
                      density: wp.array3d(dtype=float),
                      temperature: wp.array3d(dtype=float),
                      scattering: wp.array3d(dtype=float),
                      indices: wp.array2d(dtype=int),
                      irays: wp.array(dtype=int)):

        iray, inu = wp.tid()
        ir = irays[iray]

        ix, iy, iz = indices[ir][0], indices[ir][1], indices[ir][2]

        tau_cell = 0.
        intensity_abs = 0.
        alpha_ext = 0.
        alpha_sca = 0.

        tau_cell += s[ir]*kext[inu]*density[ix,iy,iz]
        alpha_ext += kext[inu]*density[ix,iy,iz]
        alpha_sca += kext[inu]*albedo[inu]*density[ix,iy,iz]
        intensity_abs += kext[inu] * (1. - albedo[inu]) * \
                density[ix,iy,iz] * planck_function(nu[inu], temperature[ix,iy,iz])

        albedo_total = alpha_sca / alpha_ext

        if alpha_ext > 0.:
            intensity_abs *= (1.0 - wp.exp(-tau_cell)) / alpha_ext

        intensity_sca = (1.0 - wp.exp(-tau_cell)) * albedo_total * scattering[ix,iy,iz]

        intensity_cell = intensity_abs

        intensity[ir,inu] += intensity_cell * wp.exp(-tau[ir,inu])
        tau[ir,inu] += tau_cell

    def propagate_rays(self, ray_list: PhotonList, frequency, pixel_size):
        nrays = ray_list.position.numpy().shape[0]
        iray = np.arange(nrays, dtype=np.int32)
        iray_original = iray.copy()
        nnu = frequency.size
        ray_list.in_grid = wp.zeros(nrays, dtype=bool)

        kext = wp.array(self.dust.interpolate_kext(frequency), dtype=float)
        albedo = wp.array(self.dust.interpolate_albedo(frequency), dtype=float)

        frequency = wp.array(frequency, dtype=float)

        s = wp.zeros(nrays, dtype=float)
        
        while nrays > 0:
            wp.launch(kernel=self.check_pixel_too_large,
                      dim=(nrays,),
                      inputs=[ray_list.indices, pixel_size, ray_list.pixel_too_large, (self.volume**(1./3)).astype(np.float32), iray],
                      device='cpu')

            wp.launch(kernel=self.next_wall_distance,
                      dim=(nrays,),
                      inputs=[ray_list, self.grid, s, iray],
                      device='cpu')

            wp.launch(kernel=self.add_intensity,
                      dim=(nrays, nnu),
                      inputs=[s, ray_list.intensity, ray_list.tau, frequency, kext, albedo, self.density, self.temperature, self.scattering, ray_list.indices, iray],
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
        told = self.temperature.copy()

        count = 0
        while count < 10:
            print("Iteration", count)
            treallyold = told.copy()
            told = self.temperature.copy()

            photon_list = self.emit(nphotons)

            t1 = time.time()
            self.propagate_photons(photon_list)
            t2 = time.time()
            print("Time:", t2 - t1)

            self.update_grid()

            if count > 1:
                R = np.maximum(told/self.temperature, self.temperature/told)
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
        self.scattering = np.zeros((len(wavelengths),)+self.density.shape, dtype=np.float32)
        
        for i, wavelength in enumerate(wavelengths):
            self.initialize_luminosity_array(wavelength=wavelength)

            photon_list = self.emit(nphotons, wavelength, scattering=True)

            t1 = time.time()
            self.propagate_photons_scattering(photon_list, i)
            t2 = time.time()
            print("Time:", t2 - t1)

            self.scattering[i] /= (4.*np.pi * self.volume)

class CartesianGrid(Grid):
    def __init__(self, _w1, _w2, _w3):
        super().__init__(_w1, _w2, _w3)

        self.volume = (np.ones((self.grid.n1, self.grid.n2, self.grid.n3)) * \
                (_w1[1] - _w1[0]) * (_w2[1] - _w2[0]) * (_w3[1] - _w3[0]))

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