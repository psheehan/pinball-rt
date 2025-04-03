from astropy.modeling import models
from typing import Any
import astropy.units as u
import astropy.constants as const
import scipy.stats.qmc
import matplotlib.pyplot as plt
import scipy.integrate
import warp as wp
import numpy as np
import torch
import time

nphotons = 1000000

EPSILON = 1.0e-6

@wp.func
def equal(x: float,
          y: float,
          tol: float):
    if (wp.abs(x-y) < wp.abs(y)*tol):
        return True
    else:
        return False

@wp.func
def planck_function(nu: float,
                    temperature: float):
    h = 6.6260755e-27
    k_B = 1.380658e-16
    c_l = 2.99792458e10

    return 2.0*h*nu*nu*nu/(c_l*c_l)*1.0/(wp.exp(h*nu/(k_B*temperature))-1.0);

class Dust:
    def __init__(self, lam, kabs, ksca):
        f_kabs = scipy.interpolate.interp1d(np.log10(lam), np.log10(kabs), kind="cubic")
        f_ksca = scipy.interpolate.interp1d(np.log10(lam), np.log10(ksca), kind="cubic")

        lam = 10.**np.linspace(np.log10(lam).min(), np.log10(lam).max(), 10000)[::-1]
        kabs = 10.**f_kabs(np.log10(lam))
        ksca = 10.**f_ksca(np.log10(lam))

        self.nu = const.c.cgs.value / lam
        self.lam = lam
        self.kabs = kabs
        self.ksca = ksca
        self.kext = kabs + ksca
        self.albedo = ksca / (kabs + ksca)

        self.temperature = np.logspace(-1.,4.,1000)
        self.log_temperature = np.log10(self.temperature)

        random_nu_PDF = np.array([kabs * models.BlackBody(temperature=T*u.K)(self.nu*u.Hz) for T in self.temperature])
        self.random_nu_CPD = scipy.integrate.cumulative_trapezoid(random_nu_PDF, self.nu, axis=1, initial=0.)
        self.random_nu_CPD /= self.random_nu_CPD[:,-1:]
        self.drandom_nu_CPD_dT = np.gradient(self.random_nu_CPD, self.temperature, axis=0)

    def interpolate_kabs(self, nu):
        return np.interp(nu, self.nu, self.kabs)

    def interpolate_ksca(self, nu):
        return np.interp(nu, self.nu, self.ksca)

    def interpolate_kext(self, nu):
        return np.interp(nu, self.nu, self.kext)

    def interpolate_albedo(self, nu):
        return np.interp(nu, self.nu, self.albedo)

    def absorb(self, temperature):
        nphotons = frequency.numpy().size

        cost = -1. + 2*numpy.random.rand(nphotons)
        sint = np.sqrt(1. - cost**2)
        phi = 2*np.pi*np.random.rand(nphotons)

        direction = np.array([sint*np.cos(phi), sint*np.sin(phi), cost]).T

        frequency = self.random_nu(temperature)

        return direction, frequency

    def random_nu_manual(self, temperature, ksi=None):
        if ksi is None:
            nphotons = temperature.size
            ksi = np.random.rand(nphotons)

        iT = ((np.log10(temperature) - self.log_temperature[0]) / (self.log_temperature[1] - self.log_temperature[0])).astype(int)

        random_nu_CPD = self.random_nu_CPD[iT,:]

        i = np.argmax(ksi[:,np.newaxis] < random_nu_CPD, axis=1)

        frequency = (ksi - random_nu_CPD[np.arange(random_nu_CPD.shape[0]),i-1]) * (self.nu[i] - self.nu[i-1]) / \
                (random_nu_CPD[np.arange(random_nu_CPD.shape[0]),i] - random_nu_CPD[np.arange(random_nu_CPD.shape[0]),i-1]) + \
                self.nu[i-1]

        return frequency

    def learn_random_nu(self, plot=False):
        sampler = scipy.stats.qmc.LatinHypercube(d=2)
        samples = sampler.random(5000)

        T = 10.**(samples[:,0]*5 - 1.)
        ksi = samples[:,1]
        samples = self.random_nu_manual(T, ksi)

        if plot:
            plt.hist(np.log10(samples), 100)
            plt.show()
        
        train_x = torch.tensor(np.vstack((ksi, np.log10(T))).T, dtype=torch.float32)
        train_y = torch.tensor(np.log10(samples), dtype=torch.float32).reshape((ksi.size,1))

        class Sampler(torch.nn.Module):
            def __init__(self):
                super().__init__()
        
                self.hidden1 = torch.nn.Linear(2, 24)
                #self.act1 = torch.nn.ReLU()
                self.act1 = torch.nn.Sigmoid()
                self.hidden2 = torch.nn.Linear(24, 24)
                self.act2 = torch.nn.Sigmoid()
                #self.hidden3 = torch.nn.Linear(48, 48)
                #self.act3 = torch.nn.Sigmoid()
                self.output = torch.nn.Linear(24, 1)
        
            def forward(self, x):
                x = self.act1(self.hidden1(x))
                x = self.act2(self.hidden2(x))
                #x = self.act3(self.hidden3(x))
                x = self.output(x)
        
                return x
        
        model = Sampler()
        
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        n_epochs = 1000
        batch_size = 1000
        
        for epoch in range(n_epochs):
            for i in range(0, len(train_x), batch_size):
                Xbatch = train_x[i:i+batch_size]
                y_pred = model(Xbatch)
                Ybatch = train_y[i:i+batch_size]
        
                loss = loss_fn(y_pred, Ybatch)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            print(f'Finished epoch {epoch}, latest loss {loss}')

        """
        T = 10.**np.random.uniform(-1.,4.,100000)
        ksi = np.random.uniform(0.,1.,100000)
        samples = self.random_nu_manual(T, ksi)

        test_x = torch.tensor(np.vstack((ksi, np.log10(T))).T, dtype=torch.float32)
        test_y = torch.tensor(np.log10(samples), dtype=torch.float32).reshape((ksi.size,1))

        predict_y = model(test_x)

        if plot:
            count, bins, patches = plt.hist(test_y[:,0], 100, histtype='step')
            plt.hist(predict_y[:,0].detach().numpy(), bins, histtype='step')
            plt.show()
        """

        self.model = model

    def random_nu(self, temperature):
        nphotons = temperature.size
        ksi = np.random.rand(int(nphotons))

        test_x = torch.tensor(np.vstack((ksi, np.log10(temperature))).T, dtype=torch.float32)
        
        nu = 10.**self.model(test_x).detach().numpy().flatten()

        return nu

    def planck_mean_opacity(self, temperature):
        vectorized_bb = np.vectorize(lambda T: scipy.integrate.trapezoid(self.kabs * \
                models.BlackBody(temperature=T*u.K)(self.nu*u.Hz).cgs.value, self.nu))

        return np.pi / (const.sigma_T.cgs.value * temperature**4) * vectorized_bb(temperature)
        
class Star:
    x: float
    y: float
    z: float
    temperature: float
    luminosity: float
    radius: float

    def __init__(self, temperature=4000., luminosity=1.0*const.L_sun, radius=1.0*const.R_sun, x=0., y=0., z=0.):
        self.temperature = temperature
        self.luminosity = luminosity.cgs.value
        self.radius = radius.cgs.value
        self.x = x
        self.y = y
        self.z = z

    def set_blackbody_spectrum(self, nu):
        self.nu = nu

        bb = models.BlackBody(temperature=self.temperature*u.K)
        
        self.Bnu = bb(self.nu*u.Hz).cgs.value

        self.random_nu_CPD = scipy.integrate.cumulative_trapezoid(self.Bnu, self.nu, initial=0.)
        self.random_nu_CPD /= self.random_nu_CPD[-1]

    def emit(self, nphotons):
        theta = np.pi*np.random.rand(nphotons)
        phi = 2*np.pi*np.random.rand(nphotons)

        position = np.hstack(((self.radius*np.sin(theta)*np.cos(phi))[:,np.newaxis],
                             (self.radius*np.sin(theta)*np.sin(phi))[:,np.newaxis],
                             (self.radius*np.cos(theta))[:,np.newaxis]))

        r_hat = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).T
        theta_hat = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)]).T
        phi_hat = np.array([-np.sin(phi), np.cos(phi), np.zeros(nphotons)]).T

        cost = np.random.rand(nphotons)
        sint = np.sqrt(1-cost**2)
        phi = 2*np.pi*np.random.rand(nphotons)

        direction = cost[:,np.newaxis]*r_hat + (sint*np.cos(phi))[:,np.newaxis]*phi_hat + (sint*np.sin(phi))[:,np.newaxis]*theta_hat

        frequency = self.random_nu(nphotons)

        photon_energy = np.repeat(self.luminosity / nphotons, nphotons)

        return wp.array(position, dtype=wp.vec3), wp.array(direction, dtype=wp.vec3), \
                frequency, wp.array(photon_energy, dtype=float)

    def random_nu(self, nphotons):
        ksi = np.random.rand(nphotons)

        i = np.array([np.where(k < self.random_nu_CPD)[0].min() for k in ksi])

        return (ksi - self.random_nu_CPD[i-1]) * (self.nu[i] - self.nu[i-1]) / \
                (self.random_nu_CPD[i] - self.random_nu_CPD[i-1]) + self.nu[i-1]

class Grid:
    w1: wp.array(dtype=float)
    w2: wp.array(dtype=float)
    w3: wp.array(dtype=float)
    n1: int
    n2: int
    n3: int

    density: wp.array3d(dtype=float)
    temperature: wp.array3d(dtype=float)
    energy: wp.array3d(dtype=float)

    def __init__(self, _w1, _w2, _w3):
        self.w1 = wp.array(_w1, dtype=float)
        self.w2 = wp.array(_w2, dtype=float)
        self.w3 = wp.array(_w3, dtype=float)
        self.n1 = _w1.size-1
        self.n2 = _w2.size-1
        self.n3 = _w3.size-1

        self.volume = np.ones((self.n1, self.n2, self.n3)) * \
                (_w1[1] - _w1[0]) * (_w2[1] - _w2[0]) * (_w3[1] - _w3[0])

    def add_density(self, _density, dust):
        self.density = wp.array3d(_density, dtype=float)

        self.energy = wp.zeros(_density.shape, dtype=float)
        self.temperature = wp.array(np.ones(_density.shape)*0.1, dtype=float)
        self.mass = self.density.numpy() * self.volume

        self.dust = dust

    def add_star(self, star):
        self.star = star

    def emit(self, nphotons):
        position, direction, frequency, photon_energy = self.star.emit(nphotons)

        indices = wp.zeros((nphotons, 3), dtype=int)
        wp.launch(kernel=self.photon_loc,
                  dim=(nphotons,),
                  inputs=[position, direction, self.w1, self.w2, self.w3, self.n1, self.n2, self.n3, indices, np.arange(nphotons, dtype=np.int32)],
                  device='cpu')

        return position, direction, indices, frequency, photon_energy

    @wp.kernel
    def next_wall_distance(position: wp.array(dtype=wp.vec3),
                           indices: wp.array2d(dtype=int),
                           direction: wp.array(dtype=wp.vec3),
                           w1: wp.array(dtype=float),
                           w2: wp.array(dtype=float),
                           w3: wp.array(dtype=float),
                           distances: wp.array(dtype=float),
                           irays: wp.array(dtype=int)):
    
        ip = irays[wp.tid()]
        #print(ip)
    
        iw1, iw2, iw3 = indices[ip][0], indices[ip][1], indices[ip][2]
    
        s = wp.inf
    
        sx1 = (w1[iw1] - position[ip][0]) / direction[ip][0]
        if sx1 > 0: s = wp.min(s, sx1)
        sx2 = (w1[iw1+1] - position[ip][0]) / direction[ip][0]
        if sx2 > 0: s = wp.min(s, sx2)
    
        sy1 = (w2[iw2] - position[ip][1]) / direction[ip][1]
        if sy1 > 0: s = wp.min(s, sy1)
        sy2 = (w2[iw2+1] - position[ip][1]) / direction[ip][1]
        if sy2 > 0: s = wp.min(s, sy2)
    
        sz1 = (w3[iw3] - position[ip][2]) / direction[ip][2]
        if sz1 > 0: s = wp.min(s, sz1)
        sz2 = (w3[iw3+1] - position[ip][2]) / direction[ip][2]
        if sz2 > 0: s = wp.min(s, sz2)
    
        distances[ip] = s

    @wp.kernel
    def outer_wall_distance(position: wp.array2d(dtype=wp.vec3),
                           direction: wp.array2d(dtype=wp.vec3),
                           w1: wp.array(dtype=float),
                           w2: wp.array(dtype=float),
                           w3: wp.array(dtype=float),
                           n1: int,
                           n2: int,
                           n3: int,
                           distances: wp.array2d(dtype=float)):
    

        ix, iy = wp.tid()

        s = 0.

        if direction[ix,iy][0] != 0:
            if position[ix,iy][0] <= w1[0]:
                sx = (w1[0] - position[ix,iy][0]) / direction[ix,iy][0]
                if sx > s:
                    s = sx
            elif position[ix,iy][0] >= w1[n1]:
                sx = (w1[n1] - position[ix,iy][0]) / direction[ix,iy][0]
                if sx > s:
                    s = sx
    
        if direction[ix,iy][1] != 0:
            if position[ix,iy][1] <= w2[0]:
                sy = (w2[0] - position[ix,iy][1]) / direction[ix,iy][1]
                if sy > s:
                    s = sy
            elif position[ix,iy][1] >= w2[n2]:
                sy = (w2[n2] - position[ix,iy][1]) / direction[ix,iy][1]
                if sy > s:
                    s = sy
    
        if direction[ix,iy][2] != 0:
            if position[ix,iy][2] <= w3[0]:
                sz = (w3[0] - position[ix,iy][2]) / direction[ix,iy][2]
                if sz > s:
                    s = sz
            elif position[ix,iy][2] >= w3[n3]:
                sz = (w3[n3] - position[ix,iy][2]) / direction[ix,iy][2]
                if sz > s:
                    s = sz

        new_position = position[ix,iy] + s*direction[ix,iy]

        if equal(new_position[0],w1[0],EPSILON):
            new_position[0] = w1[0]
        elif equal(new_position[0],w1[n1],EPSILON):
            new_position[0] = w1[n1]

        if equal(new_position[1],w2[0],EPSILON):
            new_position[1] = w2[0]
        elif equal(new_position[1],w2[n2],EPSILON):
            new_position[1] = w2[n2]

        if equal(new_position[2],w3[0],EPSILON):
            new_position[2] = w3[0]
        elif equal(new_position[2],w3[n3],EPSILON):
            new_position[2] = w3[n3]

        if ((new_position[0] < w1[0]) or (new_position[0] > w1[n1]) or (new_position[1] < w2[0]) or
                (new_position[1] > w2[n2]) or (new_position[2] < w3[0]) or (new_position[2] > w3[n3])):
            s = np.inf

        distances[ix,iy] = s

    def grid_size(self):
        return 2*np.sqrt(np.abs(self.w1).max()**2 + np.abs(self.w2).max()**2 + np.abs(self.w3).max()**2)
    
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
    def move(position: wp.array(dtype=wp.vec3),
             direction: wp.array(dtype=wp.vec3),
             distances: wp.array(dtype=float),
             iray: wp.array(dtype=int)):
    
        ip = iray[wp.tid()]
    
        position[ip][0] += distances[ip] * direction[ip][0]
        position[ip][1] += distances[ip] * direction[ip][1]
        position[ip][2] += distances[ip] * direction[ip][2]

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
    def check_in_grid(indices: wp.array2d(dtype=int),
                n1: int,
                n2: int,
                n3: int,
                in_grid: wp.array(dtype=bool),
                irays: wp.array(dtype=int)):
    
        ip = irays[wp.tid()]
    
        if (indices[ip,0] >= n1 or indices[ip,0] < 0 or \
                indices[ip,1] >= n2 or indices[ip,1] < 0 or \
                indices[ip,2] >= n3 or indices[ip,2] < 0):
            in_grid[ip] = False
        else:
            in_grid[ip] = True
    
    @wp.kernel
    def photon_loc(position: wp.array(dtype=wp.vec3),
                   direction: wp.array(dtype=wp.vec3),
                   w1: wp.array(dtype=float),
                   w2: wp.array(dtype=float),
                   w3: wp.array(dtype=float),
                   n1: int,
                   n2: int,
                   n3: int,
                   indices: wp.array2d(dtype=int),
                   iray: wp.array(dtype=int)):

        ip = iray[wp.tid()]
    
        if position[ip][0] >= w1[n1]:
            i1 = n1-1
        elif position[ip][0] <= w1[0]:
            i1 = 0
        else:
            i1 = wp.int(wp.floor((position[ip][0] - w1[0]) / (w1[1] - w1[0])))

        if equal(position[ip][0], w1[i1], EPSILON):
            position[ip][0] = w1[i1]
        elif equal(position[ip][0], w1[i1+1], EPSILON):
            position[ip][0] = w1[i1+1]

        if position[ip][0] == w1[i1] and direction[ip][0] < 0:
            i1 -= 1
        elif position[ip][0] == w1[i1+1] and direction[ip][0] > 0:
            i1 += 1
        indices[ip][0] = i1
    
        if position[ip][1] >= w2[n2]:
            i2 = n2-1
        elif position[ip][1] <= w2[0]:
            i2 = 0
        else:
            i2 = wp.int(wp.floor((position[ip][1] - w2[0]) / (w2[1] - w2[0])))

        if equal(position[ip][1], w2[i2], EPSILON):
            position[ip][1] = w2[i2]
        elif equal(position[ip][1], w2[i2+1], EPSILON):
            position[ip][1] = w2[i2+1]

        if position[ip][1] == w2[i2] and direction[ip][1] < 0:
            i2 -= 1
        elif position[ip][1] == w2[i2+1] and direction[ip][1] > 0:
            i2 += 1
        indices[ip][1] = i2
    
        if position[ip][2] >= w3[n3]:
            i3 = n3-1
        elif position[ip][2] <= w3[0]:
            i3 = 0
        else:
            i3 = wp.int(wp.floor((position[ip][2] - w3[0]) / (w3[1] - w3[0])))

        if equal(position[ip][2], w3[i3], EPSILON):
            position[ip][2] = w3[i3]
        elif equal(position[ip][2], w3[i3+1], EPSILON):
            position[ip][2] = w3[i3+1]

        if position[ip][2] == w3[i3] and direction[ip][2] < 0:
            i3 -= 1
        elif position[ip][2] == w3[i3+1] and direction[ip][2] > 0:
            i3 += 1
        indices[ip][2] = i3

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

    def interact(self, indices, direction, frequency, absorb, iabsorb, iphotons):
        nphotons = iphotons.size

        wp.launch(kernel=self.random_direction,
                  dim=(nphotons,),
                  inputs=[direction, iphotons],
                  device='cpu')

        t1 = time.time()
        nabsorb = iabsorb.size
        photon_temperature = wp.zeros(nabsorb, dtype=float)
        wp.launch(kernel=self.photon_temperature,
                  dim=(nabsorb,),
                  inputs=[indices, self.temperature, photon_temperature, iabsorb],
                  device='cpu')
        t2 = time.time()

        return_val = self.dust.random_nu(photon_temperature.numpy())

        frequency[absorb] = return_val

        return t2-t1

    def update_grid(self):
        total_energy = self.energy.numpy()
        temperature = self.temperature.numpy().copy()

        converged = False
        while not converged:
            old_temperature = temperature.copy()

            temperature = (total_energy / (4*const.sigma_T.cgs.value*\
                    self.dust.planck_mean_opacity(old_temperature)*\
                    self.mass))**0.25

            temperature[temperature < 0.1] = 0.1

            if (np.abs(old_temperature - temperature) / old_temperature).max() < 1.0e-2:
                converged = True

        self.temperature = wp.array3d(temperature, dtype=float)

    def propagate_photons(self, position, direction, indices, frequency, photon_energy, debug=False):
        nphotons = position.numpy().shape[0]
        iphotons = np.arange(nphotons, dtype=np.int32)
        iphotons_original = iphotons.copy()

        tau = -np.log(1. - np.random.rand(nphotons)).astype(np.float32)

        s1 = wp.zeros(nphotons, dtype=float)
        s2 = wp.zeros(nphotons, dtype=float)
        alpha = wp.zeros(nphotons, dtype=float)
        in_grid = wp.zeros(nphotons, dtype=bool)

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
        kabs = self.dust.interpolate_kabs(frequency).astype(np.float32)
        ksca = self.dust.interpolate_ksca(frequency)
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
                      inputs=[position, indices, direction, self.w1, self.w2, self.w3, s1, iphotons],
                      device='cpu')
            t2 = time.time()
            next_wall_time += t2 - t1
        
            t1 = time.time()
            wp.launch(kernel=self.tau_distance,
                      dim=(nphotons,),
                      inputs=[tau, self.density, indices, kabs, s2, alpha, iphotons],
                      device='cpu')
            t2 = time.time()
            tau_distance_time += t2 - t1
        
            s = np.minimum(s1.numpy(), s2.numpy())
        
            t1 = time.time()
            wp.launch(kernel=self.move,
                      dim=(nphotons,),
                      inputs=[position, direction, s, iphotons],
                      device='cpu')
            t2 = time.time()
            move_time += t2 - t1
        
            t1 = time.time()
            wp.launch(kernel=self.deposit_energy,
                      dim=(nphotons,),
                      inputs=[indices, s, kabs, self.density, self.energy, photon_energy, absorb_photon, iphotons],
                      device='cpu')
            t2 = time.time()
            deposit_energy_time += t2 - t1
        
            tau -= s*alpha.numpy()
        
            t1 = time.time()
            wp.launch(kernel=self.photon_loc,
                      dim=(nphotons,),
                      inputs=[position, direction, self.w1, self.w2, self.w3, self.n1, self.n2, self.n3, indices, iphotons],
                      device='cpu')
            t2 = time.time()
            photon_loc_time += t2 - t1
        
            t1 = time.time()
            wp.launch(kernel=self.check_in_grid,
                      dim=(nphotons,),
                      inputs=[indices, self.n1, self.n2, self.n3, in_grid, iphotons],
                      device='cpu')
            t2 = time.time()
            in_grid_time += t2 - t1
        
            t1 = time.time()
            iphotons = iphotons_original[in_grid]
            nphotons = iphotons.size
            t2 = time.time()
            removing_photons_time += t2 - t1

            t1 = time.time()
            interaction = np.logical_and(tau <= 0, in_grid)
            interaction_indices = iphotons_original[interaction]
            absorb = np.logical_and(interaction, absorb_photon)
            absorb_indices = iphotons_original[absorb]
            tmp_time = self.interact(indices, direction, frequency, absorb, absorb_indices, interaction_indices)
            t2 = time.time()
            absorb_time += t2 - t1
            #absorb_time += tmp_time
        
            t1 = time.time()
            kabs[absorb] = self.dust.interpolate_kabs(frequency[absorb])
            ksca[absorb] = self.dust.interpolate_ksca(frequency[absorb])
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

    @wp.kernel
    def add_intensity(s: wp.array(dtype=float),
                      intensity: wp.array2d(dtype=float),
                      tau: wp.array2d(dtype=float),
                      nu: wp.array(dtype=float),
                      kext: wp.array(dtype=float),
                      albedo: wp.array(dtype=float),
                      density: wp.array3d(dtype=float),
                      temperature: wp.array3d(dtype=float),
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

        intensity_cell = intensity_abs

        intensity[ir,inu] += intensity_cell * wp.exp(-tau[ir,inu])
        tau[ir,inu] += tau_cell

    def propagate_rays(self, position, direction, indices, intensity, tau, frequency):
        nrays = position.numpy().shape[0]
        iray = np.arange(nrays, dtype=np.int32)
        iray_original = iray.copy()
        nnu = frequency.size
        in_grid = wp.zeros(nrays, dtype=bool)

        kext = wp.array(self.dust.interpolate_kext(frequency), dtype=float)
        albedo = wp.array(self.dust.interpolate_albedo(frequency), dtype=float)

        frequency = wp.array(frequency, dtype=float)

        s = wp.zeros(nrays, dtype=float)

        original_intensity = intensity
        
        while nrays > 0:
            wp.launch(kernel=self.next_wall_distance,
                      dim=(nrays,),
                      inputs=[position, indices, direction, self.w1, self.w2, self.w3, s, iray],
                      device='cpu')

            wp.launch(kernel=self.add_intensity,
                      dim=(nrays, nnu),
                      inputs=[s, intensity, tau, frequency, kext, albedo, self.density, self.temperature, indices, iray],
                      device='cpu')
        
            wp.launch(kernel=self.move,
                      dim=(nrays,),
                      inputs=[position, direction, wp.array(s), iray],
                      device='cpu')

            wp.launch(kernel=self.photon_loc,
                      dim=(nrays,),
                      inputs=[position, direction, self.w1, self.w2, self.w3, self.n1, self.n2, self.n3, indices, iray],
                      device='cpu')
        
            wp.launch(kernel=self.check_in_grid,
                      dim=(nrays,),
                      inputs=[indices, self.n1, self.n2, self.n3, in_grid, iray],
                      device='cpu')

            iray = iray_original[in_grid]
            nrays = iray.size

    def thermal_mc(self, nphotons, Qthresh=2.0, Delthresh=1.1, p=99.):
        told = self.temperature.numpy().copy()

        count = 0
        while count < 10:
            print("Iteration", count)
            treallyold = told.copy()
            told = self.temperature.numpy().copy()

            position, direction, indices, frequency, photon_energy = self.emit(nphotons)

            t1 = time.time()
            self.propagate_photons(position, direction, indices, frequency, photon_energy)
            t2 = time.time()
            print("Time:", t2 - t1)

            self.update_grid()

            if count > 1:
                R = np.maximum(told/self.temperature.numpy(), self.temperature.numpy()/told)
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

class Image:
    def __init__(self, nx, ny, pixel_size, lam):
        self.nx = nx
        self.ny = ny

        x = (np.arange(nx) - nx / 2)*pixel_size
        y = (np.arange(ny) - ny / 2)*pixel_size

        self.x, self.y = np.meshgrid(x, y)

        self.pixel_size = pixel_size

        self.lam = lam.copy()
        self.nu = const.c.cgs.value / lam

        self.intensity = wp.array3d(np.zeros((nx, ny, lam.size), dtype=float), dtype=float)

class Camera:
    def __init__(self, grid):
        self.grid = grid

    def set_orientation(self, incl, pa, dpc):
        # Set viewing angle parameters.

        #self.r = (dpc*u.pc).cgs.value;
        self.r = self.grid.grid_size()
        self.incl = incl * np.pi/180.
        self.pa = pa * np.pi/180.

        phi = -np.pi/2 - self.pa

        self.i = np.array([self.r*np.sin(self.incl)*np.cos(phi), \
                self.r*np.sin(self.incl)*np.sin(phi), \
                self.r*np.cos(self.incl)])

        self.ex = np.array([-np.sin(phi), np.cos(phi), 0.0])
        self.ey = np.array([-np.cos(self.incl)*np.cos(phi), \
                -np.cos(self.incl)*np.sin(phi), \
                np.sin(self.incl)])
        self.ez = np.array([-np.sin(self.incl)*np.cos(phi), \
                -np.sin(self.incl)*np.sin(phi), \
                -np.cos(self.incl)])

    def emit_rays(self, x, y, _pixel_size, nu):
        intensity = np.zeros(x.shape+(nu.size,), dtype=np.float32)
        tau = np.zeros(x.shape+(nu.size,), dtype=float)
        image_ix, image_iy = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1]))

        pixel_size = np.ones(x.shape) * _pixel_size
        pixel_too_large = np.zeros(x.shape).astype(bool)

        position = np.broadcast_to(self.i, x.shape+(3,)) + x[:,:,np.newaxis]*self.ex + y[:,:,np.newaxis]*self.ey
        direction = np.broadcast_to(self.ez, x.shape+(3,))
        direction = np.where(np.abs(direction) < EPSILON, 0., direction)

        position = wp.array2d(position, dtype=wp.vec3)
        direction = wp.array2d(direction, dtype=wp.vec3)

        return intensity, tau, pixel_size, pixel_too_large, position, direction, image_ix, image_iy

    @wp.kernel
    def put_intensity_in_image(image_ix: wp.array(dtype=int),
                               image_iy: wp.array(dtype=int),
                               ray_intensity: wp.array2d(dtype=float),
                               image_intensity: wp.array3d(dtype=float)):

        ir, inu = wp.tid()

        ix, iy = image_ix[ir], image_iy[ir]

        image_intensity[ix, iy, inu] = ray_intensity[ir,inu]

    def make_image(self, nx, ny, pixel_size, lam, incl, pa, dpc):
        self.set_orientation(incl, pa, dpc)

        image = Image(nx, ny, (pixel_size*u.arcsecond*dpc*u.pc).cgs.value, lam)

        intensity, tau, pixel_size, pixel_too_large, position, direction, image_ix, image_iy = \
                self.emit_rays(image.x, image.y, image.pixel_size, image.nu)

        s = wp.zeros((nx,ny), dtype=float)
    
        wp.launch(kernel=self.grid.outer_wall_distance,
                  dim=(nx,ny),
                  inputs=[position, direction, self.grid.w1, self.grid.w2, self.grid.w3,
                  self.grid.n1, self.grid.n2, self.grid.n3, s],
                  device='cpu')

        s = s.numpy()
        will_be_in_grid = s < np.inf

        position = position.numpy()[will_be_in_grid]
        direction = direction.numpy()[will_be_in_grid]
        s = s[will_be_in_grid]

        position = position + s[:,np.newaxis]*direction

        position = wp.array(position, dtype=wp.vec3)
        direction = wp.array(direction, dtype=wp.vec3)
        intensity = intensity[will_be_in_grid]
        tau = wp.array(tau[will_be_in_grid], dtype=float)
        image_ix = wp.array(image_ix[will_be_in_grid], dtype=int)
        image_iy = wp.array(image_iy[will_be_in_grid], dtype=int)

        nrays = will_be_in_grid.sum()
        iray = np.arange(nrays, dtype=np.int32)

        indices = wp.zeros((nrays, 3), dtype=int)
        wp.launch(kernel=self.grid.photon_loc,
                  dim=(nrays,),
                  inputs=[position, direction, self.grid.w1, self.grid.w2, self.grid.w3, self.grid.n1, self.grid.n2, self.grid.n3, indices, iray],
                  device='cpu')

        self.grid.propagate_rays(position, direction, indices, intensity, tau, image.nu)

        wp.launch(kernel=self.put_intensity_in_image, 
                  dim=(nrays, image.lam.size),
                  inputs=[image_ix, image_iy, intensity, image.intensity])

        return image

n1, n2, n3 = 9, 9, 9
w1 = (np.linspace(-4.5, 4.5, n1+1)*u.au).cgs.value
w2 = (np.linspace(-4.5, 4.5, n2+1)*u.au).cgs.value
w3 = (np.linspace(-4.5, 4.5, n3+1)*u.au).cgs.value

density = np.ones((n1,n2,n3))*1.0e-16

# Set up the dust.

data = np.loadtxt('dustkappa_yso.inp', skiprows=2)

lam = data[::-1,0].copy() * 1.0e-4
kabs = data[::-1,1].copy()
ksca = data[::-1,2].copy()

d = Dust(lam, kabs, ksca)

d.learn_random_nu()

# Set up the star.

star = Star()
star.set_blackbody_spectrum(d.nu)

# Set up the grid.

grid = Grid(w1, w2, w3)
grid.add_density(density, d)
grid.add_star(star)

grid.thermal_mc(nphotons)

for i in range(9):
    plt.imshow(grid.temperature.numpy()[:,:,i])
    plt.savefig(f"temperature_{i}.png")

camera = Camera(grid)
image = camera.make_image(256, 256, 0.1, np.array([1000.]), 45., 45., 1.)

plt.imshow(image.intensity)
plt.savefig("image.png")
