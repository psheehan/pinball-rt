from .photons import PhotonList
from astropy.modeling import models
import astropy.units as u
import astropy.constants as const
import scipy.integrate
import warp as wp
import numpy as np
        
class Star:
    x: float
    y: float
    z: float
    temperature: float
    luminosity: float
    radius: float

    def __init__(self, temperature=4000.*u.K, luminosity=1.0*const.L_sun, radius=1.0*const.R_sun, x=0., y=0., z=0.):
        self.temperature = temperature
        self.luminosity = luminosity
        self.radius = radius
        self.x = x
        self.y = y
        self.z = z

    def set_blackbody_spectrum(self, nu):
        self.nu = nu

        bb = models.BlackBody(temperature=self.temperature)
        
        self.Bnu = bb(self.nu)

        self.random_nu_CPD = scipy.integrate.cumulative_trapezoid(self.Bnu, self.nu, initial=0.)
        self.random_nu_CPD /= self.random_nu_CPD[-1]

    def emit(self, nphotons, distance_unit, wavelength="random", simulation="thermal"):
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
            frequency = self.random_nu(nphotons)
        else:
            frequency = np.repeat((const.c / wavelength).to(u.GHz), nphotons)

        if simulation == "thermal":
            photon_energy = np.repeat(self.luminosity.to(u.L_sun).value / nphotons, nphotons)
        elif simulation == "scattering":
            photon_energy = np.repeat((4.*np.pi**2*u.steradian*self.radius**2*models.BlackBody(temperature=self.temperature)(frequency[0])).to(distance_unit**2 * u.Jy).value / nphotons, nphotons)

        photon_list = PhotonList()
        photon_list.position = wp.array(position, dtype=wp.vec3)
        photon_list.direction = wp.array(direction, dtype=wp.vec3)
        photon_list.frequency = wp.array(frequency.value, dtype=float)
        photon_list.energy = wp.array(photon_energy, dtype=float)

        return photon_list

    def random_nu(self, nphotons):
        ksi = np.random.rand(nphotons)

        i = np.array([np.where(k < self.random_nu_CPD)[0].min() for k in ksi])

        return (ksi - self.random_nu_CPD[i-1]) * (self.nu[i] - self.nu[i-1]) / \
                (self.random_nu_CPD[i] - self.random_nu_CPD[i-1]) + self.nu[i-1]