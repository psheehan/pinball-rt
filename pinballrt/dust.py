import urllib
import requests
from .sources import BlackbodyStar
from .grids import UniformSphericalGrid
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.spatial.transform import Rotation
import pandas as pd
import scipy.interpolate
from astropy.modeling import models
import astropy.units as u
import astropy.constants as const
import scipy.stats.qmc
import scipy.integrate
import warp as wp
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch
import os

from torch.distributions.multivariate_normal import MultivariateNormal

wp.config.quiet = True

class Dust(pl.LightningDataModule):
    def __init__(self, lam=None, kabs=None, ksca=None, amax=None, p=None, device="cpu"):
        """
        Initialize the Dust module with wavelength, absorption, and scattering coefficients.

        Parameters
        ----------
        lam : astropy.units.Quantity
            Wavelengths at which the dust opacities are defined.
        kabs : astropy.units.Quantity
            Absorption coefficients of the dust.
        ksca : astropy.units.Quantity
            Scattering coefficients of the dust.
        amax : astropy.units.Quantity
            Maximum grain size for the size distribution.
        p : float
            Power-law index for the grain size distribution.
        device : str
            Device to run the computations on (e.g., "cpu" or "cuda").
        """
        super().__init__()

        kunit = kabs.unit
        lam_unit = lam.unit
        amax_unit = amax.unit

        lam = lam.value
        amax = amax.value
        p = p
        kabs = kabs.value
        ksca = ksca.value

        if lam[1] > lam[0]:
            lam = lam[::-1]
            kabs = np.flip(kabs, axis=-1)
            ksca = np.flip(ksca, axis=-1)

        self.nu = (const.c / (lam * lam_unit)).decompose().to(u.GHz)
        self.kmean = np.mean(kabs) * kunit
        self.lam = lam * lam_unit
        self.amax = amax * amax_unit
        self.p = p
        self.kabs = kabs / self.kmean.value
        self.ksca = ksca / self.kmean.value
        self.kext = (kabs + ksca) / self.kmean.value
        self.albedo = ksca / (kabs + ksca)

        self.log10_nu_min = np.log10(self.nu.value.min())
        self.log10_nu_max = np.log10(self.nu.value.max())

        with wp.ScopedDevice(device):
            self.nu_wp = wp.array(self.nu.value, dtype=float)
            self.amax_wp = wp.array(self.amax.value, dtype=float)
            self.p_wp = wp.array(self.p, dtype=float)
            self.kabs_wp = wp.array3d(self.kabs, dtype=float)
            self.ksca_wp = wp.array3d(self.ksca, dtype=float)
            self.kext_wp = wp.array3d(self.kext, dtype=float)
            self.albedo_wp = wp.array3d(self.albedo, dtype=float)

        self.temperature = np.logspace(-1.,4.,999)
        self.log_temperature = np.log10(self.temperature)

    def __getstate__(self):
        state = self.__dict__.copy()
                
        for entry in state:
            if isinstance(getattr(self, entry), wp.types.array):
                state[entry] = getattr(self, entry).numpy()
            else:
                state[entry] = getattr(self, entry)

        return state
    
    def __setstate__(self, state):
        for entry in state:
            if 'wp' in entry:
                state[entry] = wp.array(state[entry])

        self.__dict__.update(state)

    def to_device(self, device):
        with wp.ScopedDevice(device):
            self.nu_wp = wp.array(self.nu.value, dtype=float)
            self.kabs_wp = wp.array(self.kabs, dtype=float)
            self.ksca_wp = wp.array(self.ksca, dtype=float)
            self.kext_wp = wp.array(self.kext, dtype=float)
            self.albedo_wp = wp.array(self.albedo, dtype=float)

        for model in ["random_nu", "ml_step", "kabs", "ksca"]:
            if hasattr(self, f"{model}_model"):
                getattr(self, f"{model}_model").to(device)
            if hasattr(self, f"{model}_x_scaler"):
                getattr(self, f"{model}_x_scaler").to(device)
            if hasattr(self, f"{model}_y_scaler"):
                getattr(self, f"{model}_y_scaler").to(device)

    def interpolate_kabs(self, p, amax, nu):
        samples = np.vstack((p.flatten(), np.log10(amax).flatten(), np.log10(nu).flatten())).T

        interpolated = scipy.interpolate.interpn((self.p, np.log10(self.amax.value), np.log10(self.nu.value)), np.log10(self.kabs), samples, method="cubic")

        return 10.**interpolated.reshape(p.shape)

    def ml_kabs(self, p=None, amax=None, nu=None, photon_list=None, iphotons=None):
        if photon_list is not None:
            p = wp.to_torch(photon_list.p)
            amax = wp.to_torch(photon_list.amax)
            if nu is None:
                nu = wp.to_torch(photon_list.frequency)

                if iphotons is not None:
                    nu = nu[iphotons]
                    p = p[iphotons]
                    amax = amax[iphotons]
            else:
                if nu.size(0) != p.size(0):
                    p = p[iphotons]
                    amax = amax[iphotons]

        samples = torch.transpose(torch.vstack((p, torch.log10(amax), torch.log10(nu))), 0, 1)

        kabs = 10.**self.kabs_y_scaler.inverse_transform(self.kabs_model(self.kabs_x_scaler.transform(samples))).detach().flatten()

        return kabs
    
    def interpolate_ksca(self, p, amax, nu):
        samples = np.vstack((p.flatten(), np.log10(amax).flatten(), np.log10(nu).flatten())).T

        interpolated = scipy.interpolate.interpn((self.p, np.log10(self.amax.value), np.log10(self.nu.value)), np.log10(self.ksca), samples, method="cubic")

        return 10.**interpolated.reshape(p.shape)

    def ml_ksca(self, p=None, amax=None, nu=None, photon_list=None, iphotons=None):
        if photon_list is not None:
            p = wp.to_torch(photon_list.p)
            amax = wp.to_torch(photon_list.amax)
            if nu is None:
                nu = wp.to_torch(photon_list.frequency)

                if iphotons is not None:
                    nu = nu[iphotons]
                    p = p[iphotons]
                    amax = amax[iphotons]
            else:
                if nu.size(0) != p.size(0):
                    p = p[iphotons]
                    amax = amax[iphotons]

        samples = torch.transpose(torch.vstack((p, torch.log10(amax), torch.log10(nu))), 0, 1)

        ksca = 10.**self.ksca_y_scaler.inverse_transform(self.ksca_model(self.ksca_x_scaler.transform(samples))).detach().flatten()

        return ksca

    def interpolate_kext(self, p, amax, nu):
        if nu.unit.is_equivalent(u.GHz):
            nu = nu.to(u.GHz).value
        elif nu.unit.is_equivalent(u.cm):
            nu = (const.c / nu).decompose().to(u.GHz)
        return self.interpolate_kabs(p, amax, nu.value) + self.interpolate_ksca(p, amax, nu.value)

    def ml_kext(self, p=None, amax=None, nu=None, photon_list=None, iphotons=None):
        if photon_list is not None:
            p = wp.to_torch(photon_list.p)
            amax = wp.to_torch(photon_list.amax)
            if nu is None:
                nu = wp.to_torch(photon_list.frequency)
            else:
                if nu.size(0) != p.size(0):
                    p = p[iphotons]
                    amax = amax[iphotons]

        samples = torch.transpose(torch.vstack((p, torch.log10(amax), torch.log10(nu))), 0, 1)

        return 10.**self.kabs_y_scaler.inverse_transform(self.kabs_model(self.kabs_x_scaler.transform(samples))).detach().flatten() + \
                10.**self.ksca_y_scaler.inverse_transform(self.ksca_model(self.ksca_x_scaler.transform(samples))).detach().flatten()

    def interpolate_albedo(self, p, amax, nu):
        kabs = self.interpolate_kabs(p, amax, nu)
        ksca = self.interpolate_ksca(p, amax, nu)

        return ksca / (kabs + ksca)

    def ml_albedo(self, p=None, amax=None, nu=None, photon_list=None, iphotons=None):
        if photon_list is not None:
            p = wp.to_torch(photon_list.p)
            amax = wp.to_torch(photon_list.amax)
            if nu is None:
                nu = wp.to_torch(photon_list.frequency)
            else:
                if nu.size(0) != p.size(0):
                    p = p[iphotons]
                    amax = amax[iphotons]

        samples = torch.transpose(torch.vstack((p, torch.log10(amax), torch.log10(nu))), 0, 1)

        kabs = 10.**self.kabs_y_scaler.inverse_transform(self.kabs_model(self.kabs_x_scaler.transform(samples))).detach().flatten()
        ksca = 10.**self.ksca_y_scaler.inverse_transform(self.ksca_model(self.ksca_x_scaler.transform(samples))).detach().flatten()

        return ksca / (kabs + ksca)

    def absorb(self, temperature):
        nphotons = frequency.numpy().size

        cost = -1. + 2*np.random.rand(nphotons)
        sint = np.sqrt(1. - cost**2)
        phi = 2*np.pi*np.random.rand(nphotons)

        direction = np.array([sint*np.cos(phi), sint*np.sin(phi), cost]).T

        frequency = self.random_nu(temperature)

        return direction, frequency

    def random_nu_manual(self, p, amax, temperature, ksi=None, batch_size=100000):
        import interpn

        if ksi is None:
            nphotons = temperature.size
            ksi = np.random.rand(nphotons)

        if not hasattr(self, "random_nu_CPD"):
            random_nu_PDF = np.array([self.kabs * models.BlackBody(temperature=T*u.K)(self.nu) for T in self.temperature])
            self.random_nu_CPD = scipy.integrate.cumulative_trapezoid(random_nu_PDF, self.nu, axis=-1, initial=0.)
            self.random_nu_CPD /= self.random_nu_CPD[:,:,:,-1:]
            self.drandom_nu_CPD_dT = np.gradient(self.random_nu_CPD, self.temperature, axis=0)

        count = 0

        dims = (self.temperature.size, self.p.size, self.amax.size, self.nu.size)
        starts = np.array([np.log10(self.temperature.min()), self.p.min(), np.log10(self.amax.min().value), np.log10(self.nu.min().value)])
        steps = np.array([(np.log10(self.temperature[1]) - np.log10(self.temperature[0])), self.p[1] - self.p[0], (np.log10(self.amax[1].value) - np.log10(self.amax[0].value)), (np.log10(self.nu[1].value) - np.log10(self.nu[0].value))])

        interpolator = interpn.MulticubicRegular.new(dims, starts, steps, self.random_nu_CPD)

        frequency = []
        while count < temperature.size:
            n = min(batch_size, temperature.size - count)
            p_batch = p[count:count+n]
            amax_batch = amax[count:count+n]
            temperature_batch = temperature[count:count+n]
            ksi_batch = ksi[count:count+n]

            samples = np.vstack([np.repeat(np.array([np.log10(temperature_batch), p_batch, np.log10(amax_batch.value)]), self.nu.size, axis=1), 
                                 np.tile(np.log10(self.nu.value), p_batch.size)])

            random_nu_CPD = interpolator.eval(samples)
            random_nu_CPD = random_nu_CPD.reshape((p_batch.size, self.nu.size))

            i = np.argmax(ksi_batch[:,np.newaxis] < random_nu_CPD, axis=1)
    
            frequency_batch = (ksi_batch - random_nu_CPD[np.arange(random_nu_CPD.shape[0]),i-1]) * (self.nu[i] - self.nu[i-1]) / \
                    (random_nu_CPD[np.arange(random_nu_CPD.shape[0]),i] - random_nu_CPD[np.arange(random_nu_CPD.shape[0]),i-1]) + \
                    self.nu[i-1]
            
            frequency.append(frequency_batch)
            
            count += n

        frequency = np.concatenate(frequency).value

        return frequency
    
    def random_nu_ml(self, p, amax, temperature):
        nphotons = temperature.size
        ksi = torch.rand(int(nphotons), device=wp.device_to_torch(wp.get_device()), dtype=torch.float32)
        test_x = torch.transpose(torch.vstack((torch.tensor(p, dtype=torch.float32), torch.log10(torch.tensor(amax, dtype=torch.float32)), torch.log10(torch.tensor(temperature, dtype=torch.float32)), ksi)), 0, 1)
        test_x = self.random_nu_x_scaler.transform(test_x)

        log10_nu = torch.clamp(self.random_nu_y_scaler.inverse_transform(self.random_nu_model(test_x).detach()), self.log10_nu_min, self.log10_nu_max)

        return 10.**log10_nu.numpy()

    def random_nu(self, photon_list, subset=None):
        p = wp.to_torch(photon_list.p)
        amax = wp.to_torch(photon_list.amax)
        temperature = wp.to_torch(photon_list.temperature)
        if subset is not None:
            p = p[subset]
            amax = amax[subset]
            temperature = temperature[subset]
            
        nphotons = temperature.size(0)
        ksi = torch.rand(int(nphotons), device=wp.device_to_torch(wp.get_device()), dtype=torch.float32)

        test_x = torch.transpose(torch.vstack((p, torch.log10(amax), torch.log10(temperature), ksi)), 0, 1)
        test_x = self.random_nu_x_scaler.transform(test_x)

        if nphotons > 250000:
            test_x = TensorDataset(test_x)
            loader = DataLoader(test_x, batch_size=250000)

            log10_nu = torch.cat([torch.clamp(self.random_nu_y_scaler.inverse_transform(self.random_nu_model(X).detach()), self.log10_nu_min, self.log10_nu_max) for X, in loader], 0)
        else:
            log10_nu = torch.clamp(self.random_nu_y_scaler.inverse_transform(self.random_nu_model(test_x).detach()), self.log10_nu_min, self.log10_nu_max)
        
        nu = wp.from_torch(10.**torch.flatten(log10_nu))

        return nu

    def planck_mean_opacity(self, p, amax, temperature):
        vectorized_bb = np.vectorize(lambda p, a, T: self.kmean.cgs.value * scipy.integrate.trapezoid(self.ml_kabs(torch.tensor(p, dtype=torch.float32).expand(self.nu.size), 
                torch.tensor(a, dtype=torch.float32).expand(self.nu.size), torch.tensor(self.nu.value, dtype=torch.float32)) * \
                models.BlackBody(temperature=T*u.K)(self.nu).cgs.value, self.nu.to(u.Hz).value) * np.pi / (const.sigma_sb.cgs.value * T**4))

        return vectorized_bb(p, amax, temperature)

    def ml_planck_mean_opacity(self, p, amax, temperature):
        samples = torch.transpose(torch.vstack((p, torch.log10(amax), torch.log10(temperature))), 0, 1)

        return 10.**self.pmo_y_scaler.inverse_transform(self.pmo_model(self.pmo_x_scaler.transform(samples))).detach().flatten()

    def ml_step(self, photon_list, s, iphotons):
        nphotons = iphotons.size(0)

        test_y = torch.transpose(torch.vstack((torch.log10(wp.to_torch(photon_list.frequency)[iphotons]),
                              torch.log10(wp.to_torch(photon_list.temperature)[iphotons]),
                              torch.log10(wp.to_torch(photon_list.amax)[iphotons]),
                              torch.log10(wp.to_torch(photon_list.p)[iphotons]),
                              torch.log10(wp.to_torch(photon_list.density)[iphotons] * wp.to_torch(photon_list.kabs)[iphotons] * s[iphotons]),
                              )), 0, 1)

        test_x = self.ml_step_model.condition(self.ml_step_y_scaler.transform(test_y)).sample(test_y.size(0)).detach()
        test_x = self.ml_step_x_scaler.inverse_transform(test_x)

        return 10.**torch.clamp(test_x[:,0], self.log10_nu0_min, self.log10_nu0_max), 10.**test_x[:,1], 10.**test_x[:,2], test_x[:,3], test_x[:,4], torch.zeros(test_x.size(0)), test_x[:,5], test_x[:,6], torch.zeros(test_x.size(0))

    def initialize_model(self, model="random_nu", input_size=2, output_size=1, hidden_units=(48, 48, 48)):
        if model == 'ml_step':
            self.ml_step_model = StackedNVP(input_size, hidden_units=hidden_units[0], conditional_size=output_size, nflows=len(hidden_units))
        elif model == 'random_nu':
            self.random_nu_model = MultiLayerPerceptron(input_size, output_size, hidden_units=hidden_units)
        elif model == "kabs":
            self.kabs_model = MultiLayerPerceptron(input_size, output_size, hidden_units=hidden_units)
        elif model == "ksca":
            self.ksca_model = MultiLayerPerceptron(input_size, output_size, hidden_units=hidden_units)
        elif model == "pmo":
            self.pmo_model = MultiLayerPerceptron(input_size, output_size, hidden_units=hidden_units)

    def learn(self, model="random_nu", nsamples=200000, test_split=0.1, valid_split=0.2, hidden_units=(48, 48, 48),
            tau_range=(3.0, 1e4), temperature_range=(0.1*u.K, 1e4*u.K), amax_range=(1*u.micron, 10.0*u.cm), p_range=(2.5, 4.5), 
            nu_range=None, overwrite=False):
        """
        Learn a model for either the random_nu function or the ml_step function.
        
        Parameters
        ----------
        model : str
            The model to learn. Either "random_nu" or "ml_step".
        nsamples : int
            The total number of samples to generate for the learning process (including validation and testing)
        test_split : float
            The fraction of the samples to use for testing.
        valid_split : float
            The fraction of the remaining samples (after testing) to use for validation.
        hidden_units : tuple
            The number of hidden units in each layer of the neural network.
        tau_range : tuple
            The range of optical depths to sample from for the ml_step model.
        temperature_range : tuple
            The range of temperatures to sample from for the ml_step model.
        amax_range : tuple
            The range of maximum grain sizes to sample from for the ml_step model.
        p_range : tuple
            The range of power-law indices to sample from for the ml_step model.
        nu_range : tuple
            The range of frequencies to sample from (in GHz) for the ml_step model. If None, use the full range of the dust opacities.
        """
        self.current_model = model
        self.nsamples = nsamples
        self.test_split = test_split
        self.valid_split = valid_split
        self.learning = model
        self.overwrite = overwrite

        # Set up the NN

        if model == "random_nu":
            input_size, output_size = 4, 1
        elif model == "ml_step":
            input_size, output_size = 7, 5

            if nu_range is None:
                nu_range = (self.nu.value.min(), self.nu.value.max())

            self.log10_nu0_min = np.log10(nu_range[0].to(u.GHz).value)
            self.log10_nu0_max = np.log10(nu_range[1].to(u.GHz).value)
            self.log10_T_min = np.log10(temperature_range[0].to(u.K).value)
            self.log10_T_max = np.log10(temperature_range[1].to(u.K).value)
            self.log10_a_min = np.log10(amax_range[0].to(u.cm).value)
            self.log10_a_max = np.log10(amax_range[1].to(u.cm).value)
            self.p_min = p_range[0]
            self.p_max = p_range[1]
            self.log10_tau_cell_nu0_min = np.log10(tau_range[0])
            self.log10_tau_cell_nu0_max = np.log10(tau_range[1])
        elif model in ["kabs", "ksca","pmo"]:
            input_size, output_size = 3, 1

        self.initialize_model(model=model, input_size=input_size, output_size=output_size, hidden_units=hidden_units)

        # Wrap the model in lightning

        self.dustLM = DustLightningModule(getattr(self, model+"_model"))

        self.trainer = pl.Trainer(max_epochs=0, callbacks=[pl.callbacks.ModelCheckpoint(save_last=True, 
                                                                                        dirpath=f'{model}_lightning_logs')])

    def fit(self, epochs=10, batch_size=100, num_workers=1, ckpt_path=None):
        '''
        Run the model fit.

        Parameters:
        -----------
        epochs : int
            The number of epochs to train the model.
        '''
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.trainer.fit_loop.max_epochs += epochs
        self.trainer.fit(model=self.dustLM, datamodule=self, ckpt_path=ckpt_path)

    def test_model(self, plot=False):
        '''
        Test the model fit.

        Parameters:
        -----------
        plot : bool
            Whether to plot the results after training.
        '''
        # Test the model.

        self.trainer.test(model=self.dustLM, datamodule=self)

        # Plot the result

        if plot:
            import matplotlib.pyplot as plt

            if self.current_model == "random_nu":
                self.plot_triangle_plots(model=self.current_model)
            elif self.current_model in ["kabs", "ksca"]:
                self.plot_opacity_model(model=self.current_model)
            elif self.current_model == "pmo":
                self.plot_pmo_model()
            else:
                #self.plot_ml_step()
                self.plot_triangle_plots(model=self.current_model)

    def prepare_data(self):
        if hasattr(self, "dataset") and not self.overwrite:
            return
        
        if hasattr(self, f"prepare_data_{self.current_model}"):
            getattr(self, f"prepare_data_{self.current_model}")()
        else:
            raise NotImplementedError(f"Data preparation for model {self.current_model} not implemented.")

    def prepare_data_random_nu(self):
        sampler = scipy.stats.qmc.LatinHypercube(d=4)
        samples = sampler.random(self.nsamples)

        samples[:,0] = samples[:,0] * (self.p.max() - self.p.min()) + self.p.min()
        samples[:,1] = samples[:,1] * (np.log10(self.amax.max().value) - np.log10(self.amax.min().value)) + np.log10(self.amax.min().value)
        samples[:,2] = samples[:,2] * (np.log10(self.temperature.max()) - np.log10(self.temperature.min())) + np.log10(self.temperature.min())

        X = torch.tensor(samples, dtype=torch.float32)
        y = torch.tensor(np.log10(self.random_nu_manual(samples[:,0], 10.**samples[:,1]*self.amax.unit, 10.**samples[:,2], ksi=samples[:,3])), dtype=torch.float32)

        X_scaler = StandardScaler()
        X_scaler.fit(X)
        X = X_scaler.transform(X)

        y_scaler = StandardScaler()
        y_scaler.fit(y)
        y = y_scaler.transform(y)

        self.random_nu_x_scaler = X_scaler
        self.random_nu_y_scaler = y_scaler

        self.dataset = TensorDataset(X, y)

    def prepare_data_kabs(self):
        sampler = scipy.stats.qmc.LatinHypercube(d=3)
        samples = sampler.random(self.nsamples)

        samples[:,0] = samples[:,0] * (self.p.max() - self.p.min()) + self.p.min()
        samples[:,1] = samples[:,1] * (np.log10(self.amax.max().value) - np.log10(self.amax.min().value)) + np.log10(self.amax.min().value)
        samples[:,2] = samples[:,2] * (np.log10(self.nu.max().value) - np.log10(self.nu.min().value)) + np.log10(self.nu.min().value)

        log10_kabs = scipy.interpolate.interpn((self.p, np.log10(self.amax.value), np.log10(self.nu.value)), np.log10(self.kabs), samples, method="cubic")

        X = torch.tensor(samples, dtype=torch.float32)
        y = torch.tensor(log10_kabs, dtype=torch.float32)
        
        X_scaler = StandardScaler()
        X_scaler.fit(X)
        X = X_scaler.transform(X)

        y_scaler = StandardScaler()
        y_scaler.fit(y)
        y = y_scaler.transform(y)

        self.kabs_x_scaler = X_scaler
        self.kabs_y_scaler = y_scaler

        self.dataset = TensorDataset(X, y)

    def prepare_data_ksca(self):
        sampler = scipy.stats.qmc.LatinHypercube(d=3)
        samples = sampler.random(self.nsamples)

        samples[:,0] = samples[:,0] * (self.p.max() - self.p.min()) + self.p.min()
        samples[:,1] = samples[:,1] * (np.log10(self.amax.max().value) - np.log10(self.amax.min().value)) + np.log10(self.amax.min().value)
        samples[:,2] = samples[:,2] * (np.log10(self.nu.max().value) - np.log10(self.nu.min().value)) + np.log10(self.nu.min().value)

        log10_ksca = scipy.interpolate.interpn((self.p, np.log10(self.amax.value), np.log10(self.nu.value)), np.log10(self.ksca), samples, method="cubic")

        X = torch.tensor(samples, dtype=torch.float32)
        y = torch.tensor(log10_ksca, dtype=torch.float32)

        X_scaler = StandardScaler()
        X_scaler.fit(X)
        X = X_scaler.transform(X)

        y_scaler = StandardScaler()
        y_scaler.fit(y)
        y = y_scaler.transform(y)

        self.ksca_x_scaler = X_scaler
        self.ksca_y_scaler = y_scaler

        self.dataset = TensorDataset(X, y)

    def prepare_data_pmo(self, device='cpu'):
        sampler = scipy.stats.qmc.LatinHypercube(d=3)
        samples = sampler.random(self.nsamples)

        samples[:,0] = samples[:,0] * (self.p.max() - self.p.min()) + self.p.min()
        samples[:,1] = samples[:,1] * (np.log10(self.amax.max().value) - np.log10(self.amax.min().value)) + np.log10(self.amax.min().value)
        samples[:,2] = samples[:,2] * (np.log10(self.temperature.max()) - np.log10(self.temperature.min())) + np.log10(self.temperature.min())

        log10_pmo = np.log10(self.planck_mean_opacity(samples[:,0], 10.**samples[:,1], 10.**samples[:,2]))

        X = torch.tensor(samples, dtype=torch.float32)
        y = torch.tensor(log10_pmo, dtype=torch.float32)

        X_scaler = StandardScaler()
        X_scaler.fit(X)
        X = X_scaler.transform(X)

        y_scaler = StandardScaler()
        y_scaler.fit(y)
        y = y_scaler.transform(y)

        self.pmo_x_scaler = X_scaler
        self.pmo_y_scaler = y_scaler

        self.dataset = TensorDataset(X, y)

    def prepare_data_ml_step(self, device='cpu'):
        if os.path.exists("sim_results.csv"):
            df = pd.read_csv("sim_results.csv", index_col=0)

            self.log10_nu0_min = df['log10_nu0'].min()
            self.log10_nu0_max = df['log10_nu0'].max()
            self.log10_T_min = df['log10_T'].min()
            self.log10_T_max = df['log10_T'].max()
            self.log10_a_min = df['log10_amax'].min()
            self.log10_a_max = df['log10_amax'].max()
            self.p_min = df['p'].min()
            self.p_max = df['p'].max()
            self.log10_tau_cell_nu0_min = df['log10_tau_cell_nu0'].min()
            self.log10_tau_cell_nu0_max = df['log10_tau_cell_nu0'].max()
        else:
            df = self.run_dust_simulation(nphotons=self.nsamples, 
                                          tau_range=(10.**self.log10_tau_cell_nu0_min, 10.**self.log10_tau_cell_nu0_max),
                                          temperature_range=(10.**self.log10_T_min*u.K, 10.**self.log10_T_max*u.K), 
                                          amax_range=(10.**self.log10_a_min*u.cm, 10.**self.log10_a_max*u.cm), 
                                          p_range=(self.p_min, self.p_max), 
                                          nu_range=(10.**self.log10_nu0_min*u.GHz, 10.**self.log10_nu0_max*u.GHz))
            df.to_csv("sim_results.csv")

        features = ["log10_nu", "log10_Eabs", "log10_tau", "yaw", "pitch", "direction_yaw", "direction_pitch"]
        targets = ["log10_nu0", "log10_T", "log10_amax", "p", "log10_tau_cell_nu0"]

        df.loc[:, "log10_tau"] = np.where(np.logical_or(df["log10_tau"] < -6.5, np.isnan(df["log10_tau"].values)), np.log10(-np.log(1. - np.random.rand(len(df)))), df["log10_tau"])

        data = df.loc[:, targets+features].values
        self.ml_step_x_scaler = StandardScaler()
        self.ml_step_x_scaler.fit(torch.tensor(df.loc[:, features].values, dtype=torch.float32))
        self.ml_step_y_scaler = StandardScaler()
        self.ml_step_y_scaler.fit(torch.tensor(df.loc[:, targets].values, dtype=torch.float32))
        self.ml_step_features = features
        self.ml_step_limits = {}
        for key in features:
            self.ml_step_limits[key] = (df[key].min(), df[key].max())

        self.df = df
        self.nsamples = len(df)

        X = self.ml_step_x_scaler.transform(torch.tensor(df.loc[:, features].values, dtype=torch.float32))
        y = self.ml_step_y_scaler.transform(torch.tensor(df.loc[:, targets].values, dtype=torch.float32))

        self.dataset = TensorDataset(X, y)

    def run_dust_simulation(self, nphotons=1000, tau_range=(3.0, 1e4), temperature_range=(0.1*u.K, 1e4*u.K), 
                            amax_range=(1*u.micron, 10.0*u.cm), p_range=(2.5, 4.5), nu_range=None, use_ml_step=False, 
                            position=0):
        """
        Run a dust simulation that can be used to learn an ML-step model with the given parameters.

        Parameters
        ----------
        nphotons : int
            The number of photons to simulate.
        tau_range : tuple
            The range of optical depths to sample from (in log10).
        temperature_range : tuple
            The range of temperatures to sample from (in log10).
        amax_range : tuple
            The range of maximum grain sizes to sample from (in log10).
        p_range : tuple
            The range of power-law indices to sample from.
        nu_range : tuple
            The range of frequencies to sample from (in GHz). If None, use the full range of the dust opacities.
        """
        if nu_range is None:
            nu_range = (self.nu.min(), self.nu.max())

        # Set up the star.

        star = BlackbodyStar()

        # Set up the grid.

        grid = UniformSphericalGrid(ncells=1, dr=1.0*u.au, mirror=False)

        density = np.ones(grid.shape) * 1e-16 * u.g / u.cm**3

        grid.add_density(density, self)
        grid.add_sources(star)

        # Emit the photons

        photon_list = grid.emit(nphotons, wavelength="random", scattering=False)

        initial_direction = np.zeros((nphotons, 3), dtype=np.float32)
        initial_direction[:,0] = 1.
        photon_list.direction = wp.array(initial_direction, dtype=wp.vec3)

        photon_list.frequency = wp.array(10.**np.random.uniform(np.log10(nu_range[0].value), np.log10(nu_range[1].value), nphotons), dtype=float)
        original_frequency = photon_list.frequency.numpy().copy()

        photon_list.temperature = wp.array(10.**np.random.uniform(np.log10(temperature_range[0].to(u.K).value), np.log10(temperature_range[1].to(u.K).value), nphotons), dtype=float)

        photon_list.amax = wp.array(10.**np.random.uniform(np.log10(amax_range[0].to(u.cm).value), np.log10(amax_range[1].to(u.cm).value), nphotons), dtype=float)
        photon_list.p = wp.array(np.random.uniform(p_range[0], p_range[1], nphotons), dtype=float)

        tau = 10.**np.random.uniform(np.log10(tau_range[0]), np.log10(tau_range[1]), nphotons)
        photon_list.density = wp.array((tau / (self.kmean * self.interpolate_kabs(photon_list.p.numpy(), photon_list.amax.numpy(), photon_list.frequency.numpy()) * 1.*u.au) * self.kmean).to(1 / u.au), dtype=float)

        grid.propagate_photons(photon_list, learning=True, use_ml_step=use_ml_step)

        # Calculate roll, pitch, and yaw for the position relative to where it started.
        # Also calculate roll, pitch, and yaw for the direction relative to the radial vector where it exits.

        ypr = []
        direction_ypr = []
        for (direction0, position, direction) in zip(initial_direction, photon_list.position.numpy(), photon_list.direction.numpy()):
            rot, _ = Rotation.align_vectors(position, direction0)
            ypr.append(rot.as_euler('ZYX'))
            
            rot, _ = Rotation.align_vectors(rot.inv().apply(direction), direction0)
            direction_ypr.append(rot.as_euler('ZYX'))

        ypr = np.array(ypr)
        direction_ypr = np.array(direction_ypr)

        # Store the results in a pandas DataFrame

        df = pd.DataFrame({"log10_nu0":np.log10(original_frequency),
                       "log10_T":np.log10(photon_list.temperature.numpy()),
                       "log10_amax":np.log10(photon_list.amax.numpy()),
                       "p":photon_list.p.numpy(),
                       "log10_tau_cell_nu0":np.log10(tau),
                       "log10_nu":np.log10(photon_list.frequency.numpy().copy()),
                       "log10_Eabs":np.log10(np.where(photon_list.deposited_energy.numpy() > 0, photon_list.deposited_energy.numpy(), photon_list.deposited_energy.numpy().min()/100)/photon_list.energy.numpy()),
                       "log10_tau":np.log10(photon_list.tau.numpy().copy()),
                       "yaw":ypr[:,0],
                       "pitch":ypr[:,1],
                       "direction_yaw":direction_ypr[:,0],
                       "direction_pitch":direction_ypr[:,1]})

        return df

    # DataModule functions

    def plot_ml_step(self, tau=30., temperature=100.0*u.K, amax=1.*u.micron, p=3.5, nu=1e3*u.GHz, nsamples=1000, 
                     plot_columns=np.array(["log10_nu", "log10_Eabs", "log10_tau", "yaw", "pitch", "direction_yaw", "direction_pitch"])):
        """
        Plot the samples drawn from a sphere with the provided optical depth, temperature, and frequency.

        Parameters
        ----------
        tau : float
            The log-optical depth to use for the samples.
        temperature : float
            The temperature to use for the samples.
        amax : float
            The maximum grain size to use for the samples.
        p : float
            The power-law index to use for the samples.
        nu : float
            The frequency to use for the samples.
        nsamples : int
            The number of samples to generate.
        plot_columns : numpy.ndarray
            The columns to plot in the triangle plots.
        """
        
        df = self.run_dust_simulation(nphotons=nsamples, 
                                      tau_range=(tau,tau), 
                                      temperature_range=(temperature, temperature), 
                                      amax_range=(amax, amax), 
                                      p_range=(p, p), 
                                      nu_range=(nu, nu), 
                                      use_ml_step=False)

        df.loc[:, "log10_tau"] = np.where(np.logical_or(df["log10_tau"] < -6.5, np.isnan(df["log10_tau"].values)), np.log10(-np.log(1. - np.random.rand(len(df)))), df["log10_tau"])

        features = np.array(["log10_nu", "log10_Eabs", "log10_tau", "yaw", "pitch", "direction_yaw", "direction_pitch"])
        targets = ["log10_nu0", "log10_T", "log10_amax", "p", "log10_tau_cell_nu0"]

        X = self.ml_step_x_scaler.transform(torch.tensor(df.loc[:, features].values, dtype=torch.float32))
        y = self.ml_step_y_scaler.transform(torch.tensor(df.loc[:, targets].values, dtype=torch.float32))

        self.dataset = TensorDataset(X, y)
        self.nsamples = nsamples
        self.test_split = 0.98
        self.valid_split = 0.01
        self.batch_size = 10000

        if hasattr(self, "train") and hasattr(self, "valid") and hasattr(self, "test"):
            del self.train, self.valid, self.test

        self.plot_triangle_plots(plot_columns=plot_columns)

    def plot_opacity_model(self, model='kabs'):
        """
        Plot the learned opacity model against the interpolated opacity.

        Parameters
        ----------
        model : str
            The model to plot. Either "kabs" or "ksca".
        """
        import matplotlib.pyplot as plt

        log10_nu = np.linspace(np.log10(self.nu.min().value), np.log10(self.nu.max().value), 10)
        log10_lam = np.log10((const.c / (10.**log10_nu * u.GHz)).to(u.cm).value)
        log10_amax = np.repeat(np.random.uniform(0, 1, 1)*(np.log10(self.amax.max().value) - np.log10(self.amax.min().value)) + np.log10(self.amax.min().value), 10)
        p = np.repeat(np.random.uniform(0, 1, 1)*(self.p.max() - self.p.min()) + self.p.min(), 10)

        print(f"log10_amax: {log10_amax[0]}, p: {p[0]}")

        samples = np.vstack((p, log10_amax, log10_nu)).T

        interpolated = np.log10(getattr(self, f"interpolate_{model}")(p, 10.**log10_amax, 10.**log10_nu))
        nned = getattr(self, f'{model}_y_scaler').inverse_transform(getattr(self, f'{model}_model')(getattr(self, f'{model}_x_scaler').transform(torch.tensor(samples, dtype=torch.float32)))).detach().numpy()

        plt.plot(log10_lam, interpolated)
        plt.plot(log10_lam, nned)
        plt.show()

    def plot_pmo_model(self):
        """
        Plot the learned Planck mean opacity model against the interpolated Planck mean opacity.
        """
        import matplotlib.pyplot as plt

        p = np.repeat(np.random.uniform(0, 1, 1)*(self.p.max() - self.p.min()) + self.p.min(), 100)
        log10_amax = np.repeat(np.random.uniform(0, 1, 1)*(np.log10(self.amax.max().value) - np.log10(self.amax.min().value)) + np.log10(self.amax.min().value), 100)
        log10_temperature = np.linspace(np.log10(self.temperature.min()), np.log10(self.temperature.max()), 100)

        print(f"log10_amax: {log10_amax[0]}, p: {p[0]}, log10_temperature: {log10_temperature[0]}")

        samples = np.vstack((p, log10_amax, log10_temperature)).T

        interpolated = np.log10(self.planck_mean_opacity(p, 10.**log10_amax, 10.**log10_temperature))
        nned = self.pmo_y_scaler.inverse_transform(self.pmo_model(self.pmo_x_scaler.transform(torch.tensor(samples, dtype=torch.float32)))).detach().numpy()

        plt.plot(log10_temperature, interpolated)
        plt.plot(log10_temperature, nned)
        plt.show()

    def plot_random_nu_model(self, nsamples=100000):
        """
        Plot samples drawn from the learned random_nu model against samples drawn from the random_nu_manual function.

        Parameters
        ----------
        nsamples : int
            The number of samples to draw from each model for the plot.
        """
        import matplotlib.pyplot as plt

        T = np.repeat(10.**np.random.uniform(-1., 4., 1), nsamples)
        amax = np.repeat(10.**np.random.uniform(-4., 1., 1), nsamples)
        p = np.repeat(np.random.uniform(2.5, 4.5, 1), nsamples)
        print(f"p: {p[0]}, amax: {amax[0]}, T: {T[0]}")

        nu = self.random_nu_manual(p, amax*self.amax.unit, T)
        nu2 = self.random_nu_ml(p, amax, T)

        counts, bins, _ = plt.hist(nu, 100)
        plt.hist(nu2, 100)

        bb = models.BlackBody(temperature=T[0] * u.K)
        pdf = bb(bins[1:]*u.GHz).value * self.interpolate_kabs(np.repeat(p[0], bins[1:].size), np.repeat(amax[0], bins[1:].size), bins[1:])
        pdf *= counts.max() / pdf.max()

        plt.plot(bins[1:], pdf, '-')

        plt.show()

    def plot_triangle_plots(self, model="ml_step", nsamples=200000, batch_size=100, num_workers=1, plot_columns='all'):
        import matplotlib.pyplot as plt

        if self.trainer is None and hasattr(self, f"{model}_model"):
            self.dustLM = DustLightningModule(getattr(self, f"{model}_model"))

            self.trainer = pl.Trainer()

            self.learning = model
            self.current_model = model
            self.nsamples = nsamples
            self.test_split = 0.98
            self.valid_split = 0.01
            self.batch_size = 10000

            self.batch_size = batch_size
            self.num_workers = num_workers

        if self.trainer is not None:
            if model == "ml_step":
                X_pred = self.trainer.predict(self.dustLM, datamodule=self)
                X_pred = torch.cat(X_pred)
                y_pred = torch.cat([batch[1] for batch in self.predict_dataloader()])
            elif model == "random_nu":
                y_pred = self.trainer.predict(self.dustLM, datamodule=self)
                y_pred = torch.cat(y_pred)
                X_pred = torch.cat([batch[0] for batch in self.predict_dataloader()])

            X_true = torch.cat([batch[0] for batch in self.predict_dataloader()])
            y_true = torch.cat([batch[1] for batch in self.predict_dataloader()])

            predict = True

        if model == "ml_step":
            features = np.array(["log10_nu", "log10_Eabs", "log10_tau", "yaw", "pitch", "direction_yaw", "direction_pitch"])
            targets = np.array(["log10_nu0", "log10_T", "log10_amax", "p", "log10_tau_cell_nu0"])
        elif model == "random_nu":
            features = np.array(["p", "log10_amax", "log10_temperature", "ksi"])
            targets = np.array(["log10_nu"])

            y_true = torch.unsqueeze(y_true, 1)

        if plot_columns == 'all':
            columns = np.concatenate((targets, features))
        else:
            columns = np.array(plot_columns)

        df_true = pd.DataFrame(torch.cat([getattr(self, f"{model}_y_scaler").inverse_transform(y_true), getattr(self, f"{model}_x_scaler").inverse_transform(X_true)], dim=1).numpy(), columns=np.concatenate((targets, features)))
        df_pred = pd.DataFrame(torch.cat([getattr(self, f"{model}_y_scaler").inverse_transform(y_pred), getattr(self, f"{model}_x_scaler").inverse_transform(X_pred)], dim=1).numpy(), columns=np.concatenate((targets, features)))

        fig, ax = plt.subplots(nrows=len(columns), ncols=len(columns), figsize=(11,11))

        if len(columns) == 1:
            ax = np.array([[ax]])

        for i, key1 in enumerate(columns):
            for j, key2 in enumerate(columns):
                if key1 == key2:
                    ax[i,j].hist(df_true[key1], bins=50, histtype='step', density=True)
                    if predict:
                        ax[i,j].hist(df_pred[key1], bins=50, histtype='step', density=True)
                elif i > j:
                    ax[i,j].scatter(df_true[key2], df_true[key1], marker='.', s=1.0, alpha=1.0)

                    if predict:
                        ax[i,j].scatter(df_pred[key2], df_pred[key1], marker='.', s=1.0, alpha=1.0)
                elif i < j:
                    ax[i,j].set_axis_off()

                if i == len(columns) - 1:
                    ax[i,j].set_xlabel(key2)

            ax[i,0].set_ylabel(key1)

        plt.show()

    def setup(self, stage=None):
        if hasattr(self, "train") and hasattr(self, "valid") and hasattr(self, "test") and not self.overwrite:
            return
        test_size = int(self.test_split * self.nsamples)
        valid_size = int((self.nsamples - test_size)*self.valid_split)
        train_size = self.nsamples - test_size - valid_size
        train_val_tmp, self.test = random_split(self.dataset, [train_size + valid_size, test_size], generator=torch.Generator().manual_seed(1))
        self.train, self.valid = random_split(train_val_tmp, [train_size, valid_size], generator=torch.Generator().manual_seed(2))

        if self.overwrite:
            self.overwrite = False

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def state_dict(self):
        state_dict = {
            "dust_properties":{
                "lam": self.lam,
                "amax": self.amax,
                "p": self.p,
                "kabs": self.kabs*self.kmean,
                "ksca": self.ksca*self.kmean,
            },
        }

        for attr in ["kabs", "ksca", "pmo", "random_nu"]:
            if hasattr(self, f"{attr}_model"):
                state_dict[f"{attr}_state_dict"] = getattr(self, f'{attr}_model').state_dict()
                state_dict[f"{attr}_x_scaler"] = getattr(self, f'{attr}_x_scaler').state_dict()
                state_dict[f"{attr}_y_scaler"] = getattr(self, f'{attr}_y_scaler').state_dict()

        if hasattr(self, "ml_step_model"):
            state_dict["ml_step_state_dict"] = self.ml_step_model.state_dict()

            state_dict["log10_nu0_min"] = self.log10_nu0_min
            state_dict["log10_nu0_max"] = self.log10_nu0_max
            state_dict["log10_T_min"] = self.log10_T_min
            state_dict["log10_T_max"] = self.log10_T_max
            state_dict["log10_tau_cell_nu0_min"] = self.log10_tau_cell_nu0_min
            state_dict["log10_tau_cell_nu0_max"] = self.log10_tau_cell_nu0_max

            state_dict["ml_step_x_scaler"] = self.ml_step_x_scaler.state_dict()
            state_dict["ml_step_y_scaler"] = self.ml_step_y_scaler.state_dict()

            state_dict["ml_step_features"] = self.ml_step_features
            state_dict["ml_step_limits"] = self.ml_step_limits

        return state_dict

    def save(self, filename):
        """
        Save the Dust object to a file.
        
        Parameters
        ----------
        filename : str
            The filename to save the Dust object to.
        """
        torch.save(self.state_dict(), filename)

    def copy(self, device="cpu"):
        return load(self.state_dict(), device=device)


def load(filename, device="cpu"):
    """
    Load a Dust object from a file or state_dict.
    
    Parameters
    ----------
    filename : str, dict, or Dust
        The filename to load the Dust object from, or a state_dict, or a Dust object.
    device : str
        The device to load the Dust object onto.
    """
    if isinstance(filename, str):
        if not os.path.exists(filename):
            if os.path.exists(os.environ["HOME"]+"/.pinballrt/data/dust/"+filename):
                filename = os.environ["HOME"]+"/.pinballrt/data/dust/"+filename
            else:
                web_data_location = 'https://raw.githubusercontent.com/psheehan/pinball-warp/main/pinballrt/tests/data/'+filename
                response = requests.get(web_data_location)
                if response.status_code == 200:
                    if not os.path.exists(os.environ["HOME"]+"/.pinballrt/data/dust"):
                        os.makedirs(os.environ["HOME"]+"/.pinballrt/data/dust")
                    urllib.request.urlretrieve(web_data_location, 
                            os.environ["HOME"]+"/.pinballrt/data/dust/"+filename)
                    filename = os.environ["HOME"]+"/.pinballrt/data/dust/"+filename
                else:
                    print(web_data_location+' does not exist')
                    return
                
        state_dict = torch.load(filename, weights_only=False)
    elif isinstance(filename, dict):
        state_dict = filename
    elif isinstance(filename, Dust):
        state_dict = filename.state_dict()

    d = Dust(**state_dict["dust_properties"], device=device)

    for attr in ["kabs", "ksca", "pmo", "random_nu"]:
        if f"{attr}_state_dict" in state_dict:
            if attr == "random_nu":
                input_size = 4
            else:
                input_size = 3
            hidden_units = [state_dict[f'{attr}_state_dict'][key].shape[0] for key in state_dict[f'{attr}_state_dict'] if 'bias' in key][0:-1]
            d.initialize_model(model=attr, input_size=input_size, output_size=1, hidden_units=hidden_units)

            getattr(d, f'{attr}_model').load_state_dict(state_dict[f'{attr}_state_dict'])
            setattr(d, f'{attr}_x_scaler', StandardScaler())
            getattr(d, f'{attr}_x_scaler').load_state_dict(state_dict[f"{attr}_x_scaler"])
            setattr(d, f'{attr}_y_scaler', StandardScaler())
            getattr(d, f'{attr}_y_scaler').load_state_dict(state_dict[f"{attr}_y_scaler"])

    if "ml_step_state_dict" in state_dict:
        hidden_units = [state_dict['ml_step_state_dict'][key].size(0) for key in state_dict['ml_step_state_dict'] if 'sig_net' in key and '0.weight' in key]

        d.initialize_model(model="ml_step", input_size=7, output_size=5, hidden_units=hidden_units)

        d.ml_step_model.load_state_dict(state_dict['ml_step_state_dict'])

        d.log10_nu0_min = state_dict["log10_nu0_min"]
        d.log10_nu0_max = state_dict["log10_nu0_max"]
        d.log10_T_min = state_dict["log10_T_min"]
        d.log10_T_max = state_dict["log10_T_max"]
        d.log10_tau_cell_nu0_min = state_dict["log10_tau_cell_nu0_min"]
        d.log10_tau_cell_nu0_max = state_dict["log10_tau_cell_nu0_max"]

        d.ml_step_x_scaler = StandardScaler()
        d.ml_step_x_scaler.load_state_dict(state_dict["ml_step_x_scaler"])
        d.ml_step_y_scaler = StandardScaler()
        d.ml_step_y_scaler.load_state_dict(state_dict["ml_step_y_scaler"])

        d.ml_step_features = state_dict["ml_step_features"]
        d.ml_step_limits = state_dict["ml_step_limits"]

    return d

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=(48, 48, 48)):
        super().__init__()
        all_layers = [nn.Flatten()]

        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.SiLU())
            input_size = hidden_unit

        all_layers.append(nn.Linear(hidden_units[-1], output_size))
        self.model = nn.Sequential(*all_layers)
    
    def forward(self, x):
        return self.model(x)

    def loss(self, x, y):
        return nn.functional.mse_loss(self(x), y)
    
def softClampAsymAdvanced(value, negAlpha, posAlpha):
    reLU = torch.nn.ReLU()
    posValues = (2.0 * posAlpha / torch.pi) * torch.arctan(reLU(value) / posAlpha)
    negValues = (2.0 * negAlpha / torch.pi) * torch.arctan(-reLU(-value) / negAlpha)
    return negValues + posValues

class RealNVP(nn.Module):
    def __init__(self, input_size, hidden_units=48, conditional_size=0):
        super().__init__()

        self.d, self.c = input_size, conditional_size

        self.sig_net = nn.Sequential(
                    nn.Linear(self.d + self.c, hidden_units),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_units, self.d),
                    nn.Tanh())

        self.mu_net = nn.Sequential(
                    nn.Linear(self.d + self.c, hidden_units),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_units, self.d),
        )

        self.mask = torch.ones(self.d)
        self.mask[::2] = 0

        base_mu, base_cov = torch.zeros(input_size), torch.eye(input_size)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def condition(self, y):
        self.y = y
        return self

    def forward(self, x, flip=False):
        if flip:
            mask = 1 - self.mask
        else:
            mask = self.mask
        
        # forward
        if self.c > 0:
            sig = (1 - mask) * self.sig_net(torch.cat([x * mask, self.y], dim=1))
            mu = (1 - mask) * self.mu_net(torch.cat([x * mask, self.y], dim=1))
        else:
            sig = (1 - mask) * self.sig_net(x * mask)
            mu = (1 - mask) * self.mu_net(x * mask)
        #sig = softClampAsymAdvanced(sig, 2.0, 0.1)

        z = x * mask + (1 - mask) * (x * torch.exp(sig) + mu)

        log_pz = self.base_dist.log_prob(z)
        log_jacob = (sig * (1 - mask)).sum(-1)

        return z, log_pz, log_jacob

    def inverse(self, Z, flip=False):
        if flip:
            mask = 1 - self.mask
        else:
            mask = self.mask

        if self.c > 0:
            sig = (1 - mask) * self.sig_net(torch.cat([Z * mask, self.y], dim=1))
            mu = (1 - mask) * self.mu_net(torch.cat([Z * mask, self.y], dim=1))
        else:
            sig = self.sig_net(Z * mask)
            mu = self.mu_net(Z * mask)
        #sig = softClampAsymAdvanced(sig, 2.0, 0.1)
        x = Z * mask + (1 - mask) * (Z - mu) * torch.exp(-sig)

        return x


class TrainableLOFTLayer(nn.Module):

    def __init__(self, dim, initial_t, train_t):
        assert(initial_t >= 1.0)
        super().__init__()
        self.dim = dim
        self.rep_t = torch.ones(dim) * (initial_t - 1.0) # reparameterization of t
        self.rep_t = torch.nn.Parameter(self.rep_t, requires_grad=train_t)

        base_mu, base_cov = torch.zeros(dim), torch.eye(dim)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def inverse(self, z):
        t = self.get_t()

        new_value, part1 = TrainableLOFTLayer.LOFT_forward_static(t, z)

        #log_derivatives = - torch.log(part1 + 1.0)

        #log_det = torch.sum(log_derivatives, axis = 1)

        return new_value

    def get_t(self):
        return 1.0 + torch.nn.functional.softplus(self.rep_t)

    def LOFT_forward_static(t, z):
        part1 = torch.max(torch.abs(z) - t, torch.tensor(0.0))
        part2 = torch.min(torch.abs(z), t)

        new_value = torch.sign(z) * (torch.log(part1 + 1) + part2)

        return new_value, part1

    def forward(self, z):
        t = self.get_t()

        part1 = torch.max(torch.abs(z) - t, torch.tensor(0.0))
        part2 = torch.min(torch.abs(z), t)

        new_value = torch.sign(z) * (torch.exp(part1) - 1.0 + part2)

        log_det = torch.sum(part1, axis = 1)
        log_pz = self.base_dist.log_prob(new_value)

        return new_value, log_pz, log_det
    

class StackedNVP(nn.Module):
    def __init__(self, input_size, hidden_units=48, conditional_size=0, nflows=1):
        super().__init__()
        self.bijectors = nn.ModuleList([
            RealNVP(input_size, hidden_units=hidden_units, conditional_size=conditional_size) for _ in range(nflows)
        ])
        self.flips = [True if i%2 else False for i in range(nflows)]

        self.loft = TrainableLOFTLayer(input_size, 1.0, True)

        base_mu, base_cov = torch.zeros(input_size), torch.eye(input_size)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def condition(self, y):
        for bijector in self.bijectors:
            bijector.condition(y)
        return self

    def forward(self, x):
        log_jacobs = []

        x, log_pz, lj = self.loft(x)

        for bijector, f in zip(self.bijectors, self.flips):
            x, log_pz, lj = bijector(x, flip=f)
            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):
        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):
            z = bijector.inverse(z, flip=f)
        z = self.loft.inverse(z)
        return z

    def loss(self, x):
        z, log_pz, log_jacob = self.forward(x)

        return (-log_pz - log_jacob).mean()

    def sample(self, n):
        z = self.base_dist.rsample(sample_shape=(n,))
        return self.inverse(z)

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        # Calculate mean and std from the training data
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True) # Use unbiased=False for consistency

    def transform(self, data):
        # Apply the standardization
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        # For reconstructing original data, useful for predictions
        return (data * self.std) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def state_dict(self):
        return {
            'mean': self.mean,
            'std': self.std
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class DustLightningModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

    def condition(self, y):
        return self.model.condition(y)
    
    def loss(self, x, y):
        if hasattr(self.model, "condition"):
            loss = self.condition(y).loss(x)
        else:
            loss = self.model.loss(x, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() == 1:
            y = y.reshape(-1,1)
        loss = self.loss(x, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() == 1:
            y = y.reshape(-1,1)
        loss = self.loss(x, y)
        self.log('valid_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() == 1:
            y = y.reshape(-1,1)
        loss = self.loss(x, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        if hasattr(self.model, "condition"):
            return self.condition(y).sample(x.size(0))
        else:
            return self(x)