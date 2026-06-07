import urllib
import requests
from .sources import BlackbodyStar
from .grids import UniformSphericalGrid
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.spatial.transform import Rotation
import pandas as pd
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
import zuko

from .utils import GridStruct, random_direction
from .photons import PhotonList

from torch.distributions.multivariate_normal import MultivariateNormal

wp.config.quiet = True

default_fiducial_values = {"amax": 1.0*u.mm, "p": 3.5}

class Dust(pl.LightningDataModule):
    def __init__(self, lam=None, kabs=None, ksca=None, amax=None, p=None, abundances=(), device="cpu", ntemperatures=300, 
                 fiducial_values={}):
        """
        Initialize the Dust module with wavelength, absorption, and scattering coefficients.

        Parameters
        ----------
        lam : astropy.units.Quantity
            Wavelengths at which the dust opacities are defined. Should be in units that convert to cm, and with a shape of (nwavelengths,).
        kabs : astropy.units.Quantity
            Absorption coefficients of the dust. Should be in units that convert to cm^2/g, and with a shape of (nsamples, nwavelengths),
            where ndims is the number of dimensions in the parameter space (e.g., p, amax, abundances).
        ksca : astropy.units.Quantity
            Scattering coefficients of the dust. Should be in units that convert to cm^2/g, and with a shape of (nsamples, nwavelengths),
            where ndims is the number of dimensions in the parameter space (e.g., p, amax, abundances).
        amax : astropy.units.Quantity
            Maximum grain size for the size distribution. Should be in units that convert to cm, and with a shape of (nsamples,).
        p : float
            Power-law index for the grain size distribution. Should have a shape of (nsamples,)
        abundances : tuple of np.arrays
            Tuple of arrays containing the abundances of different dust species. Each array should have a shape of (nsamples,).
        device : str
            Device to run the computations on (e.g., "cpu" or "cuda").
        """
        super().__init__()

        kunit = kabs.unit
        lam_unit = lam.unit

        lam = lam.value
        kabs = kabs.value
        ksca = ksca.value

        if lam[1] > lam[0]:
            lam = np.flip(lam, axis=-1)
            kabs = np.flip(kabs, axis=-1)
            ksca = np.flip(ksca, axis=-1)

        self.nu = (const.c / (lam * lam_unit)).decompose().to(u.GHz)
        self.kmean = np.mean(kabs) * kunit
        self.lam = lam * lam_unit
        self.kabs = kabs / self.kmean.value
        self.ksca = ksca / self.kmean.value
        self.kext = (kabs + ksca) / self.kmean.value
        self.albedo = ksca / (kabs + ksca)

        self.log10_nu_min = np.log10(self.nu.value.min())
        self.log10_nu_max = np.log10(self.nu.value.max())

        self.amax = amax
        if amax is not None:
            self.log10_amax = np.log10(amax.to(u.cm).value)
        self.p = p
        self.abundances = abundances

        self.dims = ()
        self.samples = ()
        for dim in ["p", "log10_amax", "abundances"]:
            if hasattr(self, dim) and getattr(self, dim) is not None and getattr(self, dim) is not ():
                if dim in ["abundances"]:
                    self.samples += getattr(self, dim)
                else:
                    self.samples += (getattr(self, dim),)
                self.dims += (dim,)

        if len(self.dims) > 0:
            self.samples = np.vstack(self.samples).T
        else:
            self.samples = np.zeros((1,0))
        self.ndims = len(self.dims) + (len(self.abundances) - 1 if len(abundances) > 0 else 0)

        self.fiducial_values = fiducial_values
        for dim in self.dims:
            if dim.replace("log10_","") not in self.fiducial_values:
                if dim != "abundances":
                    self.fiducial_values[dim.replace("log10_","")] = default_fiducial_values[dim.replace("log10_","")]
                elif len(self.abundances) > 0:
                    self.fiducial_values["abundances"] = np.ones(len(self.abundances)) / (len(self.abundances) + 1.)

        self.temperature = np.logspace(-1.,4.,ntemperatures)
        self.log_temperature = np.log10(self.temperature)

    def __getstate__(self):
        state = self.__dict__.copy()
                
        for entry in state:
            if wp.types.is_array(getattr(self, entry)):
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
        for model in ["random_nu", "ml_step", "kabs", "ksca"]:
            if hasattr(self, f"{model}_model"):
                getattr(self, f"{model}_model").to(device)
            if hasattr(self, f"{model}_x_scaler"):
                getattr(self, f"{model}_x_scaler").to(device)
            if hasattr(self, f"{model}_y_scaler"):
                getattr(self, f"{model}_y_scaler").to(device)

    def ml_kabs(self, p=None, amax=None, nu=None, abundances=None, photon_list=None, iphotons=None):
        if photon_list is not None:
            p = wp.to_torch(photon_list.p)
            amax = wp.to_torch(photon_list.amax)
            if photon_list.dust_abundances is not None:
                abundances = wp.to_torch(photon_list.dust_abundances)

            if nu is None:
                nu = wp.to_torch(photon_list.frequency)

                if iphotons is not None:
                    nu = nu[iphotons]
                    p = p[iphotons]
                    amax = amax[iphotons]
                    if abundances is not None:
                        abundances = abundances[iphotons]
            else:
                if nu.size(0) != p.size(0):
                    p = p[iphotons]
                    amax = amax[iphotons]
                    if abundances is not None:
                        abundances = abundances[iphotons]

            abundances = tuple([abundances[:,i] for i in range(len(self.abundances))])

        if amax is not None:
            log10_amax = torch.log10(amax)

        samples = ()
        for dim in self.dims:
            if dim == "abundances" and abundances is not None:
                samples += abundances
            else:
                samples += (eval(dim),)
        samples += (torch.log10(nu),)
        samples = torch.transpose(torch.vstack(samples), 0, 1)

        kabs = 10.**self.kabs_y_scaler.inverse_transform(self.kabs_model(self.kabs_x_scaler.transform(samples))).detach().flatten()

        return kabs

    def ml_ksca(self, p=None, amax=None, nu=None, abundances=None, photon_list=None, iphotons=None):
        if photon_list is not None:
            p = wp.to_torch(photon_list.p)
            amax = wp.to_torch(photon_list.amax)
            if photon_list.dust_abundances is not None:
                abundances = wp.to_torch(photon_list.dust_abundances)
                
            if nu is None:
                nu = wp.to_torch(photon_list.frequency)

                if iphotons is not None:
                    nu = nu[iphotons]
                    p = p[iphotons]
                    amax = amax[iphotons]
                    if abundances is not None:
                        abundances = abundances[iphotons]
            else:
                if nu.size(0) != p.size(0):
                    p = p[iphotons]
                    amax = amax[iphotons]
                    if abundances is not None:
                        abundances = abundances[iphotons]

            abundances = tuple([abundances[:,i] for i in range(len(self.abundances))])

        if amax is not None:
            log10_amax = torch.log10(amax)

        samples = ()
        for dim in self.dims:
            if dim == "abundances" and abundances is not None:
                samples += abundances
            else:
                samples += (eval(dim),)
        samples += (torch.log10(nu),)
        samples = torch.transpose(torch.vstack(samples), 0, 1)

        ksca = 10.**self.ksca_y_scaler.inverse_transform(self.ksca_model(self.ksca_x_scaler.transform(samples))).detach().flatten()

        return ksca

    def ml_kext(self, p=None, amax=None, nu=None, abundances=None, photon_list=None, iphotons=None):
        if photon_list is not None:
            p = wp.to_torch(photon_list.p)
            amax = wp.to_torch(photon_list.amax)
            if photon_list.dust_abundances is not None:
                abundances = wp.to_torch(photon_list.dust_abundances)

            if nu is None:
                nu = wp.to_torch(photon_list.frequency)
            else:
                if nu.size(0) != p.size(0):
                    p = p[iphotons]
                    amax = amax[iphotons]
                    if abundances is not None:
                        abundances = abundances[iphotons]

            abundances = tuple([abundances[:,i] for i in range(len(self.abundances))])

        if amax is not None:
            log10_amax = torch.log10(amax)

        samples = ()
        for dim in self.dims:
            if dim == "abundances" and abundances is not None:
                samples += abundances
            else:
                samples += (eval(dim),)
        samples += (torch.log10(nu),)
        samples = torch.transpose(torch.vstack(samples), 0, 1)

        return 10.**self.kabs_y_scaler.inverse_transform(self.kabs_model(self.kabs_x_scaler.transform(samples))).detach().flatten() + \
                10.**self.ksca_y_scaler.inverse_transform(self.ksca_model(self.ksca_x_scaler.transform(samples))).detach().flatten()

    def absorb(self, temperature):
        nphotons = frequency.numpy().size

        cost = -1. + 2*np.random.rand(nphotons)
        sint = np.sqrt(1. - cost**2)
        phi = 2*np.pi*np.random.rand(nphotons)

        direction = np.array([sint*np.cos(phi), sint*np.sin(phi), cost]).T

        frequency = self.random_nu(temperature)

        return direction, frequency
    
    def update_photon_opacities(self, photon_list, iphotons, grid=None, inu=None):
        nphotons = iphotons.size(0)

        if grid is not None and inu is not None:
            wp.launch(kernel=self.set_photon_opacities_grid,
                      dim=(nphotons,),
                      inputs=[photon_list, grid, inu, iphotons])
        else:
            wp.launch(kernel=self.set_photon_opacities,
                      dim=(nphotons,),
                      inputs=[photon_list, 
                              self.ml_kabs(photon_list=photon_list, iphotons=iphotons), 
                              self.ml_ksca(photon_list=photon_list, iphotons=iphotons), 
                              iphotons])

    @wp.kernel
    def set_photon_opacities(photon_list: PhotonList,
                          kabs: wp.array(dtype=float),
                          ksca: wp.array(dtype=float),
                          iphotons: wp.array(dtype=int)): # pragma: no cover
        i = wp.tid()
        ip = iphotons[i]

        photon_list.kabs[ip] = kabs[i]
        photon_list.ksca[ip] = ksca[i]
        photon_list.albedo[ip] = ksca[i] / (kabs[i] + ksca[i])

    @wp.kernel
    def set_photon_opacities_grid(photon_list: PhotonList,
                                  grid: GridStruct,
                                  inu: int,
                                  iphotons: wp.array(dtype=int)): # pragma: no cover
        ip = iphotons[wp.tid()]

        ix, iy, iz = photon_list.indices[ip][0], photon_list.indices[ip][1], photon_list.indices[ip][2]

        photon_list.kabs[ip] = grid.kabs[inu, ix, iy, iz]
        photon_list.ksca[ip] = grid.ksca[inu, ix, iy, iz]
        photon_list.albedo[ip] = photon_list.ksca[ip] / (photon_list.kabs[ip] + photon_list.ksca[ip])

    def set_grid_opacities(self, grid, frequency):
        p = wp.to_torch(grid.p)
        shape = p.shape
        p = p.flatten()
        amax = wp.to_torch(grid.amax).flatten()
        abundances = tuple([wp.to_torch(grid.dust_abundances)[i].flatten() for i in range(len(self.abundances))])

        kabs = [self.ml_kabs(p=p, 
                             amax=amax, 
                             abundances=abundances, 
                             nu=torch.ones(np.prod(shape), dtype=torch.float32, 
                                           device=wp.device_to_torch(wp.get_device())) * \
                                            f.to(u.GHz).value) for f in frequency]

        grid.kabs = wp.from_torch(torch.concatenate(kabs).reshape((len(frequency),) + shape))

        ksca = [self.ml_ksca(p=p, 
                             amax=amax, 
                             abundances=abundances, 
                             nu=torch.ones(np.prod(shape), dtype=torch.float32, 
                                           device=wp.device_to_torch(wp.get_device())) * 
                                            f.to(u.GHz).value) for f in frequency]
        
        grid.ksca = wp.from_torch(torch.concatenate(ksca).reshape((len(frequency),) + shape))

    def random_nu_ml(self, p, amax, temperature, abundances=None):
        nphotons = temperature.size
        ksi = torch.rand(int(nphotons), dtype=torch.float32)
        ksi = torch.clamp(torch.arctanh(2*ksi - 1.), min=-8.6643, max=8.6643)

        log10_amax = np.log10(amax)

        samples = ()
        for dim in self.dims:
            if dim == "abundances" and abundances is not None:
                samples += tuple([torch.tensor(a, dtype=torch.float32) for a in abundances])
            else:
                samples += (torch.tensor(eval(dim), dtype=torch.float32),)
        samples += (torch.log10(torch.tensor(temperature, dtype=torch.float32)), ksi)

        samples = torch.transpose(torch.vstack(samples), 0, 1)
        test_x = self.random_nu_x_scaler.transform(samples)

        log10_nu = torch.clamp(self.random_nu_y_scaler.inverse_transform(self.random_nu_model(test_x).detach()), self.log10_nu_min, self.log10_nu_max)

        return 10.**log10_nu.numpy()

    def random_nu(self, photon_list, subset=None):
        p = wp.to_torch(photon_list.p)
        amax = wp.to_torch(photon_list.amax)
        temperature = wp.to_torch(photon_list.temperature)
        if photon_list.dust_abundances is not None:
                abundances = wp.to_torch(photon_list.dust_abundances)
        if subset is not None:
            p = p[subset]
            amax = amax[subset]
            temperature = temperature[subset]
            if photon_list.dust_abundances is not None:
                abundances = abundances[subset]

        abundances = tuple([abundances[:,i] for i in range(len(self.abundances))])
            
        nphotons = temperature.size(0)
        ksi = torch.rand(int(nphotons), device=wp.device_to_torch(wp.get_device()), dtype=torch.float32)
        ksi = torch.clamp(torch.arctanh(2*ksi - 1.), min=-8.6643, max=8.6643)

        if amax is not None:
            log10_amax = torch.log10(amax)

        samples = ()
        for dim in self.dims:
            if dim == "abundances" and abundances is not None:
                samples += abundances
            else:
                samples += (eval(dim),)
        samples += (torch.log10(temperature), ksi)

        samples = torch.transpose(torch.vstack(samples), 0, 1)

        test_x = self.random_nu_x_scaler.transform(samples)

        if nphotons > 250000:
            test_x = TensorDataset(test_x)
            loader = DataLoader(test_x, batch_size=250000)

            log10_nu = torch.cat([torch.clamp(self.random_nu_y_scaler.inverse_transform(self.random_nu_model(X).detach()), self.log10_nu_min, self.log10_nu_max) for X, in loader], 0)
        else:
            log10_nu = torch.clamp(self.random_nu_y_scaler.inverse_transform(self.random_nu_model(test_x).detach()), self.log10_nu_min, self.log10_nu_max)
        
        nu = wp.from_torch(10.**torch.flatten(log10_nu))

        return nu

    def ml_planck_mean_opacity(self, p, amax, temperature, abundances=()):
        log10_amax = torch.log10(amax)
        
        samples = ()
        for dim in self.dims:
            if dim == "abundances" and abundances is not None:
                samples += abundances
            else:
                samples += (eval(dim),)
        samples += (torch.log10(temperature),)

        samples = torch.transpose(torch.vstack(samples), 0, 1)

        return 10.**self.pmo_y_scaler.inverse_transform(self.pmo_model(self.pmo_x_scaler.transform(samples))).detach().flatten()

    def ml_step(self, photon_list, s, iphotons):
        nphotons = iphotons.size(0)

        test_y = torch.transpose(torch.vstack((torch.log10(wp.to_torch(photon_list.frequency)[iphotons]),
                              torch.log10(wp.to_torch(photon_list.temperature)[iphotons]),
                              torch.log10(wp.to_torch(photon_list.amax)[iphotons]),
                              torch.log10(wp.to_torch(photon_list.p)[iphotons]),
                              torch.log10(wp.to_torch(photon_list.density)[iphotons] * wp.to_torch(photon_list.kabs)[iphotons] * s[iphotons]),
                              )), 0, 1)

        test_x = self.ml_step_model.condition(self.ml_step_y_scaler.transform(test_y)).sample().detach()
        test_x = self.ml_step_x_scaler.inverse_transform(test_x)

        return 10.**torch.clamp(test_x[:,0], self.log10_nu0_min, self.log10_nu0_max), 10.**test_x[:,1], 10.**test_x[:,2], test_x[:,3], test_x[:,4], torch.zeros(test_x.size(0)), test_x[:,5], test_x[:,6], torch.zeros(test_x.size(0))

    def initialize_model(self, model="random_nu", model_type="MLP", input_size=2, output_size=1, hidden_units=(48, 48, 48)):
        if model_type == 'flow':
            setattr(self, f"{model}_model", NeuralSplineFlow(input_size, output_size, transforms=len(hidden_units), hidden_features=hidden_units[0]))
        else:
            setattr(self, f"{model}_model", MultiLayerPerceptron(input_size, output_size, hidden_units=hidden_units))

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

        # Reset the batch_size
        if hasattr(self, "batch_size"):
            del self.batch_size

        # Set up the NN

        if model == "random_nu":
            self.input_size, self.output_size = self.ndims + 2, 1
            self.model_type = "MLP"
        elif model == "ml_step":
            self.input_size, self.output_size = 7, self.ndims + 3
            self.model_type = "flow"

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
        elif model in ["kabs", "ksca", "g", "pmo"]:
            self.input_size, self.output_size = self.ndims + 1, 1
            self.model_type = "MLP"

        self.initialize_model(model=model, model_type=self.model_type, input_size=self.input_size, output_size=self.output_size, hidden_units=hidden_units)

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
            elif self.current_model in ["kabs", "ksca", "g", "pmo"]:
                self.plot_opacity_model(model=self.current_model)
            elif self.current_model == "scattering_phase_function":
                self.plot_scattering_phase_function_model()
            else:
                #self.plot_ml_step()
                self.plot_triangle_plots(model=self.current_model)

    def prepare_data(self):
        if hasattr(self, "dataset") and not self.overwrite:
            return
        
        if hasattr(self, f"prepare_data_{self.current_model}"):
            if self.current_model in ["kabs", "ksca"]:
                samples, targets = self.prepare_data_opacity(model=self.current_model)
            else:
                samples, targets = getattr(self, f"prepare_data_{self.current_model}")()
        else:
            raise NotImplementedError(f"Data preparation for model {self.current_model} not implemented.")

        self.nsamples = samples.shape[0]

        X = torch.tensor(samples, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)
        
        X_scaler = StandardScaler()
        X_scaler.fit(X)
        X = X_scaler.transform(X)

        y_scaler = StandardScaler()
        y_scaler.fit(y)
        y = y_scaler.transform(y)

        setattr(self, f"{self.current_model}_x_scaler", X_scaler)
        setattr(self, f"{self.current_model}_y_scaler", y_scaler)

        self.dataset = TensorDataset(X, y)

    def prepare_data_random_nu(self):
        count = 0
        batch_size = 100

        total_samples = []
        total_targets = []
        total_original_indices = []

        while count < self.samples.shape[0]:
            n = min(batch_size, self.samples.shape[0] - count)
            
            samples = np.repeat(np.expand_dims(np.repeat(np.expand_dims(self.samples[count:count+n,:], 1), self.temperature.size, axis=1), 1), self.nu.size, 1)
            original_indices = np.tile(np.expand_dims(np.arange(self.samples.shape[0])[count:count+n], axis=(-1, -2)), (1, self.temperature.size, self.nu.size)).flatten()
    
            temperature = np.repeat(np.expand_dims(np.repeat(np.expand_dims(self.temperature*u.K, (0, -1)), self.nu.size, axis=0), 0), max(samples.shape[0], 1), axis=0)
            nu = np.repeat(np.repeat(np.expand_dims(self.nu, (0, -1, -2)), self.temperature.size, axis=2), max(samples.shape[0], 1), axis=0)
            
            ksi = scipy.integrate.cumulative_trapezoid(np.repeat(np.expand_dims(self.kabs[count:count+n,:], (-1, -2)), self.temperature.size, axis=-2) * models.BlackBody(temperature)(nu), np.log10(self.nu.to(u.GHz).value), axis=1, initial=0)
            ksi /= ksi[:,-1:,:,:]
            
            samples = np.concat((samples, np.log10(temperature.to(u.K).value), ksi), axis=-1)
            samples = samples.reshape((-1, samples.shape[-1]))
            
            targets = np.log10(nu.to(u.GHz).value.flatten())
    
            samples = samples.astype(np.float32)
    
            good = np.logical_not(np.logical_or(samples[:,-1] < 2e-8, samples[:,-1] == 1))
    
            samples, targets = samples[good,:], targets[good]
            original_indices = original_indices[good]
    
            samples[:,-1] = 2*samples[:,-1] - 1
            samples[:,-1] = np.arctanh(samples[:,-1])

            total_samples.append(samples)
            total_targets.append(targets)
            total_original_indices.append(original_indices)

            count += n

        samples = np.concatenate(total_samples, axis=0)
        targets = np.concatenate(total_targets, axis=0)
        self.original_indices = np.concatenate(total_original_indices, axis=0)

        return samples, targets

    def prepare_data_opacity(self, model="kabs"):  
        samples = np.moveaxis(np.repeat(np.expand_dims(self.samples, 1), self.nu.size, axis=1), -1, 0)
        self.original_indices = np.repeat(np.expand_dims(np.arange(self.samples.shape[0]), axis=-1), self.nu.size, axis=-1).flatten()

        samples = np.concat((samples, np.expand_dims(np.repeat(np.expand_dims(np.log10(self.nu.to(u.GHz).value), axis=0), max(samples.shape[1], 1), axis=0), axis=0)), axis=0)
        samples = samples.reshape((samples.shape[0], -1)).T

        targets = np.log10(getattr(self, model).flatten())

        return samples, targets

    def prepare_data_kabs(self):
        return self.prepare_data_opacity(model="kabs")

    def prepare_data_ksca(self):
        return self.prepare_data_opacity(model="ksca")

    def prepare_data_pmo(self):
        temperature = np.repeat(np.expand_dims(self.temperature, (0, -1)), self.nu.size, axis=-1)
        nu = np.repeat(np.expand_dims(self.nu, (0, 1)), self.temperature.size, axis=1)

        self.pmo = self.kmean.cgs.value * scipy.integrate.trapezoid(models.BlackBody(temperature*u.K)(nu).cgs.value * np.expand_dims(self.kabs, 1), self.nu.to(u.Hz).value, axis=-1) * np.pi / (const.sigma_sb.cgs.value * self.temperature**4)

        samples = np.moveaxis(np.repeat(np.expand_dims(self.samples, 1), self.temperature.size, axis=1), -1, 0)
        self.original_indices = np.repeat(np.expand_dims(np.arange(self.samples.shape[0]), axis=-1), self.temperature.size, axis=-1).flatten()

        samples = np.concat((samples, np.expand_dims(np.repeat(np.expand_dims(np.log10(self.temperature), axis=0), max(samples.shape[1], 1), axis=0), axis=0)), axis=0)
        samples = samples.reshape((samples.shape[0], -1)).T

        targets = np.log10(self.pmo).flatten()

        return samples, targets

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
        targets = ["log10_nu0", "log10_T"] + \
                  (["log10_amax"] if "log10_amax" in self.dims else []) + \
                  (["p"] if "p" in self.dims else []) + \
                  ([f"abundance{i}" for i in range(len(self.abundances))]) + \
                  ["log10_tau_cell_nu0"]

        df.loc[:, "log10_tau"] = np.where(np.logical_or(df["log10_tau"] < -6.5, np.isnan(df["log10_tau"].values)), np.log10(-np.log(1. - np.random.rand(len(df)))), df["log10_tau"])

        data = df.loc[:, targets+features].values

        samples = df.loc[:, features].values
        targets = df.loc[:, targets].values
        
        self.ml_step_features = features
        self.ml_step_limits = {}
        for key in features:
            self.ml_step_limits[key] = (df[key].min(), df[key].max())

        self.df = df
        self.nsamples = len(df)

        return samples, targets

    def run_dust_simulation(self, nphotons=1000, tau_range=(3.0, 1e4), temperature_range=(0.1*u.K, 1e4*u.K), 
                            amax_range=(1*u.micron, 10.0*u.cm), p_range=(2.5, 4.5), nu_range=None, use_ml_step=False, 
                            position=0, time_limit=np.inf, device="cpu"):
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

        grid = UniformSphericalGrid(ncells=1, dr=1.0*u.au, mirror=False, device=device)

        density = np.ones(grid.shape) * 1e-16 * u.g / u.cm**3

        grid.set_physical_properties(density=density, amax=1.0*u.micron, p=3.5, dust=self)
        grid.add_sources(star)

        # Emit the photons

        photon_list = grid.emit(nphotons, wavelength="random", scattering=False)

        with wp.ScopedDevice(grid.device):
            initial_direction = np.zeros((nphotons, 3), dtype=np.float32)
            initial_direction[:,0] = 1.
            photon_list.direction = wp.array(initial_direction, dtype=wp.vec3)
    
            photon_list.frequency = wp.array(10.**np.random.uniform(np.log10(nu_range[0].value), np.log10(nu_range[1].value), nphotons), dtype=float)
            original_frequency = photon_list.frequency.numpy().copy()
    
            photon_list.temperature = wp.array(10.**np.random.uniform(np.log10(temperature_range[0].to(u.K).value), np.log10(temperature_range[1].to(u.K).value), nphotons), dtype=float)
    
            samples = suggest_opacity_sampling(nphotons, p_range=p_range, amax_range=amax_range, n_dust_subspecies=len(self.abundances)+1, mode="random")
    
            photon_list.amax = wp.array(samples[:,1], dtype=float)
            photon_list.p = wp.array(samples[:,0], dtype=float)
            if len(self.abundances) > 0:
                photon_list.dust_abundances = wp.array2d(samples[:,2:], dtype=float)
    
            tau = 10.**np.random.uniform(np.log10(tau_range[0]), np.log10(tau_range[1]), nphotons)
            photon_list.density = wp.array((tau / (self.kmean * self.ml_kabs(photon_list=photon_list) * \
                                                                                1.*u.au) * self.kmean).to(1 / u.au), dtype=float)

        grid.propagate_photons(photon_list, learning=True, use_ml_step=use_ml_step, time_limit=time_limit)

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

        for i in range(len(self.abundances)):
            df[f"abundance{i}"] = photon_list.dust_abundances.numpy()[:,i]

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

    def plot_opacity_model(self, model='kabs', show_scipy_interpolation=False):
        """
        Plot the learned opacity model against the interpolated opacity.

        Parameters
        ----------
        model : str
            The model to plot. Either "kabs" or "ksca".
        """
        import matplotlib.pyplot as plt

        if hasattr(self, "test_indices") and len(self.test_indices) > 0:
            index = np.random.choice(self.test_indices, size=1)[0]
        else:
            index = np.random.randint(0, self.samples.shape[0], 1)[0]

        if model in ["kabs", "ksca", "g"]:
            nx = self.nu.size
        elif model in ["pmo"]:
            nx = self.temperature.size

        if "log10_amax" in self.dims:
            log10_amax = np.repeat(np.log10(self.amax[index].to(u.cm).value), nx)
        if "p" in self.dims:
            p = np.repeat(self.p[index], nx)
        if "abundances" in self.dims:
            abundances = tuple([np.repeat(a[index], nx) for a in self.abundances])
        
        log10_lam = np.log10(self.lam.value)
        log10_nu = np.log10(self.nu.to(u.GHz).value)
        log10_temperature = np.log10(self.temperature)

        if model == "g":
            interpolated = getattr(self, model)[index,:]
        else:
            interpolated = np.log10(getattr(self, model)[index,:])

        print_str = ""
        for dim in self.dims:
            if dim == "abundances":
                print_str += f"{dim}: {[abundances[i][0] for i in range(len(abundances))]}, "
            else:
                print_str += f"{dim}: {locals()[dim][0]}, "
        print(print_str)

        samples = ()
        for dim in self.dims:
            if dim == "abundances":
                samples += abundances
            else:
                samples += (locals()[dim],)
        
        if model in ["kabs", "ksca", "g"]:
            samples = np.vstack(samples + (log10_nu,)).T
            plot_x = log10_lam
        else:
            samples = np.vstack(samples + (log10_temperature,)).T
            plot_x = log10_temperature

        nned = getattr(self, f'{model}_y_scaler').inverse_transform(getattr(self, f'{model}_model')(getattr(self, f'{model}_x_scaler').transform(torch.tensor(samples, dtype=torch.float32)))).detach().numpy()

        plt.plot(plot_x, interpolated)
        plt.plot(plot_x, nned)

        if show_scipy_interpolation:
            X = getattr(self, f'{model}_x_scaler').inverse_transform(self.train.dataset.tensors[0][self.train.indices,:]).numpy()
            y = getattr(self, f'{model}_y_scaler').inverse_transform(self.train.dataset.tensors[1][self.train.indices]).numpy()

            distance = np.sqrt((X[:,0] - self.p[index])**2 + (X[:,1] - np.log10(self.amax[index].to(u.cm).value))**2)
            good = distance <= np.unique(np.sort(distance))[50]
            
            scipy_interpolated = scipy.interpolate.griddata(X[good,:], y[good], samples)

            plt.plot(plot_x, scipy_interpolated)
        
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
        abundances = tuple([np.repeat(np.random.uniform(0, 1, 1), nsamples) for i in range(len(self.abundances))])
        print(f"p: {p[0]}, amax: {amax[0]}, T: {T[0]}, abundances: {[abundances[i][0] for i in range(len(self.abundances))]}")

        nu2 = self.random_nu_ml(p, amax, T, abundances=abundances)

        counts, bins, _ = plt.hist(np.log10(nu2), 300)

        bb = models.BlackBody(temperature=T[0] * u.K)
        pdf = bb(10.**bins[1:]*u.GHz).value * self.ml_kabs(torch.tensor(p[0], dtype=torch.float32).repeat((bins[1:].size,)), 
                                                           torch.tensor(amax[0], dtype=torch.float32).repeat((bins[1:].size,)), 
                                                           torch.tensor(10.**bins[1:], dtype=torch.float32), 
                                                           abundances=tuple([torch.tensor(a[0], dtype=torch.float32).repeat((bins[1:].size,)) for a in abundances])).numpy()
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
            if model in ["ml_step"]:
                X_pred = self.trainer.predict(self.dustLM, datamodule=self)
                X_pred = torch.cat(X_pred)
                y_pred = torch.cat([batch[1] for batch in self.predict_dataloader()])
            elif model in ["random_nu", "random_direction"]:
                y_pred = self.trainer.predict(self.dustLM, datamodule=self)
                y_pred = torch.cat(y_pred)
                X_pred = torch.cat([batch[0] for batch in self.predict_dataloader()])

            X_true = torch.cat([batch[0] for batch in self.predict_dataloader()])
            y_true = torch.cat([batch[1] for batch in self.predict_dataloader()])

            predict = True

        if model == "ml_step":
            features = np.array(["log10_nu", "log10_Eabs", "log10_tau", "yaw", "pitch", "direction_yaw", "direction_pitch"])
            targets = ["log10_nu0", "log10_T"] + \
                       (["log10_amax"] if "log10_amax" in self.dims else []) + \
                       (["p"] if "p" in self.dims else []) + \
                       ([f"abundance{i}" for i in range(len(self.abundances))]) + \
                       ["log10_tau_cell_nu0"]
        elif model == "random_nu":
            features = ()
            for dim in self.dims:
                if dim == "abundances":
                    features += tuple([f"abundance_{i}" for i in range(len(self.abundances))])
                else:
                    features += (dim,)
            features = np.array(features + ("log10_temperature", "ksi"))
            targets = np.array(["log10_nu"])

            y_true = torch.unsqueeze(y_true, 1)
        elif model == "random_direction":
            features = ()
            for dim in self.dims:
                if dim == "abundances":
                    features += tuple([f"abundance_{i}" for i in range(len(self.abundances))])
                else:
                    features += (dim,)
            features = np.array(features + ("log10_nu", "ksi"))
            targets = np.array(["theta"])

            y_true = torch.unsqueeze(y_true, 1)

        if isinstance(plot_columns, str) and plot_columns == 'all':
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
        
        if self.learning == "ml_step" or self.ndims == 0:
            test_size = int(self.test_split * self.nsamples)
            valid_size = int((self.nsamples - test_size)*self.valid_split)
            train_size = self.nsamples - test_size - valid_size
            train_val_tmp, self.test = random_split(self.dataset, [train_size + valid_size, test_size], generator=torch.Generator().manual_seed(1))
            self.train, self.valid = random_split(train_val_tmp, [train_size, valid_size], generator=torch.Generator().manual_seed(2))
        else:
            train_indices, valid_indices, test_indices = torch.utils.data.random_split(range(self.samples.shape[0]), 
                                                                                   (1.-(self.test_split + self.valid_split), 
                                                                                    self.valid_split, self.test_split), 
                                                                                   generator=torch.Generator().manual_seed(2))
            self.test_indices = test_indices.indices

            splits = np.repeat(0, self.dataset.tensors[0].size(0))
            for ind in valid_indices.indices:
                splits[self.original_indices == ind] = 1
            for ind in test_indices.indices:
                splits[self.original_indices == ind] = 2

            train_indices = np.where(splits.flatten() == 0)[0]
            valid_indices = np.where(splits.flatten() == 1)[0]
            test_indices = np.where(splits.flatten() == 2)[0]

            self.train = torch.utils.data.Subset(self.dataset, train_indices)
            self.valid = torch.utils.data.Subset(self.dataset, valid_indices)
            self.test = torch.utils.data.Subset(self.dataset, test_indices)

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
                "abundances": self.abundances,
                "kabs": self.kabs*self.kmean,
                "ksca": self.ksca*self.kmean,
                "fiducial_values": self.fiducial_values,
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

class IsotropicDust(Dust):
    def scatter(self, photon_list, iphotons):
        nphotons = iphotons.size(0)

        wp.launch(kernel=random_direction,
                  dim=(nphotons,),
                  inputs=[photon_list.direction, iphotons, np.random.randint(0, 100000)])

    def update_photon_scattering_phase_function(self, photon_list, direction, iphotons):
        nphotons = iphotons.size(0)

        wp.launch(kernel=self.scattering_phase_function_wp,
                  dim=(nphotons,),
                  inputs=[photon_list, direction, iphotons])

    @wp.kernel
    def scattering_phase_function_wp(photon_list: PhotonList,
                                     direction: wp.vec3, 
                                     iphotons: wp.array(dtype=int)): # pragma: no cover
        i = wp.tid()
        ip = iphotons[i]

        photon_list.scattering_phase_function[ip] = 1.


class HenyeyGreensteinDust(Dust):
    def __init__(self, lam=None, kabs=None, ksca=None, g=None, amax=None, p=None, abundances=(), device="cpu", ntemperatures=1000):
        """
        Initialize the Henyey-Greenstein dust model.

        Parameters
        ----------
        lam : numpy.ndarray
            The wavelengths to use for the dust properties.
        kabs : numpy.ndarray
            The absorption opacities to use for the dust properties.
        ksca : numpy.ndarray
            The scattering opacities to use for the dust properties.
        g : numpy.ndarray
            The Henyey-Greenstein asymmetry parameter to use for the dust properties.
        amax : float
            The maximum grain size to use for the dust properties.
        p : float
            The power-law index for the grain size distribution.
        device : str
            The device to place the dust properties on ("cpu" or "cuda").
        """
        super().__init__(lam=lam, kabs=kabs, ksca=ksca, amax=amax, p=p, abundances=abundances, device=device, 
                         ntemperatures=ntemperatures)

        self.g = g

    def to_device(self, device):
        super().to_device(device)

        for model in ["g"]:
            if hasattr(self, f"{model}_model"):
                getattr(self, f"{model}_model").to(device)
            if hasattr(self, f"{model}_x_scaler"):
                getattr(self, f"{model}_x_scaler").to(device)
            if hasattr(self, f"{model}_y_scaler"):
                getattr(self, f"{model}_y_scaler").to(device)

    def scatter(self, photon_list, iphotons):
        nphotons = iphotons.size(0)

        wp.launch(kernel=self.random_direction,
                  dim=(nphotons,),
                  inputs=[photon_list.direction, photon_list.g, iphotons, np.random.randint(0, 100000)])

    @wp.kernel
    def random_direction(direction: wp.array(dtype=wp.vec3),
                         g: wp.array(dtype=float),
                         iphotons: wp.array(dtype=int),
                         seed: int): # pragma: no cover
        i = wp.tid()
        ip = iphotons[i]

        rng = wp.rand_init(seed, i)

        cost = (1. + g[ip]**2. - ((1. - g[ip]**2.)/(1. - g[ip] + 2.*g[ip]*wp.randf(rng)))**2.) / (2. * g[ip])
        theta = wp.acos(cost)
        phi = 2.*np.pi*wp.randf(rng)

        rpy_quat1 = wp.quat_rpy(phi, 0., 0.)
        rpy_quat2 = wp.quat_rpy(0., theta, 0.)
        direction_quat = wp.quat_between_vectors(wp.vec3(1., 0., 0.), direction[ip])
        total_quat = direction_quat * rpy_quat1 * rpy_quat2

        direction[ip] = wp.quat_rotate(total_quat, wp.vec3(1., 0., 0.))

    def update_photon_scattering_phase_function(self, photon_list, direction, iphotons):
        nphotons = iphotons.size(0)

        wp.launch(kernel=self.scattering_phase_function_heneygreenstein_wp,
                  dim=(nphotons,),
                  inputs=[photon_list, direction, iphotons])

    @wp.kernel
    def scattering_phase_function_heneygreenstein_wp(photon_list: PhotonList,
                                                     direction: wp.vec3, 
                                                     iphotons: wp.array(dtype=int)): # pragma: no cover
        i = wp.tid()
        ip = iphotons[i]

        mu = wp.dot(photon_list.direction[ip], direction)

        photon_list.scattering_phase_function[ip] = (1. - photon_list.g[ip]**2.) / (1. + photon_list.g[ip]**2. - 2. * photon_list.g[ip] * mu)

    def ml_g(self, p=None, amax=None, nu=None, abundances=None, photon_list=None, iphotons=None):
        if photon_list is not None:
            p = wp.to_torch(photon_list.p)
            amax = wp.to_torch(photon_list.amax)
            if photon_list.dust_abundances is not None:
                abundances = wp.to_torch(photon_list.dust_abundances)

            if nu is None:
                nu = wp.to_torch(photon_list.frequency)

                if iphotons is not None:
                    nu = nu[iphotons]
                    p = p[iphotons]
                    amax = amax[iphotons]
                    if abundances is not None:
                        abundances = abundances[iphotons]
            else:
                if nu.size(0) != p.size(0):
                    p = p[iphotons]
                    amax = amax[iphotons]
                    if abundances is not None:
                        abundances = abundances[iphotons]

            abundances = tuple([abundances[:,i] for i in range(len(self.abundances))])

        if amax is not None:
            log10_amax = torch.log10(amax)

        samples = ()
        for dim in self.dims:
            if dim == "abundances" and abundances is not None:
                samples += abundances
            else:
                samples += (eval(dim),)
        samples += (torch.log10(nu),)
        samples = torch.transpose(torch.vstack(samples), 0, 1)

        g = self.g_y_scaler.inverse_transform(self.g_model(self.g_x_scaler.transform(samples))).detach().flatten()

        return g

    def prepare_data_g(self):
        return self.prepare_data_opacity(model="g")

    def update_photon_opacities(self, photon_list, iphotons, grid=None, inu=None):
        nphotons = iphotons.size(0)

        if grid is not None and inu is not None:
            wp.launch(kernel=self.set_photon_opacities_grid,
                      dim=(nphotons,),
                      inputs=[photon_list, grid, inu, iphotons])
        else:
            wp.launch(kernel=self.set_photon_opacities,
                      dim=(nphotons,),
                      inputs=[photon_list, 
                              self.ml_kabs(photon_list=photon_list, iphotons=iphotons), 
                              self.ml_ksca(photon_list=photon_list, iphotons=iphotons), 
                              self.ml_g(photon_list=photon_list, iphotons=iphotons),
                              iphotons])

    @wp.kernel
    def set_photon_opacities(photon_list: PhotonList,
                          kabs: wp.array(dtype=float),
                          ksca: wp.array(dtype=float),
                          g: wp.array(dtype=float), 
                          iphotons: wp.array(dtype=int)): # pragma: no cover
        i = wp.tid()
        ip = iphotons[i]

        photon_list.kabs[ip] = kabs[i]
        photon_list.ksca[ip] = ksca[i]
        photon_list.g[ip] = g[i]
        photon_list.albedo[ip] = ksca[i] / (kabs[i] + ksca[i])

    @wp.kernel
    def set_photon_opacities_grid(photon_list: PhotonList,
                                  grid: GridStruct,
                                  inu: int,
                                  iphotons: wp.array(dtype=int)): # pragma: no cover
        ip = iphotons[wp.tid()]

        ix, iy, iz = photon_list.indices[ip][0], photon_list.indices[ip][1], photon_list.indices[ip][2]

        photon_list.kabs[ip] = grid.kabs[inu, ix, iy, iz]
        photon_list.ksca[ip] = grid.ksca[inu, ix, iy, iz]
        photon_list.g[ip] = grid.g[inu, ix, iy, iz]
        photon_list.albedo[ip] = photon_list.ksca[ip] / (photon_list.kabs[ip] + photon_list.ksca[ip])

    def set_grid_opacities(self, grid, frequency):
        super().set_grid_opacities(grid, frequency)

        p = wp.to_torch(grid.p)
        shape = p.shape
        p = p.flatten()
        amax = wp.to_torch(grid.amax).flatten()
        abundances = tuple([wp.to_torch(grid.dust_abundances)[i].flatten() for i in range(len(self.abundances))])

        g = [self.ml_g(p=p, 
                       amax=amax, 
                       abundances=abundances,
                       nu=torch.ones(np.prod(shape), dtype=torch.float32, 
                                     device=wp.device_to_torch(wp.get_device())) * \
                                        f.to(u.GHz).value) for f in frequency]

        grid.g = wp.from_torch(torch.concatenate(g).reshape((len(frequency),) + shape))

    def state_dict(self):
        state_dict = super().state_dict()

        state_dict["dust_properties"]["g"] = self.g

        if hasattr(self, "g_model"):
            state_dict["g_state_dict"] = self.g_model.state_dict()
            state_dict["g_x_scaler"] = self.g_x_scaler.state_dict()
            state_dict["g_y_scaler"] = self.g_y_scaler.state_dict()

        return state_dict

class GeneralDust(Dust):
    def __init__(self, lam=None, kabs=None, ksca=None, scattering_phase_function=None, theta=None, amax=None, 
                 p=None, abundances=(), device="cpu", ntemperatures=1000):
        """
        Initialize the General dust model.

        Parameters
        ----------
        lam : numpy.ndarray
            The wavelengths to use for the dust properties.
        kabs : numpy.ndarray
            The absorption opacities to use for the dust properties.
        ksca : numpy.ndarray
            The scattering opacities to use for the dust properties.
        scattering_phase_function : numpy.ndarray
            The scattering phase function to use for the dust properties, as a function of theta.
        theta : numpy.ndarray
            The angles at which the scattering phase function is tabulated.
        amax : float
            The maximum grain size to use for the dust properties.
        p : float
            The power-law index for the grain size distribution.
        device : str
            The device to place the dust properties on ("cpu" or "cuda").
        """
        super().__init__(lam=lam, kabs=kabs, ksca=ksca, amax=amax, p=p, abundances=abundances, device=device, 
                         ntemperatures=ntemperatures)

        # Ensure that the scattering phase function is normalized for each combination of p, amax, and nu.

        scattering_phase_function = -1. * scattering_phase_function / \
                                    np.expand_dims(scipy.integrate.trapezoid(scattering_phase_function, 
                                                                             np.cos(theta), axis=-1), axis=-1)

        self.scattering_phase_function = scattering_phase_function
        self.theta = theta

    def to_device(self, device):
        super().to_device(device)

    def scatter(self, photon_list, iphotons):
        nphotons = iphotons.size(0)

        p = wp.to_torch(photon_list.p)
        amax = wp.to_torch(photon_list.amax)
        frequency = wp.to_torch(photon_list.frequency)
        if photon_list.dust_abundances is not None:
                abundances = wp.to_torch(photon_list.dust_abundances)
        if iphotons is not None:
            p = p[iphotons]
            amax = amax[iphotons]
            frequency = frequency[iphotons]
            if photon_list.dust_abundances is not None:
                abundances = abundances[iphotons]

        abundances = tuple([abundances[:,i] for i in range(len(self.abundances))])
            
        nphotons = iphotons.size(0)
        ksi = torch.rand(int(nphotons), device=wp.device_to_torch(wp.get_device()), dtype=torch.float32)
        ksi = torch.clamp(torch.arctanh(2*ksi - 1.), min=-8.6643, max=8.6643)

        if amax is not None:
            log10_amax = torch.log10(amax)

        samples = ()
        for dim in self.dims:
            if dim == "abundances" and abundances is not None:
                samples += abundances
            else:
                samples += (eval(dim),)
        samples += (torch.log10(frequency), ksi)

        samples = torch.transpose(torch.vstack(samples), 0, 1)

        test_x = self.random_direction_x_scaler.transform(samples)

        theta = self.random_direction_y_scaler.inverse_transform(self.random_direction_model(test_x).detach()).flatten()

        wp.launch(kernel=self.random_direction,
                  dim=(nphotons,),
                  inputs=[photon_list.direction,
                          wp.from_torch(theta),
                          iphotons, 
                          np.random.randint(0, 100000)])
    
    @wp.kernel
    def random_direction(direction: wp.array(dtype=wp.vec3),
                         theta: wp.array(dtype=float),
                         iphotons: wp.array(dtype=int),
                         seed: int): # pragma: no cover
        i = wp.tid()
        ip = iphotons[i]

        rng = wp.rand_init(seed, i)

        phi = 2.*np.pi*wp.randf(rng)

        rpy_quat1 = wp.quat_rpy(phi, 0., 0.)
        rpy_quat2 = wp.quat_rpy(0., theta[i], 0.)
        direction_quat = wp.quat_between_vectors(wp.vec3(1., 0., 0.), direction[ip])
        total_quat = direction_quat * rpy_quat1 * rpy_quat2

        direction[ip] = wp.quat_rotate(total_quat, wp.vec3(1., 0., 0.))

    def update_photon_scattering_phase_function(self, photon_list, direction, iphotons):
        nphotons = iphotons.size(0)

        theta = torch.acos((wp.to_torch(photon_list.direction)[iphotons] * torch.tensor(wp.array(direction))).sum(axis=1))

        scattering_phase_function = self.ml_scattering_phase_function(photon_list=photon_list, iphotons=iphotons, theta=theta)

        wp.launch(kernel=self.scattering_phase_function_general_dust_wp,
                  dim=(nphotons,),
                  inputs=[photon_list,
                          wp.from_torch(scattering_phase_function),
                          iphotons])

    @wp.kernel
    def scattering_phase_function_general_dust_wp(photon_list: PhotonList,
                                                  scattering_phase_function: wp.array(dtype=float),
                                                  iphotons: wp.array(dtype=int)): # pragma: no cover

        i = wp.tid()
        ip = iphotons[i]

        photon_list.scattering_phase_function[ip] = 2. * scattering_phase_function[i]

    def ml_scattering_phase_function(self, p=None, amax=None, nu=None, theta=None, photon_list=None, iphotons=None):
        if photon_list is not None:
            p = wp.to_torch(photon_list.p)
            amax = wp.to_torch(photon_list.amax)
            if photon_list.dust_abundances is not None:
                abundances = wp.to_torch(photon_list.dust_abundances)

            if nu is None:
                nu = wp.to_torch(photon_list.frequency)

                if iphotons is not None:
                    nu = nu[iphotons]
                    p = p[iphotons]
                    amax = amax[iphotons]
                    if abundances is not None:
                        abundances = abundances[iphotons]
            else:
                if nu.size(0) != p.size(0):
                    p = p[iphotons]
                    amax = amax[iphotons]
                    if abundances is not None:
                        abundances = abundances[iphotons]

            abundances = tuple([abundances[:,i] for i in range(len(self.abundances))])

        if amax is not None:
            log10_amax = torch.log10(amax)

        samples = ()
        for dim in self.dims:
            if dim == "abundances" and abundances is not None:
                samples += abundances
            else:
                samples += (eval(dim),)
        samples += (torch.log10(nu), theta)
        samples = torch.transpose(torch.vstack(samples), 0, 1)

        scattering_phase_function = 10.**self.scattering_phase_function_y_scaler.inverse_transform(
            self.scattering_phase_function_model(self.scattering_phase_function_x_scaler.transform(samples))).\
                detach().flatten()

        return scattering_phase_function

    def learn(self, model="random_nu", **kwargs):
        if model == "scattering_phase_function":
            self.input_size, self.output_size = self.ndims + 2, 1
            self.model_type = "MLP"
        elif model == "random_direction":
            self.input_size, self.output_size = self.ndims + 2, 1
            self.model_type = "MLP"

        super().learn(model=model, **kwargs)

    def prepare_data_scattering_phase_function(self):
        samples = np.repeat(np.repeat(np.expand_dims(np.moveaxis(self.samples, 0, 1), (-1, -2)), self.lam.size, axis=-2), self.theta.size, axis=-1)
        self.original_indices = np.tile(np.expand_dims(np.arange(self.samples.shape[0]), axis=(-1, -2)), (1, self.nu.size, self.theta.size)).flatten()

        samples = np.concat((samples, 
                             np.tile(np.expand_dims(np.log10(self.nu.to(u.GHz).value), 
                                                    axis=(0,1,-1)), 
                                                    (1, samples.shape[1], 1, self.theta.size)),
                             np.tile(np.expand_dims(self.theta.to(u.radian).value, 
                                                    axis=(0,1,2)), 
                                                    (1, samples.shape[1], self.lam.size, 1))), axis=0)
        samples = samples.reshape((samples.shape[0], -1)).T

        targets = np.log10(self.scattering_phase_function.flatten())

        return samples, targets

    def prepare_data_random_direction(self):
        ksi = -1. * scipy.integrate.cumulative_trapezoid(self.scattering_phase_function, np.cos(self.theta), initial=0, axis=-1)

        samples = np.repeat(np.repeat(np.expand_dims(np.moveaxis(self.samples, 0, 1), (-1, -2)), self.lam.size, axis=-2), self.theta.size, axis=-1)
        self.original_indices = np.tile(np.expand_dims(np.arange(self.samples.shape[0]), axis=(-1, -2)), (1, self.nu.size, self.theta.size)).flatten()
        
        samples = np.concat((samples, 
                             np.tile(np.expand_dims(np.log10(self.nu.to(u.GHz).value), 
                                                    axis=(0,1,-1)), 
                                                    (1, samples.shape[1], 1, self.theta.size)),
                             np.expand_dims(ksi, axis=(0,))), axis=0)

        targets = np.tile(np.expand_dims(self.theta.to(u.radian).value, axis=(0,1,2)), (1, samples.shape[1], self.lam.size, 1))

        samples = samples.reshape((samples.shape[0], -1)).T
        targets = targets.flatten()
        self.nsamples = samples.shape[0]

        samples = samples.astype(np.float32)

        good = np.logical_not(np.logical_or(samples[:,-1] < 2e-8, samples[:,-1] == 1))

        samples, targets = samples[good,:], targets[good]
        self.original_indices = self.original_indices[good]

        samples[:,-1] = 2*samples[:,-1] - 1
        samples[:,-1] = np.arctanh(samples[:,-1])

        return samples, targets

    def plot_scattering_phase_function_model(self):
        import matplotlib.pyplot as plt

        index_samples = np.random.randint(0, self.samples.shape[0], 1)[0]

        if "log10_amax" in self.dims:
            log10_amax = np.repeat(np.log10(self.amax[index_samples].to(u.cm).value), self.theta.size)
        if "p" in self.dims:
            p = np.repeat(self.p[index_samples], self.theta.size)
        if "abundances" in self.dims:
            abundances = tuple([np.repeat(a[index_samples], self.theta.size) for a in self.abundances])

        index_nu = np.random.randint(0, self.nu.size, 1)[0]
        
        log10_lam = np.repeat(np.log10(self.lam[index_nu].value), self.theta.size)
        log10_nu = np.repeat(np.log10(self.nu[index_nu].to(u.GHz).value), self.theta.size)

        interpolated = np.log10(self.scattering_phase_function[index_samples, index_nu, :])

        print_str = ""
        for dim in self.dims:
            if dim == "abundances":
                print_str += f"{dim}: {[abundances[i][0] for i in range(len(abundances))]}, "
            else:
                print_str += f"{dim}: {locals()[dim][0]}, "
        print(print_str)

        samples = ()
        for dim in self.dims:
            if dim == "abundances":
                samples += abundances
            else:
                samples += (locals()[dim],)
        
        samples = np.vstack(samples + (log10_nu, self.theta.to(u.radian).value)).T
        plot_x = self.theta

        nned = self.scattering_phase_function_y_scaler.inverse_transform(self.scattering_phase_function_model(self.scattering_phase_function_x_scaler.transform(torch.tensor(samples, dtype=torch.float32)))).detach().numpy()

        plt.plot(plot_x, interpolated)
        plt.plot(plot_x, nned)
        plt.show()

    def state_dict(self):
        state_dict = super().state_dict()

        state_dict["dust_properties"]["scattering_phase_function"] = self.scattering_phase_function
        state_dict["dust_properties"]["theta"] = self.theta

        if hasattr(self, "scattering_phase_function_model"):
            state_dict["scattering_phase_function_state_dict"] = self.scattering_phase_function_model.state_dict()
            state_dict["scattering_phase_function_x_scaler"] = self.scattering_phase_function_x_scaler.state_dict()
            state_dict["scattering_phase_function_y_scaler"] = self.scattering_phase_function_y_scaler.state_dict()

        if hasattr(self, "random_direction_model"):
            state_dict["random_direction_state_dict"] = self.random_direction_model.state_dict()
            state_dict["random_direction_x_scaler"] = self.random_direction_x_scaler.state_dict()
            state_dict["random_direction_y_scaler"] = self.random_direction_y_scaler.state_dict()

        return state_dict

def suggest_opacity_sampling(nsamples, p_range=None, amax_range=None, n_dust_subspecies=1, mode='lhs'):
    """
    Suggest samples for learning the opacity as a function of the dust properties using Latin Hypercube sampling.

    Parameters
    ----------
    nsamples : int
        The number of samples to generate.
    p_range : tuple
        The range of power-law indices to sample from. If None, do not sample over power-law index.
    amax_range : tuple
        The range of maximum dust grain sizes to sample from. If None, do not sample over maximum
        dust grain size.
    n_dust_subspecies: int
        The number of component dust species whose abundances may vary. Note that the returned samples will
        have n_dust_subspecies - 1 abundance samples because the abundances must sum to 1, and the value of the
        last species abundance is implicit.

    Returns
    -------
    samples : numpy.ndarray
        An array of shape (nsamples, ndims) where ndims is the number of dimensions sampled over (i.e. 1 for each of 
        p and amax if they are not None, plus n_dust_subspecies - 1 for the dust species abundances).
    """
    ndims = 0
    if p_range is not None:
        ndims += 1
    if amax_range is not None:
        ndims += 1
    ndims += n_dust_subspecies - 1

    if mode == 'lhs':
        sampler = scipy.stats.qmc.LatinHypercube(d=ndims)
        samples = sampler.random(nsamples)
    elif mode == 'random':
        samples = np.random.rand(nsamples, ndims)
    else:
        raise ValueError("Invalid mode. Must be either 'lhs' or 'random'.")

    index = 0
    if p_range is not None:
        samples[:,index] = samples[:,index] * (p_range[1] - p_range[0]) + p_range[0]
        index += 1
    if amax_range is not None:
        samples[:,index] = 10.**(samples[:,index] * (np.log10(amax_range[1].to(u.cm).value) - np.log10(amax_range[0].to(u.cm).value)) + np.log10(amax_range[0].to(u.cm).value))
        index += 1
    
    for i in range(1, n_dust_subspecies-1):
        samples[:, index+i] = (1. - samples[:, index:index+i].sum(axis=1)) * samples[:, index+i]

    return samples

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

    if "g" in state_dict["dust_properties"]:
        d = HenyeyGreensteinDust(**state_dict["dust_properties"], device=device)
    elif "scattering_phase_function" in state_dict["dust_properties"]:
        d = GeneralDust(**state_dict["dust_properties"], device=device)
    else:
        d = IsotropicDust(**state_dict["dust_properties"], device=device)

    for attr in ["kabs", "ksca", "pmo", "random_nu", "g", "scattering_phase_function"]:
        if f"{attr}_state_dict" in state_dict:
            if attr in ["random_nu", "scattering_phase_function"]:
                input_size = d.ndims + 2
            else:
                input_size = d.ndims + 1

            hidden_units = [state_dict[f'{attr}_state_dict'][key].shape[0] for key in state_dict[f'{attr}_state_dict'] if 'bias' in key][0:-1]
            d.initialize_model(model=attr, model_type="MLP", input_size=input_size, output_size=1, hidden_units=hidden_units)

            getattr(d, f'{attr}_model').load_state_dict(state_dict[f'{attr}_state_dict'])
            setattr(d, f'{attr}_x_scaler', StandardScaler())
            getattr(d, f'{attr}_x_scaler').load_state_dict(state_dict[f"{attr}_x_scaler"])
            setattr(d, f'{attr}_y_scaler', StandardScaler())
            getattr(d, f'{attr}_y_scaler').load_state_dict(state_dict[f"{attr}_y_scaler"])

    if "random_direction_state_dict" in state_dict:
        hidden_units = (tuple([state_dict['random_direction_state_dict'][key].size(1) for key in state_dict['random_direction_state_dict'] if '0.hyper' in key and 'weight' in key and '0.weight' not in key]),) * len([state_dict['random_direction_state_dict'][key].size(0) for key in state_dict['random_direction_state_dict'] if '0.weight' in key])

        d.initialize_model(model="random_direction", model_type="flow", input_size=1, output_size=3, hidden_units=hidden_units)

        d.random_direction_model.load_state_dict(state_dict['random_direction_state_dict'])

        d.random_direction_x_scaler = StandardScaler()
        d.random_direction_x_scaler.load_state_dict(state_dict["random_direction_x_scaler"])
        d.random_direction_y_scaler = StandardScaler()
        d.random_direction_y_scaler.load_state_dict(state_dict["random_direction_y_scaler"])

    if "ml_step_state_dict" in state_dict:
        hidden_units = (tuple([state_dict['ml_step_state_dict'][key].size(1) for key in state_dict['ml_step_state_dict'] if '0.hyper' in key and 'weight' in key and '0.weight' not in key]),) * len([state_dict['ml_step_state_dict'][key].size(0) for key in state_dict['ml_step_state_dict'] if '0.weight' in key])

        d.initialize_model(model="ml_step", model_type="flow", input_size=7, output_size=5, hidden_units=hidden_units)

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

class NeuralSplineFlow(zuko.flows.NSF):
    def condition(self, y):
        return self(y)

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
        return torch.where(self.std == 0, 0., (data - self.mean) / self.std)

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
            loss = -self.condition(y).log_prob(x).mean()
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
            return self.condition(y).sample()
        else:
            return self(x)
