import urllib
import requests
from .sources import Star
from .grids import UniformSphericalGrid
from .utils import log_uniform_interp, log_uniform_interp_extra_dim
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.spatial.transform import Rotation
import pandas as pd
import scipy.interpolate
from scipy.stats import gaussian_kde
from tqdm import trange
from astropy.modeling import models
import astropy.units as u
import astropy.constants as const
import scipy.stats.qmc
import scipy.integrate
from scipy.interpolate import RegularGridInterpolator
import warp as wp
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch
import os
import PyMieScatt
from KDEpy import FFTKDE
import interpn
import psutil

class Dust(pl.LightningDataModule):
    def __init__(self, lam=None, kabs=None, ksca=None, recipe=None, optical_constants=None, 
                 amin=None, amax=None, p=None, with_dhs=False, fmax=0.8, nf=50, interpolate=10000, 
                 device="cpu"):
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
        recipe : str or dict
            Recipe to use for the dust composition. If a string, it should be one of the predefined recipes.
        optical_constants : DustOpticalConstants
            Precomputed optical constants for the dust to use to calculate the opacities.
        amin : astropy.units.Quantity
            Minimum grain size for the size distribution.
        amax : astropy.units.Quantity
            Maximum grain size for the size distribution.
        p : float
            Power-law index for the grain size distribution.
        with_dhs : bool
            Whether to use the Distribution of Hollow Spheres method for calculating opacities.
        fmax : float
            Maximum volume fraction of the hollow sphere in the DHS method.
        nf : int
            Number of discrete hollow sphere fractions to use in the DHS method.
        interpolate : int
            Number of points to interpolate the dust opacities.
        device : str
            Device to run the computations on (e.g., "cpu" or "cuda").
        """
        super().__init__()

        if recipe is not None or optical_constants is not None:
            if recipe is not None:
                optical_constants = DustOpticalConstants(recipe=recipe)

                if type(recipe) == str and recipe in DustOpticalConstants.recipes:
                    recipe = recipe_dict[recipe]

                if "with_dhs" in recipe:
                    with_dhs = recipe["with_dhs"]
                if "filling" in recipe:
                    fmax = recipe["filling"]

            optical_constants.calculate_size_distribution_opacity(amin, amax, p, nang=1, with_dhs=with_dhs, fmax=fmax, nf=nf)

            lam = optical_constants.lam
            kabs = optical_constants.kabs
            ksca = optical_constants.ksca

        kunit = kabs.unit
        lam_unit = lam.unit

        if interpolate > 0:
            f_kabs = scipy.interpolate.interp1d(np.log10(lam.value), np.log10(kabs.value), kind="cubic")
            f_ksca = scipy.interpolate.interp1d(np.log10(lam.value), np.log10(ksca.value), kind="cubic")

            lam = 10.**np.linspace(np.log10(lam.value).min(), np.log10(lam.value).max(), interpolate)[::-1]
            kabs = 10.**f_kabs(np.log10(lam))
            ksca = 10.**f_ksca(np.log10(lam))
        else:
            lam = lam.value
            kabs = kabs.value
            ksca = ksca.value

        self.nu = (const.c / (lam * lam_unit)).decompose().to(u.GHz)
        self.kmean = np.mean(kabs) * kunit
        self.lam = lam * lam_unit
        self.kabs = kabs / self.kmean.value
        self.ksca = ksca / self.kmean.value
        self.kext = (kabs + ksca) / self.kmean.value
        self.albedo = ksca / (kabs + ksca)

        with wp.ScopedDevice(device):
            self.nu_wp = wp.array(self.nu.value, dtype=float)
            self.kabs_wp = wp.array(self.kabs, dtype=float)
            self.ksca_wp = wp.array(self.ksca, dtype=float)
            self.kext_wp = wp.array(self.kext, dtype=float)
            self.albedo_wp = wp.array(self.albedo, dtype=float)

        self.temperature = np.logspace(-1.,4.,1000)
        self.log_temperature = np.log10(self.temperature)

        random_nu_PDF = np.array([kabs * models.BlackBody(temperature=T*u.K)(self.nu) for T in self.temperature])
        self.random_nu_CPD = scipy.integrate.cumulative_trapezoid(random_nu_PDF, self.nu, axis=1, initial=0.)
        self.random_nu_CPD /= self.random_nu_CPD[:,-1:]
        self.drandom_nu_CPD_dT = np.gradient(self.random_nu_CPD, self.temperature, axis=0)

        vectorized_bb = np.vectorize(lambda T: self.kmean.cgs.value * scipy.integrate.trapezoid(self.kabs * \
                models.BlackBody(temperature=T*u.K)(self.nu).cgs.value, self.nu.to(u.Hz).value))

        self.pmo = np.pi / (const.sigma_sb.cgs.value * self.temperature**4) * vectorized_bb(self.temperature)

    def to_device(self, device):
        with wp.ScopedDevice(device):
            self.nu_wp = wp.array(self.nu.value, dtype=float)
            self.kabs_wp = wp.array(self.kabs, dtype=float)
            self.ksca_wp = wp.array(self.ksca, dtype=float)
            self.kext_wp = wp.array(self.kext, dtype=float)
            self.albedo_wp = wp.array(self.albedo, dtype=float)

        if hasattr(self, "random_nu_model"):
            self.random_nu_model.to(device)
        if hasattr(self, "ml_step_model"):
            self.ml_step_model.to(device)

    def interpolate_kabs(self, nu):
        return np.interp(nu, self.nu, self.kabs)

    def interpolate_ksca(self, nu):
        return np.interp(nu, self.nu, self.ksca)

    def interpolate_kabs_wp(self, photon_list, iphotons, frequency=None):
        if frequency is None:
            frequency = photon_list.frequency

        kabs = wp.zeros(len(frequency), dtype=float)
        wp.launch(log_uniform_interp, dim=len(frequency), inputs=[frequency, self.nu_wp, self.kabs_wp, kabs])
        return kabs

    def interpolate_ksca_wp(self, photon_list, iphotons, frequency=None):
        if frequency is None:
            frequency = photon_list.frequency

        ksca = wp.zeros(len(frequency), dtype=float)
        wp.launch(log_uniform_interp, dim=len(frequency), inputs=[frequency, self.nu_wp, self.ksca_wp, ksca])
        return ksca

    def interpolate_kext(self, nu):
        return np.interp(nu, self.nu, self.kext)
    
    def interpolate_kext_wp(self, photon_list, iphotons, frequency=None):
        if frequency is None:
            frequency = photon_list.frequency

        kext = wp.zeros((len(photon_list.in_grid), len(frequency)), dtype=float)
        wp.launch(log_uniform_interp_extra_dim, dim=(len(photon_list.in_grid), len(frequency)), inputs=[frequency, self.nu_wp, self.kext_wp, kext])
        return kext

    def interpolate_albedo(self, nu):
        return np.interp(nu, self.nu, self.albedo)

    def interpolate_albedo_wp(self, photon_list, iphotons, frequency=None):
        if frequency is None:
            frequency = photon_list.frequency

        albedo = wp.zeros((len(photon_list.in_grid), len(frequency)), dtype=float)
        wp.launch(log_uniform_interp_extra_dim, dim=(len(photon_list.in_grid), len(frequency)), inputs=[frequency, self.nu_wp, self.albedo_wp, albedo])
        return albedo

    def absorb(self, temperature):
        nphotons = frequency.numpy().size

        cost = -1. + 2*np.random.rand(nphotons)
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

    def random_nu(self, photon_list, subset=None):
        temperature = wp.to_torch(photon_list.temperature)
        if subset is not None:
            temperature = temperature[subset]
            
        nphotons = temperature.size(0)
        ksi = torch.rand(int(nphotons), device=wp.device_to_torch(wp.get_device()), dtype=torch.float32)

        test_x = torch.transpose(torch.vstack((torch.log10(temperature), ksi)), 0, 1)

        nu = wp.from_torch(10.**torch.flatten(self.random_nu_model(test_x).detach()))

        return nu

    def planck_mean_opacity(self, temperature, grid):
        """
        vectorized_bb = np.vectorize(lambda T: self.kmean.cgs.value * scipy.integrate.trapezoid(self.kabs * \
                models.BlackBody(temperature=T*u.K)(self.nu).cgs.value, self.nu.to(u.Hz).value))

        return np.pi / (const.sigma_sb.cgs.value * temperature**4) * vectorized_bb(temperature)
        """
        return np.interp(temperature, self.temperature, self.pmo)

    def ml_step(self, photon_list, s, iphotons):
        nphotons = iphotons.size

        test_x = torch.tensor(np.vstack((np.log10(photon_list.frequency.numpy()[iphotons]),
                              np.log10(photon_list.temperature.numpy()[iphotons]),
                              np.log10(photon_list.density.numpy()[iphotons] * photon_list.kabs.numpy()[iphotons] * s[iphotons]),
                              np.random.rand(int(nphotons)),
                              np.random.rand(int(nphotons)),
                              np.random.rand(int(nphotons)),
                              np.random.rand(int(nphotons)),
                              np.random.rand(int(nphotons)),
                              np.random.rand(int(nphotons)),
                              np.random.rand(int(nphotons)),
                              np.random.rand(int(nphotons)),
                              np.random.rand(int(nphotons)))).T, dtype=torch.float32)

        vals = self.ml_step_model(test_x).detach().numpy()

        return 10.**vals[:,0], 10.**vals[:,1], vals[:,2], vals[:,3], vals[:,4], vals[:,5], vals[:,6], vals[:,7], vals[:,8]

    def initialize_model(self, model="random_nu", input_size=2, output_size=1, hidden_units=(48, 48, 48)):
        all_layers = [nn.Flatten()]

        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.Sigmoid())
            input_size = hidden_unit

        all_layers.append(nn.Linear(hidden_units[-1], output_size))

        if model == "random_nu":
            self.random_nu_model = nn.Sequential(*all_layers)
        elif model == "ml_step":
            self.ml_step_model = nn.Sequential(*all_layers)

    def learn(self, model="random_nu", nsamples=200000, test_split=0.1, valid_split=0.2, hidden_units=(48, 48, 48), max_epochs=10, plot=False, 
            tau_range=(0.5, 4.0), temperature_range=(-1.0, 4.0), nu_range=None):
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
        max_epochs : int
            The maximum number of epochs to train the model.
        plot : bool
            Whether to plot the results after training.
        """
        self.nsamples = nsamples
        self.test_split = test_split
        self.valid_split = valid_split
        self.learning = model

        # Set up the NN

        if model == "random_nu":
            input_size, output_size = 2, 1
        elif model == "ml_step":
            input_size, output_size = 12, 9

            if nu_range is None:
                nu_range = (self.nu.value.min(), self.nu.value.max())

            self.log10_nu0_min = nu_range[0]
            self.log10_nu0_max = nu_range[1]
            self.log10_T_min = temperature_range[0]
            self.log10_T_max = temperature_range[1]
            self.log10_tau_cell_nu0_min = tau_range[0]
            self.log10_tau_cell_nu0_max = tau_range[1]

        self.initialize_model(model=model, input_size=input_size, output_size=output_size, hidden_units=hidden_units)

        # Wrap the model in lightning

        self.dustLM = DustLightningModule(getattr(self, model+"_model"))

        self.trainer = pl.Trainer(max_epochs=max_epochs)
        self.trainer.fit(model=self.dustLM, datamodule=self)

        # Test the model.

        self.trainer.test(model=self.dustLM, datamodule=self)

        # Plot the result

        if plot:
            import matplotlib.pyplot as plt

            if model == "random_nu":
                y_pred = trainer.predict(dustLM, datamodule=self)
                y_pred = torch.cat(y_pred)
                y_true = torch.cat([batch[1] for batch in self.predict_dataloader()])

                with torch.no_grad():
                    count, bins, patches = plt.hist(y_true, 100, histtype='step')
                    plt.hist(y_pred, bins, histtype='step')
                    plt.savefig("predicted_vs_actual.png")
            else:
                self.plot_ml_step()

    def prepare_data(self):
        if self.learning == "random_nu":
            self.prepare_data_random_nu()
        elif self.learning == "ml_step":
            self.prepare_data_ml_step()

    def prepare_data_random_nu(self):
        sampler = scipy.stats.qmc.LatinHypercube(d=2)
        samples = sampler.random(self.nsamples)

        logT = 5*samples[:,0] - 1.
        ksi = samples[:,1]
        X = torch.tensor(np.vstack((logT, ksi)).T, dtype=torch.float32)

        y = np.concatenate([self.random_nu_manual(10.**X_batch[:,0].numpy(), X_batch[:,1].numpy()) for X_batch in DataLoader(X, batch_size=1000)])
        y = torch.tensor(np.log10(y.value), dtype=torch.float32)

        self.dataset = TensorDataset(X, y)

    def prepare_data_ml_step(self, device='cpu'):
        if hasattr(self, "dataset"):
            return

        if os.path.exists("sim_results.csv"):
            df = pd.read_csv("sim_results.csv", index_col=0)

            self.log10_nu0_min = df['log10_nu0'].min()
            self.log10_nu0_max = df['log10_nu0'].max()
            self.log10_T_min = df['log10_T'].min()
            self.log10_T_max = df['log10_T'].max()
            self.log10_tau_cell_nu0_min = df['log10_tau_cell_nu0'].min()
            self.log10_tau_cell_nu0_max = df['log10_tau_cell_nu0'].max()
        else:
            df = self.run_dust_simulation(nphotons=self.nsamples, tau_range=(self.log10_tau_cell_nu0_min, self.log10_tau_cell_nu0_max),
                    temperature_range=(self.log10_T_min, self.log10_T_max), nu_range=(self.log10_nu0_min, self.log10_nu0_max))
            df.to_csv("sim_results.csv")

        self.df = df
        self.nsamples = len(df)

        new_order = df.columns[[0, 1, 2, 3, 4, 9, 10, 11, 8, 6, 7, 5]]

        df = df.loc[:, new_order]

        dependencies = {
            "log10_nu":["log10_nu0", "log10_T", "log10_tau_cell_nu0"],
            "log10_Eabs":["log10_nu0", "log10_T", "log10_tau_cell_nu0", "log10_nu"],
            "direction_yaw":["log10_nu0", "log10_T", "log10_tau_cell_nu0","log10_nu", "log10_Eabs"],
            "direction_pitch":["log10_nu0", "log10_T", "log10_nu", "log10_Eabs"],
            "direction_roll":["log10_nu0", "log10_T", "log10_tau_cell_nu0", "log10_nu", "log10_Eabs"],
            "roll":["log10_nu", "log10_Eabs", "direction_yaw", "direction_pitch", "direction_roll"],
            "yaw":["log10_nu", "log10_Eabs", "direction_yaw", "direction_pitch", "direction_roll", "roll"],
            "pitch":["log10_nu", "log10_Eabs", "direction_yaw", "direction_pitch", "direction_roll", "roll"],
            "tau":["log10_nu", "log10_Eabs", "direction_yaw", "direction_pitch", "direction_roll", "roll", "pitch"],
        }

        new_columns = []
        for i in trange(3,df.shape[1]):
            column = df.columns[i]

            bw = len(df)**(-1./(len(dependencies[column])+1 + 4))

            kde = FFTKDE(bw=bw, kernel="gaussian")
            kde.fit(df[dependencies[column] + [column]].values)

            Ndim = len(dependencies[column]) + 1
            N = min(100, round((0.1*psutil.virtual_memory().available / 8 / Ndim )**(1./Ndim)))
            shape = (N,)*Ndim

            result = kde.evaluate(shape)

            cdf = np.cumsum(result[1].reshape(shape), axis=-1)

            slices = (slice(None),) * len(dependencies[column])
            cdf /= cdf[slices + (-1,)][slices + (np.newaxis,)]

            dims = cdf.shape
            starts = result[0].min(axis=0)
            steps = np.array([np.unique(result[0][:,j])[1] - np.unique(result[0][:,j])[0] for j in range(result[0].shape[1])])

            cdf_interp = interpn.MultilinearRegular.new(dims, starts, steps, cdf)
            ksi = cdf_interp.eval(df.loc[:, dependencies[column] + [column]].values.T)

            new_columns.append(("ksi_"+column, ksi))

        for name, values in new_columns:
            df[name] = values

        features = ["log10_nu0", "log10_T", "log10_tau_cell_nu0", "ksi_log10_nu", "ksi_log10_Eabs", "ksi_tau", "ksi_yaw", "ksi_pitch", "ksi_roll", "ksi_direction_yaw", "ksi_direction_pitch", "ksi_direction_roll"]
        targets = ["log10_nu", "log10_Eabs", "tau", "yaw", "pitch", "roll", "direction_yaw", "direction_pitch", "direction_roll"]

        X = torch.tensor(df.loc[:, features].values, dtype=torch.float32)
        y = torch.tensor(df.loc[:, targets].values, dtype=torch.float32)

        self.dataset = TensorDataset(X, y)

    def run_dust_simulation(self, nphotons=1000, tau_range=(0.5, 4.0), temperature_range=(-1.0, 4.0), nu_range=None, use_ml_step=False):
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
        nu_range : tuple
            The range of frequencies to sample from (in GHz). If None, use the full range of the dust opacities.
        """
        if nu_range is None:
            nu_range = (self.nu.value.min(), self.nu.value.max())

        # Set up the star.

        star = Star()
        star.set_blackbody_spectrum(self.nu)

        # Set up the grid.

        grid = UniformSphericalGrid(ncells=1, dr=1.0*u.au, mirror=False)

        density = np.ones(grid.shape) * 1e-16 * u.g / u.cm**3

        grid.add_density(density, self)
        grid.add_star(star)

        # Emit the photons

        photon_list = grid.emit(nphotons, wavelength="random", scattering=False)

        initial_direction = np.zeros((nphotons, 3), dtype=np.float32)
        initial_direction[:,0] = 1.
        photon_list.direction = wp.array(initial_direction, dtype=wp.vec3)

        photon_list.frequency = wp.array(10.**np.random.uniform(np.log10(nu_range[0].value), np.log10(nu_range[1].value), nphotons), dtype=float)
        original_frequency = photon_list.frequency.numpy().copy()

        photon_list.temperature = wp.array(10.**np.random.uniform(temperature_range[0], temperature_range[1], nphotons), dtype=float)

        tau = 10.**np.random.uniform(tau_range[0], tau_range[1], nphotons)
        photon_list.density = wp.array((tau / (self.kmean * self.interpolate_kabs(photon_list.frequency.numpy()*u.GHz) * 1.*u.au) * self.kmean).to(1 / u.au), dtype=float)

        grid.propagate_photons(photon_list, learning=True, use_ml_step=use_ml_step)

        # Calculate roll, pitch, and yaw for the position relative to where it started.

        ypr = []
        for (direction0, direction) in zip(initial_direction, photon_list.direction.numpy()):
            rot, _ = Rotation.align_vectors(direction, direction0)
            ypr.append(rot.as_euler('zyx'))
        ypr = np.array(ypr)

        # Calculate roll, pitch, and yaw for the direction relative to the radial vector where it exits.

        direction_ypr = []
        for (position, direction) in zip(photon_list.position.numpy(), photon_list.direction.numpy()):
            rot, _ = Rotation.align_vectors(direction, position)
            direction_ypr.append(rot.as_euler('zyx'))
        direction_ypr = np.array(direction_ypr)

        # Store the results in a pandas DataFrame

        df = pd.DataFrame({"log10_nu0":np.log10(original_frequency),
                       "log10_T":np.log10(photon_list.temperature.numpy()),
                       "log10_tau_cell_nu0":np.log10(tau),
                       "log10_nu":np.log10(photon_list.frequency.numpy().copy()),
                       "log10_Eabs":np.log10(np.where(photon_list.deposited_energy.numpy() > 0, photon_list.deposited_energy.numpy(), 1.0e-5)/photon_list.energy.numpy()),
                       "tau":photon_list.tau.numpy().copy(),
                       "yaw":ypr[:,0],
                       "pitch":ypr[:,1],
                       "roll":ypr[:,2],
                       #"direction_theta":np.acos((photon_list.position.numpy() * photon_list.direction.numpy()).sum(axis=1))})
                       "direction_yaw":direction_ypr[:,0],
                       "direction_pitch":direction_ypr[:,1],
                       "direction_roll":direction_ypr[:,2]})

        return df

    # DataModule functions

    def plot_ml_step(self):
        import matplotlib.pyplot as plt

        if self.trainer is None and hasattr(self, "ml_step_model"):
            self.dustLM = DustLightningModule(getattr(self, "ml_step_model"))

            self.trainer = pl.Trainer()

            self.learning = 'ml_step'
            self.test_split = 0.1
            self.valid_split = 0.2

        if self.trainer is not None:
            y_pred = self.trainer.predict(self.dustLM, datamodule=self)
            y_pred = torch.cat(y_pred).numpy()
            y_true = torch.cat([batch[1] for batch in self.predict_dataloader()])

            X_test = torch.cat([batch[0] for batch in self.predict_dataloader()]).numpy()

            predict = True

        columns = self.df.columns
        features = np.array(["log10_nu0", "log10_T", "log10_tau_cell_nu0", "ksi_log10_nu", "ksi_log10_Eabs", "ksi_tau", "ksi_yaw", "ksi_pitch", "ksi_roll", "ksi_direction_yaw", "ksi_direction_pitch", "ksi_direction_roll"])
        targets = np.array(["log10_nu", "log10_Eabs", "tau", "yaw", "pitch", "roll", "direction_yaw", "direction_pitch", "direction_roll"])

        fig, ax = plt.subplots(nrows=len(columns), ncols=len(columns), figsize=(11,11))

        for i, key1 in enumerate(columns):
            for j, key2 in enumerate(columns):
                if key1 == key2:
                    ax[i,j].hist(self.df[key1], bins=50, histtype='step', density=True)
                    if key1 in targets and predict:
                        ax[i,j].hist(y_pred[:,np.where(targets == key1)[0][0]], bins=50, histtype='step', density=True)
                elif i > j:
                    ax[i,j].scatter(self.df[key2], self.df[key1], marker='.', s=0.025)

                    if (key1 in targets or key2 in targets) and predict:
                        if key2 in targets:
                            x = y_pred[:,np.where(targets == key2)[0][0]]
                        else:
                            x = X_test[:,np.where(features == key2)[0][0]]

                        if key1 in targets:
                            y = y_pred[:,np.where(targets == key1)[0][0]]
                        else:
                            y = X_test[:,np.where(features == key1)[0][0]]

                        ax[i,j].scatter(x, y, marker='.', s=0.025)
                elif i < j:
                    ax[i,j].set_axis_off()

                if i == len(columns) - 1:
                    ax[i,j].set_xlabel(key2)

            ax[i,0].set_ylabel(key1)

        plt.show()

    def setup(self, stage=None):
        if hasattr(self, "train") and hasattr(self, "valid") and hasattr(self, "test"):
            return
        test_size = int(self.test_split * self.nsamples)
        valid_size = int((self.nsamples - test_size)*self.valid_split)
        train_size = self.nsamples - test_size - valid_size
        train_val_tmp, self.test = random_split(self.dataset, [train_size + valid_size, test_size], generator=torch.Generator().manual_seed(1))
        self.train, self.val = random_split(train_val_tmp, [train_size, valid_size], generator=torch.Generator().manual_seed(2))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=100, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=100, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=100, num_workers=2)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=100, num_workers=2)

    def state_dict(self):
        state_dict = {
            "dust_properties":{
                "lam": self.lam,
                "kabs": self.kabs*self.kmean,
                "ksca": self.ksca*self.kmean,
            },
        }

        if hasattr(self, "random_nu_model"):
            state_dict["random_nu_state_dict"] = self.random_nu_model.state_dict()

        if hasattr(self, "ml_step_model"):
            state_dict["ml_step_state_dict"] = self.ml_step_model.state_dict()

            state_dict["log10_nu0_min"] = self.log10_nu0_min
            state_dict["log10_nu0_max"] = self.log10_nu0_max
            state_dict["log10_T_min"] = self.log10_T_min
            state_dict["log10_T_max"] = self.log10_T_max
            state_dict["log10_tau_cell_nu0_min"] = self.log10_tau_cell_nu0_min
            state_dict["log10_tau_cell_nu0_max"] = self.log10_tau_cell_nu0_max

        return state_dict

    def save(self, filename):
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
        state_dict = torch.load(filename, weights_only=False)
    elif isinstance(filename, dict):
        state_dict = filename
    elif isinstance(filename, Dust):
        state_dict = filename.state_dict()

    d = Dust(**state_dict["dust_properties"], interpolate=-1, device=device)

    if "random_nu_state_dict" in state_dict:
        hidden_units = [state_dict['random_nu_state_dict'][key].shape[0] for key in state_dict['random_nu_state_dict'] if 'bias' in key][0:-1]
        d.initialize_model(model="random_nu", input_size=2, output_size=1, hidden_units=hidden_units)

        d.random_nu_model.load_state_dict(state_dict['random_nu_state_dict'])

    if "ml_step_state_dict" in state_dict:
        hidden_units = [state_dict['ml_step_state_dict'][key].shape[0] for key in state_dict['ml_step_state_dict'] if 'bias' in key][0:-1]
        d.initialize_model(model="ml_step", input_size=12, output_size=9, hidden_units=hidden_units)

        d.ml_step_model.load_state_dict(state_dict['ml_step_state_dict'])

        d.log10_nu0_min = state_dict["log10_nu0_min"]
        d.log10_nu0_max = state_dict["log10_nu0_max"]
        d.log10_T_min = state_dict["log10_T_min"]
        d.log10_T_max = state_dict["log10_T_max"]
        d.log10_tau_cell_nu0_min = state_dict["log10_tau_cell_nu0_min"]
        d.log10_tau_cell_nu0_max = state_dict["log10_tau_cell_nu0_max"]

    return d


class DustLightningModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() == 1:
            y = y.reshape(-1,1)
        loss = nn.functional.mse_loss(self(x), y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() == 1:
            y = y.reshape(-1,1)
        loss = nn.functional.mse_loss(self(x), y)
        self.log('valid_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() == 1:
            y = y.reshape(-1,1)
        loss = nn.functional.mse_loss(self(x), y)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

recipe_dict = {
    "draine":{
        "dust":["astronomical_silicates","graphite_parallel_0.01","graphite_perpendicular_0.01"],
        "format":["draine","draine","draine"],
        "density":np.array([3.3,2.24,2.24]),
        "abundance":np.array([0.65,0.35*1./3,0.35*2./3])
    },
    "dsharp":{
        "dust":["astronomical_silicates","troilite","organics","water_ice"],
        "format":["draine","henn","henn","henn"],
        "density":np.array([3.3,4.83,1.5,0.92]),
        "abundance":np.array([0.1670,0.0258,0.4430,0.3642])
    },
    "pollack":{
        "dust":["astronomical_silicates","troilite","organics","water_ice"],
        "format":["draine","henn","henn","henn"],
        "density":np.array([3.3,4.83,1.5,0.92]),
        "mass_fraction":np.array([3.41e-3,7.68e-4,4.13e-3,5.55e-3]),
    },
    "diana":{
        "dust":["amorphous_silicates_extrapolated","amorphous_carbon_zubko1996_extrapolated"],
        "format":["henn","henn"],
        "density":np.array([3.3,1.0]),
        "abundance":np.array([0.8,0.2]),
        "filling":0.75,
        "with_dhs":True,
    },
    "diana_wice":{
        "dust":["amorphous_silicates_extrapolated","amorphous_carbon_zubko1996_extrapolated","water_ice"],
        "format":["henn","henn","henn"],
        "density":np.array([3.3,1.0,0.92]),
        "abundance":np.array([0.8,0.2,0.5]),
        "filling":0.75,
        "with_dhs":True,
    },
}
    
class DustOpticalConstants:
    recipes = ["draine","pollack","diana","diana_wice"]

    def __init__(self, recipe=None):
        if recipe is not None:
            if type(recipe) == str:
                recipe = recipe_dict[recipe]
            elif type(recipe) == dict:
                pass

            water_ice = DustOpticalConstants()
            water_ice.set_optical_constants_from_henn("water_ice.txt")

            species = []
            for i in range(len(recipe["dust"])):
                species.append(DustOpticalConstants())
                if recipe["format"][i] == "henn":
                    species[-1].set_optical_constants_from_henn(recipe["dust"][i]+".txt")
                    if "extrapolated" in recipe["dust"][i]:
                        species[-1].calculate_optical_constants_on_wavelength_grid(water_ice.lam)
                elif recipe["format"][i] == "draine":
                    species[-1].set_optical_constants_from_draine(recipe["dust"][i]+".txt")
                    species[-1].calculate_optical_constants_on_wavelength_grid(water_ice.lam)
                species[-1].set_density(recipe["density"][i])

            if "mass_fraction" in recipe:
                abundances = (recipe["mass_fraction"]/recipe["density"])/(recipe["mass_fraction"]/recipe["density"]).sum()
            else:
                abundances = recipe["abundance"] / recipe["abundance"].sum()

            if "filling" not in recipe:
                recipe["filling"] = 1.

            dust = mix_dust(species, abundances, filling=recipe["filling"])

            self.set_optical_constants(dust.lam, dust.n, dust.k)
            self.set_density(dust.rho)
            
    def add_coat(self, coat):
        self.coat = coat

    def calculate_optical_constants_on_wavelength_grid(self, lam):
        f = scipy.interpolate.interp1d(self.lam, self.n)
        n = f(lam)
        f = scipy.interpolate.interp1d(self.lam, self.k)
        k = f(lam)

        self.lam = lam
        self.nu = const.c.to('cm/s').value / self.lam

        self.n = n
        self.k = k
        self.m = self.n + 1j*self.k

    def calculate_size_distribution_opacity(self, amin, amax, p, \
            coat_volume_fraction=0.0, nang=1000, with_dhs=False, fmax=0.8, \
            nf=50):
        na = int(round(np.log10(amax.to(u.um).value) - np.log10(amin.to(u.um).value))*100+1)
        a = np.logspace(np.log10(amin.to(u.um).value),np.log10(amax.to(u.um).value),na) * u.um
        kabsgrid = np.zeros((self.lam.size,na))
        kscagrid = np.zeros((self.lam.size,na))
        
        normfunc = a**(3-p)

        for i in range(na):
            if with_dhs:
                self.calculate_dhs_opacity(a[i], fmax=fmax, nf=nf, nang=nang)
            else:
                self.calculate_opacity(a[i], \
                        coat_volume_fraction=coat_volume_fraction, nang=nang)
            
            kabsgrid[:,i] = self.kabs*normfunc[i]
            kscagrid[:,i] = self.ksca*normfunc[i]
        
        norm = scipy.integrate.trapezoid(normfunc,x=a)
        
        self.kabs = scipy.integrate.trapezoid(kabsgrid,x=a)/norm
        self.ksca = scipy.integrate.trapezoid(kscagrid,x=a)/norm
        self.kext = self.kabs + self.ksca
        self.albedo = self.ksca / self.kext

    def calculate_opacity(self, a, coat_volume_fraction=0.0, nang=1000):
        self.kabs = np.zeros(self.lam.size) * a.unit**2 / u.g
        self.ksca = np.zeros(self.lam.size) * a.unit**2 / u.g
        
        if not hasattr(self, 'coat'):
            mdust = 4*np.pi*a**3/3*self.rho
            
            for i in range(self.lam.size):                
                Qext,Qsca,Qabs,gsca,Qpr,Qback,Qratio=PyMieScatt.MieQ(self.m[i],self.lam[i].to(u.nm).value,a.to(u.nm).value)

                self.kabs[i] = np.pi*a**2*Qabs/mdust
                self.ksca[i] = np.pi*a**2*Qsca/mdust
        else:
            a_coat = a*(1+coat_volume_fraction)**(1./3)

            mdust = 4*np.pi*a**3/3*self.rho+ \
                    4*np.pi/3*(a_coat**3-a**3)*self.coat.rho
            
            for i in range(self.lam.size):
                Qext,Qsca,Qabs,gsca,Qpr,Qback,Qratio=PyMieScatt.MieQCoreShell(self.m[i],self.coat.m[i],self.lam[i].to(u.nm).value,a.to(u.nm).value,a_coat.to(u.nm).value)

                self.kabs[i] = np.pi*a_coat**2*Qabs/mdust
                self.ksca[i] = np.pi*a_coat**2*Qsca/mdust

        self.kext = self.kabs + self.ksca
        self.albedo = self.ksca / self.kext

    def calculate_dhs_opacity(self, a, fmax=0.8, nf=50, nang=1000):
        self.kabs = np.zeros(self.lam.size) * a.unit**2 / u.g
        self.ksca = np.zeros(self.lam.size) * a.unit**2 / u.g

        for i in range(self.lam.size):
            for j, f in enumerate(np.linspace(0., fmax, nf)):
                x = 2*np.pi*a*f**(1./3)/self.lam[i]
                y = 2*np.pi*a/self.lam[i]

                if f == 0:
                    Qext,Qsca,Qabs,gsca,Qpr,Qback,Qratio=PyMieScatt.MieQ(self.m[i],self.lam[i].to(u.nm).value,a.to(u.nm).value)
                else:
                    Qext,Qsca,Qabs,gsca,Qpr,Qback,Qratio=PyMieScatt.MieQCoreShell(1.0+1j,self.m[i],self.lam[i].to(u.nm).value,a.to(u.nm).value*f**(1./3),a.to(u.nm).value)

                mdust = 4*np.pi*a**3*(1.-f)/3*self.rho

                self.kabs[i] += np.pi*a**2*Qabs/mdust * 1./fmax * fmax/nf
                self.ksca[i] += np.pi*a**2*Qsca/mdust * 1./fmax * fmax/nf

        self.kext = self.kabs + self.ksca
        self.albedo = self.ksca / self.kext

    def set_density(self, rho):
        if not isinstance(rho, u.Quantity):
            rho = rho * u.g / u.cm**3
        self.rho = rho

    def set_optical_constants(self, lam, n, k):
        self.lam = lam
        if not isinstance(self.lam, u.Quantity):
            self.lam = self.lam * u.cm
        self.nu = (const.c / self.lam).to(u.GHz)

        self.n = n
        self.k = k
        self.m = n+1j*k

    def load_optical_constants_file_generic(self, filename):
        if not os.path.exists(filename):
            if os.path.exists(os.environ["HOME"]+"/.pinballrt/data/optical_constants/"+filename):
                filename = os.environ["HOME"]+"/.pinballrt/data/optical_constants/"+filename
            else:
                web_data_location = 'https://raw.githubusercontent.com/psheehan/pdspy/master/pdspy/dust/data/optical_constants/'+filename
                response = requests.get(web_data_location)
                if response.status_code == 200:
                    if not os.path.exists(os.environ["HOME"]+"/.pinballrt/data/optical_constants"):
                        os.makedirs(os.environ["HOME"]+"/.pinballrt/data/optical_constants")
                    urllib.request.urlretrieve(web_data_location, 
                            os.environ["HOME"]+"/.pinballrt/data/optical_constants/"+filename)
                    filename = os.environ["HOME"]+"/.pinballrt/data/optical_constants/"+filename
                else:
                    print(web_data_location+' does not exist')
                    return

        opt_data = np.loadtxt(filename)

        return opt_data

    def set_optical_constants_from_draine(self, filename):
        opt_data = self.load_optical_constants_file_generic(filename)

        self.lam = np.flipud(opt_data[:,0])*1.0e-4 * u.cm
        self.nu = (const.c / self.lam).to(u.GHz)

        self.n = np.flipud(opt_data[:,3])+1.0
        self.k = np.flipud(opt_data[:,4])
        self.m = self.n+1j*self.k

    def set_optical_constants_from_henn(self, filename):
        opt_data = self.load_optical_constants_file_generic(filename)

        self.lam = opt_data[:,0]*1.0e-4 * u.cm
        self.nu = (const.c / self.lam).to(u.GHz)

        self.n = opt_data[:,1]
        self.k = opt_data[:,2]
        self.m = self.n+1j*self.k

    def set_optical_constants_from_jena(self, filename, type="standard"):
        opt_data = self.load_optical_constants_file_generic(filename)

        if type == "standard":
            self.lam = np.flipud(1./opt_data[:,0]) * u.cm
            self.n = np.flipud(opt_data[:,1])
            self.k = np.flipud(opt_data[:,2])
        elif type == "umwave":
            self.lam = np.flipud(opt_data[:,0])*1.0e-4 * u.cm
            self.n = np.flipud(opt_data[:,1])
            self.k = np.flipud(opt_data[:,2])

        self.nu = (const.c / self.lam).to(u.GHz)
        self.m = self.n+1j*self.k

    def set_optical_constants_from_oss(self, filename):
        opt_data = self.load_optical_constants_file_generic(filename)
        
        self.lam = opt_data[:,0] * u.cm # in cm
        self.nu = (const.c / self.lam).to(u.GHz)

        self.n = opt_data[:,1]
        self.k = opt_data[:,2]
        self.m = self.n+1j*self.k

def mix_dust(dust, abundance, medium=None, rule="Bruggeman", filling=1.):

    if rule == "Bruggeman":
        meff = np.zeros(dust[0].lam.size,dtype=complex)
        rho = 0.0
        
        for i in range(dust[0].lam.size):
            temp = scipy.optimize.fsolve(bruggeman,np.array([1.0,0.0]),\
                    args=(dust,abundance,i, filling))
            meff[i] = temp[0]+1j*temp[1]
        
        for i in range(len(dust)):
            rho += dust[i].rho*abundance[i]

        rho *= filling
    
    elif rule == "MaxGarn":
        numerator = 0.0+1j*0.0
        denominator = 0.0+1j*0.0
        rho = 0.0
        
        for i in range(len(dust)):
            gamma = 3. / (dust[i].m**2 + 2)

            numerator += abundance[i] * gamma * dust[i].m**2
            denominator += abundance[i] * gamma

            rho += dust[i].rho*abundance[i]

        mmix = np.sqrt(numerator / denominator)
        
        F = (mmix**2 - 1.) / (mmix**2 + 2.)

        meff = np.sqrt((1. + 2.*filling*F) / (1. - filling*F))

        rho *= filling

    new = DustOpticalConstants()
    new.set_density(rho)
    new.set_optical_constants(dust[0].lam, meff.real, meff.imag)
    
    return new

def bruggeman(meff, dust, abundance, index, filling):
    
    m_eff = meff[0]+1j*meff[1]
    tot = 0+0j
    
    for j in range(len(dust)):
        tot += filling * abundance[j]*(dust[j].m[index]**2-m_eff**2)/ \
                (dust[j].m[index]**2+2*m_eff**2)

    # Add in the void.

    tot += (1 - filling) * (1. - m_eff**2) / (1. + 2*m_eff**2)
    
    return np.array([tot.real,tot.imag])
