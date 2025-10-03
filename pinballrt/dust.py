import urllib
import requests
from .sources import Star
from .grids import UniformSphericalGrid
from .utils import log_uniform_interp, log_uniform_interp_extra_dim
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.spatial.transform import Rotation
import pandas as pd
import scipy.interpolate
from tqdm import trange
from astropy.modeling import models
import astropy.units as u
import astropy.constants as const
import scipy.stats.qmc
import scipy.integrate
import warp as wp
import numpy as np
import torch.nn as nn
import torch
import os
import PyMieScatt
from sklearn.preprocessing import MinMaxScaler

class Dust:
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

        self.log10_nu_min = np.log10(self.nu.value.min())
        self.log10_nu_max = np.log10(self.nu.value.max())

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

        log10_nu = torch.clamp(self.random_nu_model(test_x).detach(), self.log10_nu_min, self.log10_nu_max)
        nu = wp.from_torch(10.**torch.flatten(log10_nu))

        return nu

    def planck_mean_opacity(self, temperature, grid):
        """
        vectorized_bb = np.vectorize(lambda T: self.kmean.cgs.value * scipy.integrate.trapezoid(self.kabs * \
                models.BlackBody(temperature=T*u.K)(self.nu).cgs.value, self.nu.to(u.Hz).value))

        return np.pi / (const.sigma_sb.cgs.value * temperature**4) * vectorized_bb(temperature)
        """
        return np.interp(temperature, self.temperature, self.pmo)

    def ml_step(self, photon_list, s, iphotons):
        nphotons = iphotons.size(0)

        test_x = torch.transpose(torch.vstack((torch.log10(wp.to_torch(photon_list.frequency)[iphotons]),
                              torch.log10(wp.to_torch(photon_list.temperature)[iphotons]),
                              torch.log10(wp.to_torch(photon_list.density)[iphotons] * wp.to_torch(photon_list.kabs)[iphotons] * s[iphotons]),
                              torch.rand(int(nphotons)),
                              torch.rand(int(nphotons)),
                              torch.rand(int(nphotons)),
                              torch.rand(int(nphotons)),
                              torch.rand(int(nphotons)),
                              torch.rand(int(nphotons)),
                              torch.rand(int(nphotons)),
                              torch.rand(int(nphotons)),
                              torch.rand(int(nphotons)))), 0, 1)

        vals = self.ml_step_model(test_x).detach()

        vals[:,0] = torch.clamp(vals[:,0], self.log10_nu_min, self.log10_nu_max)

        return 10.**vals[:,0], 10.**vals[:,1], 10.**vals[:,2], vals[:,3], vals[:,4], vals[:,5], vals[:,6], vals[:,7], vals[:,8]

    def initialize_model(self, model="random_nu", input_size=2, output_size=1, num_hidden_layers=3, num_hidden_units=48):
        setattr(self, model+"_model", Generator(input_size=input_size, num_hidden_layers=num_hidden_layers, num_hidden_units=num_hidden_units, num_output_units=output_size).to(self.device))

    def learn(self, model="random_nu", nsamples=200000, test_split=0.1, num_hidden_layers=3, num_hidden_units=48, max_epochs=10, 
            tau_range=(0.5, 4.0), temperature_range=(-1.0, 4.0), nu_range=None, device="cpu"):
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
        self.learning = model
        self.device = torch.device(device)

        # Set up the NN

        if model == "random_nu":
            self.input_size, output_size = 2, 1
        elif model == "ml_step":
            self.input_size, output_size = 12, 9

            if nu_range is None:
                nu_range = (self.nu.value.min(), self.nu.value.max())

            self.log10_nu0_min = nu_range[0]
            self.log10_nu0_max = nu_range[1]
            self.log10_T_min = temperature_range[0]
            self.log10_T_max = temperature_range[1]
            self.log10_tau_cell_nu0_min = tau_range[0]
            self.log10_tau_cell_nu0_max = tau_range[1]

        self.initialize_model(model=model, input_size=self.input_size, output_size=output_size, 
                num_hidden_layers=num_hidden_layers, num_hidden_units=num_hidden_units)
        self.gen_model = getattr(self, model+"_model")
        self.disc_model = Discriminator(input_size=self.input_size, num_hidden_layers=num_hidden_layers, num_hidden_units=num_hidden_units).to(self.device)

        # Wrap the model in lightning

        self.setup()

        self.g_optimizer = torch.optim.Adam(self.gen_model.parameters(), lr=0.0001)
        self.d_optimizer = torch.optim.Adam(self.disc_model.parameters(), lr=0.0001)

        self.loss_fn = nn.BCELoss()

        all_d_losses = []
        all_g_losses = []
        all_d_real = []
        all_d_fake = []
        
        for epoch in range(1, max_epochs + 1):
            d_losses = []
            g_losses = []
            d_vals_real, d_vals_fake = [], []

            for i, (x, l) in enumerate(self.train):
                d_loss, d_proba_real, d_proba_fake = self.d_train(x, l)
                d_losses.append(d_loss)
                g_losses.append(self.g_train(x, l))

                d_vals_real.append(d_proba_real.mean().cpu())
                d_vals_fake.append(d_proba_fake.mean().cpu())

            all_d_losses.append(torch.tensor(d_losses).mean())
            all_g_losses.append(torch.tensor(g_losses).mean())

            all_d_real.append(torch.tensor(d_vals_real).mean())
            all_d_fake.append(torch.tensor(d_vals_fake).mean())

            print(f'Epoch {epoch:03d} | Avg Losses >> D: {all_d_losses[-1]:.4f} | G: {all_g_losses[-1]:.4f}')

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
        y = torch.unsqueeze(torch.tensor(np.log10(y.value), dtype=torch.float32), 1)

        X = torch.unsqueeze(torch.tensor(logT, dtype=torch.float32), 1)

        self.df = pd.DataFrame({"log10_T":logT, "log10_nu":y.numpy().flatten()})
        self.features = ["log10_nu"]
        self.labels = ["log10_T"]

        self.random_nu_scaler = MinMaxScaler()
        data = self.random_nu_scaler.fit_transform(self.df.values)
        self.df = pd.DataFrame(data, columns=self.df.columns)

        X = torch.tensor(self.df.loc[:, self.features].values, dtype=torch.float32)
        y = torch.tensor(self.df.loc[:, self.labels].values, dtype=torch.float32)

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

        df.loc[df["log10_tau"] < -5., "log10_tau"] = -5.
        df.loc[np.isnan(df["log10_tau"]), "log10_tau"] = -5.
        df.loc[df["log10_Eabs"] < -7., "log10_Eabs"] = -7.

        self.scaler = MinMaxScaler()
        data = self.scaler.fit_transform(df.values)
        df = pd.DataFrame(data, columns=df.columns)

        self.df = df
        self.nsamples = len(df)

        self.features = ["log10_nu", "log10_Eabs", "log10_tau", "yaw", "pitch", "roll", "direction_yaw", "direction_pitch", "direction_roll"]
        self.labels = ["log10_nu0", "log10_T", "log10_tau_cell_nu0"]

        X = torch.tensor(df.loc[:, self.features].values, dtype=torch.float32)
        y = torch.tensor(df.loc[:, self.labels].values, dtype=torch.float32)

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
                       "log10_Eabs":np.log10(np.where(photon_list.deposited_energy.numpy() > 0, photon_list.deposited_energy.numpy(), photon_list.deposited_energy.numpy().min()/100)/photon_list.energy.numpy()),
                       "log10_tau":np.log10(photon_list.tau.numpy().copy()),
                       "yaw":ypr[:,0],
                       "pitch":ypr[:,1],
                       "roll":ypr[:,2],
                       #"direction_theta":np.acos((photon_list.position.numpy() * photon_list.direction.numpy()).sum(axis=1))})
                       "direction_yaw":direction_ypr[:,0],
                       "direction_pitch":direction_ypr[:,1],
                       "direction_roll":direction_ypr[:,2]})

        return df
    
    def d_train(self, x, labels):
        self.disc_model.zero_grad()

        batch_size = x.size(0)
        x = x.view(batch_size, -1).to(self.device)

        d_labels_real = torch.ones(batch_size, 1, device=self.device)
        d_proba_real = self.disc_model(x, labels)
        d_loss_real = self.loss_fn(d_proba_real, d_labels_real)

        input_z = create_noise(batch_size, self.input_size - len(self.labels)).to(self.device)
        fake_labels = torch.rand(batch_size, labels.size(1), device=self.device)
        g_output = self.gen_model(input_z, fake_labels)

        d_proba_fake = self.disc_model(g_output, fake_labels)
        d_labels_fake = torch.zeros(batch_size, 1, device=self.device)
        d_loss_fake = self.loss_fn(d_proba_fake, d_labels_fake)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.data.item(), d_proba_real.detach(), d_proba_fake.detach()

    def g_train(self, x, labels):
        self.gen_model.zero_grad()

        batch_size = x.size(0)

        input_z = create_noise(batch_size, self.input_size - len(self.labels)).to(self.device)
        fake_labels = torch.rand(batch_size, labels.size(1), device=self.device)

        g_output = self.gen_model(input_z, fake_labels)

        d_proba_fake = self.disc_model(g_output, fake_labels)

        g_labels_real = torch.ones(batch_size, 1, device=self.device)
        g_loss = self.loss_fn(d_proba_fake, g_labels_real)

        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.data.item()

    # DataModule functions

    def plot_learned_model(self):
        import matplotlib.pyplot as plt

        samples = create_samples(getattr(self, self.learning+"_model"), create_noise(len(self.df), self.input_size - len(self.labels)).to(self.device), 
                torch.tensor(self.df.loc[:, self.labels].values, dtype=torch.float32).to(self.device)).detach().cpu().numpy()

        df_gen = pd.DataFrame(samples, columns=self.features)
        for label in self.labels:
            df_gen[label] = self.df[label].values

        fig, ax = plt.subplots(nrows=len(self.df.columns), ncols=len(self.df.columns), figsize=(11,11))

        for i, key1 in enumerate(self.df.columns):
            for j, key2 in enumerate(self.df.columns):
                if key1 == key2:
                    ax[i,j].hist(self.df[key1], bins=50, histtype='step', density=True)
                    ax[i,j].hist(df_gen[key1], bins=50, histtype='step', density=True)
                elif i > j:
                    ax[i,j].scatter(self.df[key2], self.df[key1], marker='.', s=0.025)
                    ax[i,j].scatter(df_gen[key2], df_gen[key1], marker='.', s=0.025)
                elif i < j:
                    ax[i,j].set_axis_off()

                if i == len(self.df.columns) - 1:
                    ax[i,j].set_xlabel(key2)

            ax[i,0].set_ylabel(key1)

        plt.show()
        plt.savefig("ml_step_results.png", dpi=300)

    def setup(self):
        if hasattr(self, "train") and hasattr(self, "test"):
            return
        if not hasattr(self, "dataset"):
            self.prepare_data()

        test_size = int(self.test_split * self.nsamples)
        train_size = self.nsamples - test_size
        self.train, self.test = random_split(self.dataset, [train_size, test_size], generator=torch.Generator().manual_seed(1))
        self.train = DataLoader(self.train, batch_size=32, shuffle=False)

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

class Generator(nn.Module):
    def __init__(self, input_size=3, num_hidden_layers=3, num_hidden_units=48, num_output_units=3):
        super().__init__()
        self.model = nn.Sequential()
        for i in range(num_hidden_layers):
            self.model.add_module(f'fc_g{i}', nn.Linear(input_size, num_hidden_units, bias=False))
            self.model.add_module(f'bn_g{i}', nn.BatchNorm1d(num_hidden_units))
            self.model.add_module(f'relu_g{i}', nn.LeakyReLU(0.2))
            input_size = num_hidden_units
        self.model.add_module(f'fc_g{num_hidden_layers}', nn.Linear(input_size, num_output_units))
        self.model.add_module(f'sigmoid_g', nn.Sigmoid())

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_size=3, num_hidden_layers=3, num_hidden_units=48, num_output_units=1):
        super().__init__()
        self.model = nn.Sequential()
        for i in range(num_hidden_layers):
            self.model.add_module(f'fc_d{i}', nn.Linear(input_size, num_hidden_units, bias=False))
            self.model.add_module(f'bn_d{i}', nn.BatchNorm1d(num_hidden_units))
            self.model.add_module(f'relu_d{i}', nn.LeakyReLU(0.2))
            self.model.add_module('dropout', nn.Dropout(p=0.5))
            input_size = num_hidden_units
        self.model.add_module(f'fc_d{num_hidden_layers}', nn.Linear(input_size, num_output_units))
        self.model.add_module(f'sigmoid', nn.Sigmoid())

    def forward(self, x, labels):
        x = torch.cat([x, labels], dim=1)
        return self.model(x)

def create_noise(batch_size, input_size):
    return torch.randn(batch_size, input_size)


def create_samples(g_model, input_z, input_labels):
    g_output = g_model(input_z, input_labels)
    return g_output


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
