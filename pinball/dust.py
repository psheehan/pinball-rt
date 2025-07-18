from .sources import Star
from .grids import UniformSphericalGrid
from torch.utils.data import DataLoader, TensorDataset, random_split
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

class Dust(pl.LightningDataModule):
    def __init__(self, lam, kabs, ksca, interpolate=10000):
        super().__init__()
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

        self.temperature = np.logspace(-1.,4.,1000)
        self.log_temperature = np.log10(self.temperature)

        random_nu_PDF = np.array([kabs * models.BlackBody(temperature=T*u.K)(self.nu) for T in self.temperature])
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

    def learn(self, model="random_nu", nsamples=200000, test_split=0.1, valid_split=0.2, hidden_units=(48, 48, 48), max_epochs=10, plot=False):

        self.nsamples = nsamples
        self.test_split = test_split
        self.valid_split = valid_split
        self.learning = model

        # Set up the NN

        if model == "random_nu":
            input_size, output_size = 2, 1
        elif model == "ml_step":
            input_size, output_size = 12, 9

        self.initialize_model(model=model, input_size=input_size, output_size=output_size, hidden_units=hidden_units)

        # Wrap the model in lightning

        dustLM = DustLightningModule(getattr(self, model+"_model"))

        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(model=dustLM, datamodule=self)

        # Test the model.

        trainer.test(model=dustLM, datamodule=self)

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
                y_pred = trainer.predict(dustLM, datamodule=self)
                y_pred = torch.cat(y_pred)
                y_true = torch.cat([batch[1] for batch in self.predict_dataloader()])

                X_test = torch.cat([batch[0] for batch in self.predict_dataloader()])

                columns = self.df.columns
                features = np.array(["log10_nu0", "log10_T", "log10_tau_cell_nu0", "ksi_log10_nu", "ksi_log10_Eabs", "ksi_tau", "ksi_yaw", "ksi_pitch", "ksi_roll", "ksi_direction_yaw", "ksi_direction_pitch", "ksi_direction_roll"])
                targets = np.array(["log10_nu", "log10_Eabs", "tau", "yaw", "pitch", "roll", "direction_yaw", "direction_pitch", "direction_roll"])

                fig, ax = plt.subplots(nrows=len(columns), ncols=len(columns), figsize=(11,11))

                for i, key1 in enumerate(columns):
                    for j, key2 in enumerate(columns):
                        if key1 == key2:
                            ax[i,j].hist(self.df[key1], bins=50, histtype='step', density=True)
                            if key1 in targets:
                                ax[i,j].hist(y_true.numpy()[:,np.where(targets == key1)[0][0]], bins=50, histtype='step', density=True)
                        elif i > j:
                            ax[i,j].scatter(self.df[key2], self.df[key1], marker='.', s=0.1)

                            if key1 in targets or key2 in targets:
                                if key2 in targets:
                                    x = y_true.numpy()[:,np.where(targets == key2)[0][0]]
                                else:
                                    x = X_test.numpy()[:,np.where(features == key2)[0][0]]

                                if key1 in targets:
                                    y = y_true.numpy()[:,np.where(targets == key1)[0][0]]
                                else:
                                    y = X_test.numpy()[:,np.where(features == key1)[0][0]]

                                ax[i,j].scatter(x, y, marker='.', s=0.1)
                        elif i < j:
                            ax[i,j].set_axis_off()

                plt.show()

    def random_nu(self, temperature):
        nphotons = temperature.size
        ksi = np.random.rand(int(nphotons))

        test_x = torch.tensor(np.vstack((np.log10(temperature), ksi)).T, dtype=torch.float32)

        nu = 10.**self.random_nu_model(test_x).detach().numpy().flatten()

        return nu

    def planck_mean_opacity(self, temperature):
        vectorized_bb = np.vectorize(lambda T: self.kmean.cgs.value * scipy.integrate.trapezoid(self.kabs * \
                models.BlackBody(temperature=T*u.K)(self.nu).cgs.value, self.nu.to(u.Hz).value))

        return np.pi / (const.sigma_sb.cgs.value * temperature**4) * vectorized_bb(temperature)

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

    def run_dust_simulation(self, nphotons=1000, tau_range=(0.5, 4.0), temperature_range=(-1.0, 4.0), nu_range=None):
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

        photon_list.frequency = wp.array(10.**np.random.uniform(np.log10(nu_range[0]), np.log10(nu_range[1]), nphotons), dtype=float)
        original_frequency = photon_list.frequency.numpy().copy()

        photon_list.temperature = wp.array(10.**np.random.uniform(temperature_range[0], temperature_range[1], nphotons), dtype=float)

        tau = 10.**np.random.uniform(tau_range[0], tau_range[1], nphotons)
        photon_list.density = wp.array((tau / (self.kmean * self.interpolate_kabs(photon_list.frequency.numpy()*u.GHz) * 1.*u.au) * self.kmean).to(1 / u.au), dtype=float)

        grid.propagate_photons(photon_list, learning=True, use_ml_step=False)

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

    def prepare_data_random_nu(self):
        sampler = scipy.stats.qmc.LatinHypercube(d=2)
        samples = sampler.random(self.nsamples)

        logT = 5*samples[:,0] - 1.
        ksi = samples[:,1]
        X = torch.tensor(np.vstack((logT, ksi)).T, dtype=torch.float32)

        y = np.concatenate([self.random_nu_manual(10.**X_batch[:,0].numpy(), X_batch[:,1].numpy()) for X_batch in DataLoader(X, batch_size=1000)])
        y = torch.tensor(np.log10(y.value), dtype=torch.float32)

        self.dataset = TensorDataset(X, y)

    def prepare_data_ml_step(self):
        if hasattr(self, "dataset"):
            return

        if os.path.exists("sim_results.csv"):
            df = pd.read_csv("sim_results.csv", index_col=0)
        else:
            df = self.run_dust_simulation(nphotons=self.nsamples)
            df.to_csv("sim_results.csv")

        self.df = df
        self.nsamples = len(df)

        base_features = ["log10_nu0", "log10_T", "log10_tau_cell_nu0"]

        new_columns = []
        for j in range(3,df.shape[1]):
            ksi = []

            column = df.columns[j]

            kde = gaussian_kde(df.loc[:, base_features + [column]].values.T)
            x = np.linspace(df.loc[:, column].values.min(), df.loc[:, column].values.max(), 1000)

            for i in trange(df.shape[0]):
                X = np.vstack(tuple(np.repeat(df.loc[i, col], 1000) for col in base_features) + (x,))

                pdf = kde(X)

                cdf = np.cumsum(pdf)
                cdf /= cdf.max()

                interp = scipy.interpolate.interp1d(x, cdf, kind='cubic')
                ksi.append(float(interp(df.loc[i, column])))

            new_columns.append(("ksi_"+column, ksi))

            base_features.append(column)

        for name, values in new_columns:
            df[name] = values

        features = ["log10_nu0", "log10_T", "log10_tau_cell_nu0", "ksi_log10_nu", "ksi_log10_Eabs", "ksi_tau", "ksi_yaw", "ksi_pitch", "ksi_roll", "ksi_direction_yaw", "ksi_direction_pitch", "ksi_direction_roll"]
        targets = ["log10_nu", "log10_Eabs", "tau", "yaw", "pitch", "roll", "direction_yaw", "direction_pitch", "direction_roll"]

        X = torch.tensor(df.loc[:, features].values, dtype=torch.float32)
        y = torch.tensor(df.loc[:, targets].values, dtype=torch.float32)

        self.dataset = TensorDataset(X, y)

    def prepare_data(self):
        if self.learning == "random_nu":
            self.prepare_data_random_nu()
        elif self.learning == "ml_step":
            self.prepare_data_ml_step()

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

    def save(self, filename):
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

        torch.save(state_dict, filename)


def load(filename):
    state_dict = torch.load(filename, weights_only=False)

    d = Dust(**state_dict["dust_properties"], interpolate=-1 )

    if "random_nu_state_dict" in state_dict:
        hidden_units = [state_dict['random_nu_state_dict'][key].shape[0] for key in state_dict['random_nu_state_dict'] if 'bias' in key][0:-1]
        d.initialize_model(model="random_nu", input_size=2, output_size=1, hidden_units=hidden_units)

        d.random_nu_model.load_state_dict(state_dict['random_nu_state_dict'])

    if "ml_step_state_dict" in state_dict:
        hidden_units = [state_dict['ml_step_state_dict'][key].shape[0] for key in state_dict['ml_step_state_dict'] if 'bias' in key][0:-1]
        d.initialize_model(model="ml_step", input_size=12, output_size=9, hidden_units=hidden_units)

        d.ml_step_model.load_state_dict(state_dict['ml_step_state_dict'])

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
        loss = nn.functional.mse_loss(self(x), y.reshape(-1,1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y.reshape(-1,1))
        self.log('valid_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y.reshape(-1,1))
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)