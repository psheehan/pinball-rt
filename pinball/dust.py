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

    def initialize_model(self, hidden_units=(48, 48, 48)):
        all_layers = [nn.Flatten()]

        input_size = 2

        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.Sigmoid())
            input_size = hidden_unit

        all_layers.append(nn.Linear(hidden_units[-1], 1))

        self.model = nn.Sequential(*all_layers)

    def learn_random_nu(self, nsamples=200000, test_split=0.1, valid_split=0.2, hidden_units=(48, 48, 48), max_epochs=10, plot=False):

        self.nsamples = nsamples
        self.test_split = test_split
        self.valid_split = valid_split

        # Set up the NN

        self.initialize_model(hidden_units=hidden_units)

        # Wrap the model in lightning

        dustLM = DustLightningModule(self.model)

        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(model=dustLM, datamodule=self)

        # Test the model.

        trainer.test(model=dustLM, datamodule=self)

        # Plot the result

        if plot:
            import matplotlib.pyplot as plt

            y_pred = trainer.predict(dustLM, datamodule=self)
            y_pred = torch.cat(y_pred)
            y_true = torch.cat([batch[1] for batch in self.predict_dataloader()])
            
            with torch.no_grad():
                count, bins, patches = plt.hist(y_true, 100, histtype='step')
                plt.hist(y_pred, bins, histtype='step')
                plt.savefig("predicted_vs_actual.png")

    def random_nu(self, temperature):
        nphotons = temperature.size
        ksi = np.random.rand(int(nphotons))

        test_x = torch.tensor(np.vstack((np.log10(temperature), ksi)).T, dtype=torch.float32)

        nu = 10.**self.model(test_x).detach().numpy().flatten()

        return nu

    def planck_mean_opacity(self, temperature):
        vectorized_bb = np.vectorize(lambda T: self.kmean.cgs.value * scipy.integrate.trapezoid(self.kabs * \
                models.BlackBody(temperature=T*u.K)(self.nu).cgs.value, self.nu.to(u.Hz).value))

        return np.pi / (const.sigma_T.cgs.value * temperature**4) * vectorized_bb(temperature)

    # DataModule functions

    def prepare_data(self):
        sampler = scipy.stats.qmc.LatinHypercube(d=2)
        samples = sampler.random(self.nsamples)

        logT = 5*samples[:,0] - 1.
        ksi = samples[:,1]
        X = torch.tensor(np.vstack((logT, ksi)).T, dtype=torch.float32)
        
        y = np.concatenate([self.random_nu_manual(10.**X_batch[:,0].numpy(), X_batch[:,1].numpy()) for X_batch in DataLoader(X, batch_size=1000)])
        y = torch.tensor(np.log10(y.value), dtype=torch.float32)

        self.dataset = TensorDataset(X, y)

    def setup(self, stage=None):
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
        
        if hasattr(self, "model"):
            state_dict["model_state_dict"] = self.model.state_dict()

        torch.save(state_dict, filename)



def load(filename):
    state_dict = torch.load(filename, weights_only=False)

    d = Dust(**state_dict["dust_properties"], interpolate=-1 )

    if "model_state_dict" in state_dict:
        hidden_units = [state_dict['model_state_dict'][key].shape[0] for key in state_dict['model_state_dict'] if 'bias' in key][0:-1]
        d.initialize_model(hidden_units=hidden_units)

        d.model.load_state_dict(state_dict['model_state_dict'])

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