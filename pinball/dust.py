from astropy.modeling import models
import astropy.units as u
import astropy.constants as const
import scipy.stats.qmc
import matplotlib.pyplot as plt
import scipy.integrate
import warp as wp
import numpy as np
import torch.nn as nn

from skorch import NeuralNetRegressor
from skorch.dataset import Dataset
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import torch

class Dust:
    def __init__(self, lam, kabs, ksca):
        kunit = kabs.unit
        lam_unit = lam.unit

        f_kabs = scipy.interpolate.interp1d(np.log10(lam.value), np.log10(kabs.value), kind="cubic")
        f_ksca = scipy.interpolate.interp1d(np.log10(lam.value), np.log10(ksca.value), kind="cubic")

        lam = 10.**np.linspace(np.log10(lam.value).min(), np.log10(lam.value).max(), 10000)[::-1]
        kabs = 10.**f_kabs(np.log10(lam))
        ksca = 10.**f_ksca(np.log10(lam))

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

    def learn_random_nu(self, nsamples=200000, max_epochs=10, plot=False):
        sampler = scipy.stats.qmc.LatinHypercube(d=2)
        X = sampler.random(nsamples).astype(np.float32)
        X[:,0] = 10.**(5*X[:,0] - 1.)

        y = np.concatenate([self.random_nu_manual(X_batch[:,0].numpy(), X_batch[:,1].numpy()) for X_batch, _ in DataLoader(Dataset(X), batch_size=1000)]).astype(np.float32)

        if plot:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        else:
            X_train, y_train = X, y

        net = NeuralNetRegressor(
                MultiLayerPerceptron,
                max_epochs=max_epochs,
                criterion=nn.MSELoss,
                optimizer=torch.optim.Adam,
                lr=0.01,
                # Shuffle training data on each epoch
                iterator_train__shuffle=True)
        
        pipe = TransformedTargetRegressor(
            regressor=Pipeline([
                ('LogT', ColumnTransformer([
                    ('T', FunctionTransformer(func=np.log10, inverse_func=ten_to_the_x, check_inverse=False), [0]),
                    ('ksi', 'passthrough', [1])])),
                ('Normalize', MinMaxScaler()),
                ('MLP', net)]),
            transformer=Pipeline([
                ('log10', FunctionTransformer(func=np.log10, inverse_func=ten_to_the_x, check_inverse=False)),
                ('Normalize', MinMaxScaler())]))
                
        pipe.fit(X_train, y_train)

        if plot:
            count, bins, patches = plt.hist(np.log10(y_test.value), 100, histtype='step')
            plt.hist(np.log10(pipe.predict(X_test)), bins, histtype='step')
            plt.savefig("predicted_vs_actual.png")
            plt.clf()
            plt.close()

        self.model = pipe

    def random_nu(self, temperature):
        nphotons = temperature.size
        ksi = np.random.rand(int(nphotons))

        X_pred = np.vstack((temperature, ksi)).T.astype(np.float32)

        return self.model.predict(X_pred)

    def planck_mean_opacity(self, temperature):
        vectorized_bb = np.vectorize(lambda T: self.kmean.cgs.value * scipy.integrate.trapezoid(self.kabs * \
                models.BlackBody(temperature=T*u.K)(self.nu).cgs.value, self.nu.to(u.Hz).value))

        return np.pi / (const.sigma_T.cgs.value * temperature**4) * vectorized_bb(temperature)
    
    def write(self, filename):
        import pickle
        pickle.dump(self, open(filename, "wb"))

def load(filename):
    import pickle
    return pickle.load(open(filename, "rb")) 

def ten_to_the_x(x):
    return 10.**x

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size=2, nunits=48, nlayers=3):
        super().__init__()

        all_layers = []
        for hidden_unit in [nunits]*nlayers:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.Sigmoid())
            input_size = hidden_unit

        all_layers.append(nn.Linear(nunits, 1))

        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.model(x)[:,0]
        return x