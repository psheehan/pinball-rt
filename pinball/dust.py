from astropy.modeling import models
import astropy.units as u
import astropy.constants as const
import scipy.stats.qmc
import matplotlib.pyplot as plt
import scipy.integrate
import warp as wp
import numpy as np
import torch

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