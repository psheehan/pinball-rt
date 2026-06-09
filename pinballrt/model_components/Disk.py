from ..model import Model
from ..grids import Grid
import numpy as np
import astropy.units as u

class Disk(Model):
    def __init__(self, grid: Grid, grid_kwargs={}, ncores=1, mpi=False, model_params={}):
        super().__init__(grid, grid_kwargs=grid_kwargs, ncores=ncores, mpi=mpi)

        self.mass = model_params.get("mass", 1e-3 * u.Msun)
        self.rin = model_params.get("rin", 0.1 * u.au)
        self.rout = model_params.get("rout", 100.0 * u.au)
        self.gamma = model_params.get("gamma", 1.0)
        self.h_0 = model_params.get("h_0", 0.05 * u.au)
        self.beta = model_params.get("beta", 1.0)

        self.model_name = "Disk"

    def surface_density(self, r):
        sigma0 = ((2.0 - self.gamma) * self.mass / (2.0 * np.pi * self.rout**2))
        sigma = sigma0 * (r / self.rout)**(-self.gamma) * np.exp(-(r / self.rout)**(2.0 - self.gamma))
        return sigma.to(u.g / u.cm**2)
    
    def scale_height(self, r):
        h = self.h_0 * (r / (1*u.au))**self.beta
        return h.to(u.au)
    
    def density(self, r, z):
        sigma = self.surface_density(r)
        h = self.scale_height(r)
        rho = sigma / (np.sqrt(2 * np.pi) * h) * np.exp(-0.5 * (z / h)**2)
        return rho.to(u.g / u.cm**3)
    
    def set_parameters(self, model_params={}, **params):

        self.mass = model_params.get("mass", 1e-3 * u.Msun)
        self.rin = model_params.get("rin", 0.1 * u.au)
        self.rout = model_params.get("rout", 100.0 * u.au)
        self.gamma = model_params.get("gamma", 1.0)
        self.h_0 = model_params.get("h_0", 0.05 * u.au)
        self.beta = model_params.get("beta", 1.0)

        if self.grid.coordinate_system == "cartesian":
            # get bin edges and centers
            x_edges = self.grid.grid.w1.numpy() * u.au
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_edges = self.grid.grid.w2.numpy() * u.au
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            z_edges = self.grid.grid.w3.numpy() * u.au
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
            
            # create 3D meshgrid
            xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

            # get cylindrical radius
            rcyl = np.sqrt(xx**2 + yy**2)

        elif self.grid.coordinate_system == "spherical":
            # get bin edges and centers
            r_edges = self.grid.grid.w1.numpy() * u.au
            r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
            theta_edges = self.grid.grid.w2.numpy()
            theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
            phi_edges = self.grid.grid.w3.numpy()
            phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
            
            # create 3D meshgrid
            rr, tt, pp = np.meshgrid(r_centers, theta_centers, phi_centers, indexing='ij')

            # get cylindrical coordinates
            rcyl = rr * np.sin(tt)
            zz = rr * np.cos(tt)
        
        # compute density grid
        density_grid = self.density(rcyl, zz)

        self.set_physical_properties(density=density_grid, **params)

    