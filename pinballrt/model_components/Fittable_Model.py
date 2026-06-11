from ..model import Model
from ..grids import Grid
import numpy as np
import astropy.units as u

class Fittable_Model(Model):
    default_params = {}
    def __init__(self, grid: Grid, grid_kwargs={}, ncores=1, mpi=False, model_params={}):
        super().__init__(grid, grid_kwargs=grid_kwargs, ncores=ncores, mpi=mpi)

        self.model_params = self.default_params.copy()
        self.set_attributes(model_params)
        self.model_name = "generic_fittable_model"

    def density(self):
        raise NotImplementedError("The density method must be implemented in the subclass.")
    
    def set_attributes(self, model_params):
        for key, value in model_params.items():
            setattr(self, key, value)
            self.model_params[key] = value

    def set_parameters(self, model_params={}, dusttogasratio=0.01, dust=None, amax=None,
                       p=None, dust_abundances=(), gases=None, abundances=None, velocity=None,
                       microturbulence=None, density_coordinates="cylindrical"):
        
        self.set_attributes(model_params)

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
        if density_coordinates == "cylindrical":
            density_grid = self.density(rcyl, zz)

        self.set_physical_properties(density_grid, dusttogasratio, dust, amax, p, dust_abundances,
                                     gases, abundances, velocity, microturbulence)