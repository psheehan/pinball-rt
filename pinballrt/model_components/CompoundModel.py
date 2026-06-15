from .Fittable_Model import Fittable_Model
from ..grids import Grid
import numpy as np
import astropy.units as u

class CompoundModel(Fittable_Model):
    def __init__(self, model1, model2, grid: Grid, grid_kwargs={}, ncores=1, mpi=False):
        super().__init__(grid, grid_kwargs=grid_kwargs, ncores=ncores, mpi=mpi)

        self.model1 = model1
        self.model2 = model2
        self.model_parameters = {model1.model_name: model1.model_params,
                                 model2.model_name: model2.model_params}

    def set_parameters(self, compound_model_params={}, dusttogasratio=0.01, dust=None, amax=None,
                       p=None, dust_abundances=(), gases=None, abundances=None, velocity=None,
                       microturbulence=None):
        
        self.model1.set_attributes(compound_model_params.get(self.model1.model_name, {}))
        model1_params = self.model1.model_params
        self.model2.set_attributes(compound_model_params.get(self.model2.model_name, {}))
        model2_params = self.model2.model_params

        density_grid1 = self.model1.density_grid(model1_params)
        density_grid2 = self.model2.density_grid(model2_params)

        total_density_grid = density_grid1 + density_grid2

        self.set_physical_properties(total_density_grid, dusttogasratio, dust, amax,
                                     p, dust_abundances, gases, abundances, velocity, microturbulence)

