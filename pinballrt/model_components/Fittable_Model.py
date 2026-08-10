import copy
import warnings

import astropy.units as u
import numpy as np
import pandas as pd

from ..dust import load
from ..grids import LogUniformSphericalGrid, UniformCartesianGrid, UniformSphericalGrid
from ..model import Model

# class FittableModelMeta(type):
#     def __add__(cls, other_cls):
#         class CompoundModel(Fittable_Model):
#             default_params = cls.default_params + other_cls.default_params
#             def __init__(self, grid, ncores=1, mpi=False):
#                 super().__init__(grid, ncores=ncores, mpi=mpi)

#             def _calculate_density_grid(self):
#                 mgrid1 = self._get_density_meshgrid(cls.density_coordinates)
#                 mgrid2 = self._get_density_meshgrid(other_cls.density_coordinates)

#                 density_grid1 = cls.density(self, *mgrid1)
#                 density_grid2 = other_cls.density(self, *mgrid2)

#                 return density_grid1 + density_grid2

#         return CompoundModel


class Fittable_Model(Model):
    model_name = "fittable_model"
    density_coordinates = "cylindrical"
    model_default_params = pd.DataFrame([])

    def __init__(self, grid, components=None, params=None):

        # dust_default_params = pd.DataFrame.from_dict({
        #     'dusttogasratio': {'value': 0.01, 'component':'dust'},
        #     'dust_file': {'value': 'diana_wice.dst', 'component':'dust'},
        #     'amax': {'value': 1.0, 'component':'dust', 'unit': u.cm},
        #     'p': {'value': 3.0, 'component':'dust'},
        #     'dust_abundances': {'value': [], 'component':'dust'}}, orient='index')

        # gas_default_params = pd.DataFrame.from_dict({
        #     'gases': {'value': None, 'component':'gas'},
        #     'abundances': {'value': None, 'component':'gas'},
        #     'velocity': {'value': None, 'component':'dust'},
        #     'microturbulence': {'value': None, 'component':'gas'}}, orient='index')

        hyper_default_params = pd.DataFrame.from_dict(
            {
                "mpi": {"value": True, "component": "hyper"},
                "ncores": {"value": 1, "component": "hyper"},
            },
            orient="index",
        )

        self.parameters = hyper_default_params.copy()
        for component in components:
            self.parameters = pd.concat([self.parameters, component.parameters])

        # self.parameters = self.default_parameters.copy()
        # if isinstance(params, pd.DataFrame):
        #     self._update_parameters_from_dataframe(df=params)

        # if self.coord_sys == 'uniform_cartesian':
        #     grid = UniformCartesianGrid(ncells=self.ncells, dx=self.dx, device=self.device)
        #     self.grid_unit = self.parameters.loc['dx', 'unit']
        # elif self.coord_sys == 'uniform_spherical':
        #     grid = UniformSphericalGrid(ncells=self.ncells, dr=self.dr, mirror=self.mirror,
        #                                       device=self.device)
        #     self.grid_unit = self.parameters.loc['dr', 'unit']
        # elif self.coord_sys == 'log_uniform_spherical':
        #     grid = LogUniformSphericalGrid(ncells=self.ncells, rmin=self.grid_rmin, rmax=self.grid_rmax,
        #                                          mirror=self.mirror, device=self.device)
        #     self.grid_unit = self.parameters.loc['rmin', 'unit']

        super().__init__(grid, ncores=self.ncores, mpi=self.mpi)
        self.update_parameters()

    # def __getattr__(self, param_name):
    #     if param_name.startswith('_'):
    #         raise AttributeError(param_name)

    #     matches =  self.parameters.loc[self.parameters.name == param_name]
    #     if matches.empty:
    #         raise KeyError(f"No parameter named '{param_name}'")
    #     elif len(matches) > 1:
    #         warnings.warn("Multiple parameters with name '{param_name}'")
    #         return matches
    #     elif len(matches) == 1:
    #         return self.get_param_quantity(param_name)

    def __getattr__(self, param_name):
        value = self.parameters.loc[param_name, "value"]
        unit = self.parameters.loc[param_name, "unit"]
        log = self.parameters.loc[param_name, "log"]

        if pd.isna(unit):
            if log == True:
                return 10**value
            else:
                return value
        else:
            if log == True:
                return 10**value * unit
            else:
                return value * unit

    @property
    def free_parameters(self):
        return self.parameters.loc[self.parameters["fixed"] == False]

    # def param_lookup(self, param_name, attr, component=None):
    #     if component:
    #         matches = self.parameters.loc[(self.parameters['name'] == param_name) &
    #                                        (self.parameters['component'] == component), attr]

    #         if matches.empty:
    #             raise KeyError(f"No parameter named '{name}', in component '{component}'")
    #         elif len(matches) > 1:
    #             raise KeyError(f"Ambiguous, repeated parameter")
    #         else:
    #             return matches.iloc[0]
    #     else:
    #         matches = self.parameters.loc[self.parameters['name'] == param_name, attr]

    #         if matches.empty:
    #             raise KeyError(f"No parameter named '{name}'")
    #         elif len(matches) > 1:
    #             raise KeyError(f"Ambiguous, please specify component")
    #         else:
    #             return matches.iloc[0]

    # def get_param_quantity(self, param_name, component=None):
    #     value = self.param_lookup(param_name, 'value', component=component)
    #     unit = self.param_lookup(param_name, 'unit', component=component)
    #     log = self.param_lookup(param_name, 'log', component=component)

    #     if pd.isna(unit):
    #         if log == True:
    #             return 10**value
    #         else:
    #             return value
    #     else:
    #         if log == True:
    #             return 10**value * unit
    #         else:
    #             return value * unit

    # def update_param(self, param_name, attr, new_value, component=None):
    #     if component:
    #         matches = self.parameters.loc[(self.parameters['name'] == param_name) &
    #                                       (self.parameters['component'] == component), attr]

    #         if matches.empty:
    #             raise KeyError(f"No parameter named '{name}', in component '{component}'")
    #         elif len(matches) > 1:
    #             raise KeyError(f"Ambiguous, repeated parameter")
    #         else:
    #             self.parameters.loc[(self.parameters['name'] == param_name) &
    #                                 (self.parameters['component'] == component), attr] = new_value
    #     else:
    #         matches = self.parameters.loc[self.parameters['name'] == param_name, attr]

    #         if matches.empty:
    #             raise KeyError(f"No parameter named '{name}'")
    #         elif len(matches) > 1:
    #             raise KeyError(f"Ambiguous, please specify component")
    #         else:
    #             self.parameters.loc[self.parameters['name'] == param_name, attr] = new_value

    def update_param(self, param_name, attr, new_value):
        if param_name not in self.parameters.index:
            raise KeyError(f"No parameter named '{param_name}'")
        else:
            self.parameters.loc[param_name, attr] = new_value

    # def _update_parameters_from_dict(self, param_dict, component=None):
    #     if 'name' not in param_dict.keys():
    #         raise KeyError("Need to specify parameter name in dictionary")
    #     param_name = param_dict['name']
    #     attrs = [attr for attr in param_dict.keys() if attr != 'name']
    #     for attr in attrs:
    #         value = param_dict[attr]
    #         self.update_param(param_name, attr, value, component=component)

    def _update_parameters_from_dict(self, param_dict):
        if "name" not in param_dict.keys():
            raise KeyError("Need to specify parameter name in dictionary")
        param_name = param_dict["name"]
        attrs = [attr for attr in param_dict.keys() if attr != "name"]
        for attr in attrs:
            value = param_dict[attr]
            self.update_param(param_name, attr, value)

    def _update_parameters_from_keywords(self, **kwargs):
        for param_name, value in kwargs.items():
            self.update_param(param_name, "value", value)

    # def _update_parameters_from_keywords(self, component=None, **kwargs):
    #     for param_name, value in kwargs.items():
    #         self.update_param(param_name, "value", value, component=component)

    def _update_parameters_from_array(self, param_arr):
        if len(self.free_parameters.index) != len(param_arr):
            raise KeyError(
                "Number of input parameters does not match number of free parameters"
            )
        free_param_names = m.free_parameters.index
        for i in range(len(param_arrs)):
            self.update_param(free_param_names[i], "value", param_arr[i])

    # def _update_parameters_from_array(self, param_arr):
    #     if len(self.free_parameters.index) != len(param_arr):
    #         raise KeyError("Number of input parameters does not match number of free parameters")
    #     else:
    #         param_inds = m.free_parameters.index
    #         for i in range(len(param_arrs)):
    #             m.parameters.loc[param_inds[i], "value"] = param_arr[i]

    def _update_parameters_from_dataframe(self, df):
        self.parameters.update(df)

    # def _update_parameters_from_dataframe(self, df):
    #     df_dict = df.to_dict(orient='list')
    #     nrow, ncol = df.shape
    #     attrs = [attr for attr in df.columns if attr != 'name']
    #     for i in range(nrow):
    #         if 'component' in attrs:
    #             component = df_dict['component'][i]
    #         else:
    #             component=None
    #         for attr in attrs:
    #             self.update_param(df_dict['name'][i], attr, df_dict[attr][i],
    #                               component=component)

    def update_parameters(self, params=None, component=None, **kwargs):
        if params:
            if isinstance(params, dict):
                self._update_parameters_from_dict(params, component=component)
            elif isinstance(params, np.ndarray):
                self._update_parameters_from_array(params)
            elif isinstance(params, pd.DataFrame):
                self._update_parameters_from_dataframe(params)
            else:
                self._update_parameters_from_keywords(component=component, **kwargs)

        density_grid = self._calculate_density_grid()
        self.set_physical_properties(
            density=density_grid,
            dusttogasratio=self.dusttogasratio,
            dust=self.dust_file,
            amax=self.amax,
            p=self.p,
            dust_abundances=self.dust_abundances,
            gases=self.gases,
            abundances=self.abundances,
            velocity=self.velocity,
            microturbulence=self.microturbulence,
        )

    # def __getitem__(self, key):
    #     """Return the quantity of the parameter identified by a `(name, kind)` tuple."""
    #     if isinstance(key, tuple) and len(key) == 2:
    #         name, kind = key
    #         return self.parameters[(name, kind)].quantity
    #     raise TypeError(f"key must be a (name, kind) tuple, not {type(key).__name__}")

    # def _param(self, name, kind):
    #     """Return the quantity of the parameter with the given `name` and `kind`."""
    #     return self.parameters[(name, kind)].quantity

    def _calculate_density_grid(self):
        """Build the coordinate meshgrid for `self.density_coordinates` and evaluate `self.density`."""
        mgrid = self._get_density_meshgrid(self.density_coordinates)
        return self.density(*mgrid)

    def _get_density_meshgrid(self, density_coords):
        """Build a coordinate meshgrid over the model's grid cell centers.

        Reads cell-wall coordinates off `self.grid.grid` (cartesian or spherical,
        per `self.grid.coordinate_system`), converts to cell-center coordinates,
        and returns them in the coordinate system requested by `density_coords`.

        Parameters
        ----------
        density_coords : {"cylindrical", "spherical"}
            Coordinate system to return the meshgrid in.

        Returns
        -------
        tuple of np.ndarray
            `(r, z)` if `density_coords == "cylindrical"`, or `(r, theta, phi)`
            if `density_coords == "spherical"`.
        """
        if self.grid.coordinate_system == "cartesian":
            x_edges = (self.grid.grid.w1.numpy() * self.grid_unit).cgs.value
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_edges = (self.grid.grid.w2.numpy() * self.grid_unit).cgs.value
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            z_edges = (self.grid.grid.w3.numpy() * self.grid.unit).cgs.value
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

            xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

            rr = np.sqrt(xx + yy + zz)
            tt = np.arccos(zz / rr)
            pp = np.atan2(yy, xx)

            rcyl = np.sqrt(xx**2 + yy**2)

        elif self.grid.coordinate_system == "spherical":
            r_edges = (self.grid.grid.w1.numpy() * self.grid_unit).cgs.value
            r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
            theta_edges = self.grid.grid.w2.numpy()
            theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
            phi_edges = self.grid.grid.w3.numpy()
            phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

            rr, tt, pp = np.meshgrid(
                r_centers, theta_centers, phi_centers, indexing="ij"
            )
            rcyl = rr * np.sin(tt)
            zz = rr * np.cos(tt)

        if density_coords == "cylindrical":
            return rcyl, zz
        elif density_coords == "spherical":
            return rr, tt, pp
