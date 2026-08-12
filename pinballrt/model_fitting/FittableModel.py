import astropy.units as u
import numpy as np

from ..model import Model
from .ModelComponent import ModelComponent
from .Parameters import Parameter, ParameterSet


class FittableModel(Model):
    hyper_defaults = ParameterSet(
        [
            Parameter(name="mpi", value=True, component="hyper"),
            Parameter(name="ncores", value=1, component="hyper"),
        ]
    )
    dust_defaults = ParameterSet(
        [
            Parameter(name="dusttogasratio", value=0.01, component="dust"),
            Parameter(name="dust_file", value="diana_wice.dst", component="dust"),
            Parameter(name="amax", value=1.0, unit=u.cm, component="dust"),
            Parameter(name="p", value=3.0, component="dust"),
            Parameter(name="dust_abundances", value=[], component="dust"),
        ]
    )
    gas_defaults = ParameterSet(
        [
            Parameter(name="gases", value=None, component="gas"),
            Parameter(name="abundances", value=None, component="gas"),
            Parameter(name="velocity", value=None, component="gas"),
            Parameter(name="microturbulence", value=None, component="gas"),
        ]
    )

    def __init__(self, grid, mpi=None, ncores=None, components=None, parameters=None):
        if components:
            self.components = components
            component_defaults = [
                component.default_parameters for component in components
            ]
        else:
            self.components = []
            component_defaults = []
        all_default_parameters = [
            self.hyper_defaults,
            self.dust_defaults,
            self.gas_defaults,
        ] + component_defaults
        self.component_names = [component.name for component in self.components]
        self.default_parameters = ParameterSet.merge(all_default_parameters)
        self.parameters = self.default_parameters.copy()
        if mpi:
            self.parameters["mpi"].value = mpi
        if ncores:
            self.parameters["ncores"].value = ncores

        super().__init__(grid, ncores=self.parameters.ncores, mpi=self.parameters.mpi)

        self.update_parameters(parameters)

    def __getattr__(self, component_name):
        ind = self.component_names.index(component_name)
        return self.components[ind]

    def update_parameters(self, updates, **kwargs):
        ModelComponent.update_parameters(self, updates=updates, **kwargs)
        density_grids = []
        for component in self.components:
            new_component_params = self.parameters.get_component(component.name)
            component.update_parameters(new_component_params)
            mgrid = self._get_density_meshgrid(component.density_coordinates)
            density_grids.append(component.density(*mgrid))

        density_grid = np.mean(np.array(density_grids), axis=0) * u.g / u.cm**3
        print(np.shape(density_grid), density_grid.unit)
        self.set_physical_properties(
            density=density_grid,
            dusttogasratio=self.parameters.dusttogasratio,
            dust=self.parameters.dust_file,
            amax=self.parameters.amax,
            p=self.parameters.p,
            dust_abundances=self.parameters.dust_abundances,
            gases=self.parameters.gases,
            abundances=self.parameters.abundances,
            velocity=self.parameters.velocity,
            microturbulence=self.parameters.microturbulence,
        )

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
            x_edges = (self.grid.grid.w1.numpy() * self.grid.distance_unit).cgs.value
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_edges = (self.grid.grid.w2.numpy() * self.grid.distance_unit).cgs.value
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            z_edges = (self.grid.grid.w3.numpy() * self.grid.unit).cgs.value
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

            xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

            rr = np.sqrt(xx + yy + zz)
            tt = np.arccos(zz / rr)
            pp = np.atan2(yy, xx)

            rcyl = np.sqrt(xx**2 + yy**2)

        elif self.grid.coordinate_system == "spherical":
            r_edges = (self.grid.grid.w1.numpy() * self.grid.distance_unit).cgs.value
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
