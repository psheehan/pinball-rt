import copy
from collections.abc import Sequence

import numpy as np


class Parameter:
    """Class representing a single parameter"""

    def __init__(
        self, name, value, component, prior_range=None, fixed=True, unit=None, log=False
    ):
        """Initialize the parameter

        Args:
            name (str): name of the parameter
            value (str, bool, or float): value of the parameter
            component (str): name of the component the parameter belongs to
            prior_range (tuple or list, optional): lower and upper bounds of the parameter. Defaults to None.
            fixed (bool, optional): Determines if the parameter is fixed (True) or variable (False). Defaults to True.
            unit (astropy unit, optional): Astropy unit of the parameter if applicable. Defaults to None.
            log (bool, optional): Indicates if the parameter is in log space or not. Defaults to False.
        """
        self.name = name
        self.value = value
        self.prior_range = prior_range
        self.fixed = fixed
        self.unit = unit
        self.log = log
        self.component = component

    @property
    def quantity(self):
        """Extract the physical value of the parameter

        Returns:
            float or quantity: physical value of parameter, with units and un-logged when applicable
        """
        if isinstance(self.value, (str, bool)):
            return self.value
        else:
            physical = 10**self.value if self.log else self.value
            return physical * self.unit if self.unit is not None else physical


class ParameterSet(Sequence):
    """Class representing a set of parameters"""

    def __init__(self, parameters=None):
        """Iniialize the parameter set

        Args:
            parameter_list (iterable[Parameter], optional): Iterable of parameter objects. Defaults to None.
        """
        if parameters:
            self.parameters = list(parameters)
        else:
            self.parameters = []

    def __getitem__(self, key):
        """Allows the object to return the parameter at a given index

        Args:
            key (int or str): index or name of desired parameter

        Returns:
            Parameter: Paramter at the appropiate index
        """
        if isinstance(key, int):
            return self.parameters[key]
        elif isinstance(key, str):
            ind = self.index(key)
            return self.parameters[ind]

    def index(self, name):
        ind = self.names.index(name)
        return ind

    def __len__(self):
        return len(self.parameters)

    def __getattr__(self, param_name):
        return self[param_name].quantity

    @property
    def names(self):
        """Get the names of the paramters

        Returns:
            list: list of parameter names
        """
        return [p.name for p in self.parameters]

    @property
    def free(self):
        """Get a paramter set of just the free parameters

        Returns:
            ParameterSet: subset of free parameters
        """
        free_params = [p for p in self.parameters if p.fixed == False]
        return ParameterSet(free_params)

    def get_component(self, component):
        """Get a parameter set of parameters of one component

        Args:
            component (string): Subset of parameters of the given compoent

        Returns:
            _type_: _description_
        """
        params = [p for p in self.parameters if p.component == component]
        return ParameterSet(params)

    @property
    def vector(self):
        """Get the vector of free parameters, useful for fitting

        Returns:
            np.ndarray: array corresponding to values of the free parameters
        """
        free_params = self.free
        return np.array([p.value for p in free_params])

    def add(self, parameter, overlap_method="warn"):
        """Add a new parameter to the parameter set

        Args:
            parameter (Parameter): new parameter to add
            overlap_method (str, optional):
                If "replace", new parameter with replace one with the same name.
                If "keep", will keep original parameter with the same name.
                If "warn", will raise an error if the parameter already exists.
                Defaults to "warn".

        Raises:
            KeyError: If overlap_method is "warn", will raise if paramter with the same name
                is already in the parameter set
        """
        if parameter.name in self.names:
            if overlap_method == "replace":
                index = self.names.index(parameter.name)
                self.parameters[index] = parameter
            elif overlap_method == "keep":
                pass
            elif overlap_method == "warn":
                raise KeyError(f"{parameter.name} already in ParameterSet")
        else:
            self.parameters.append(parameter)

    def copy(self):
        """Make a copy of the parameter set

        Returns:
            ParameterSet: Copy of parameter set
        """
        result = ParameterSet()
        for p in self.parameters:
            result.add(copy.deepcopy(p))
        return result

    @classmethod
    def merge(cls, parameter_sets, overlap_method="warn"):
        new_set = ParameterSet()
        if isinstance(parameter_sets, ParameterSet):
            for p in parameter_sets:
                new_set.add(parameter=p, overlap_method=overlap_method)
        else:
            for parameter_set in parameter_sets:
                for p in parameter_set:
                    new_set.add(parameter=p, overlap_method=overlap_method)

        return new_set
