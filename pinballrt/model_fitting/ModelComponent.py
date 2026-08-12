import numpy as np

from .Parameters import Parameter, ParameterSet


class ModelComponent:
    default_parameters = ParameterSet()
    name = "model_component"

    def __init__(self, parameters=None):

        self.parameters = self.default_parameters.copy()
        if parameters:
            self.update_parameters(parameters)

    def __getattr__(self, param_name):
        return self.parameters[param_name].quantity

    def update_parameters(self, updates=None, **kwargs):
        valid_params = [p.name for p in self.default_parameters]
        if isinstance(updates, Parameter):
            if updates.name not in valid_params:
                raise KeyError("Invalid parameter")
            else:
                self.parameters.add(parameter=updates, overlap_method="replace")
        elif isinstance(updates, ParameterSet):
            new_params = [p.name for p in updates]
            invalid = set(new_params) - set(valid_params)
            if invalid:
                raise KeyError("The following parameters are not valid: ", invalid)
            else:
                self.parameters = ParameterSet.merge(updates, overlap_method="replace")
        elif isinstance(updates, (np.ndarray, list)):
            if len(updates) != len(self.vector):
                raise KeyError(
                    "Number of updates does not match number of free parameters"
                )
            else:
                free_param_names = [p.name for p in self.parameters.free]
                free_param_indicies = [
                    self.parameters.index(name) for name in free_param_names
                ]
                for i, val in enumerate(updates):
                    self.parameters[free_param_indicies[i]].value = val
        else:
            for param_name, val in kwargs.items():
                ind = self.parameters.index(param_name)
                self.parameters[ind].value = val
