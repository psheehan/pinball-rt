import pandas as pd

class ModelComponent:
    model_defaults = pd.DataFrame([])
    model_default_params = pd.DataFrame.from_dict({
            "disk_mass":  {"value": -3., "prior_min": -5., "prior_max": 0., 
            "Fixed": False, "unit": u.Msun, "log":True, "component": model_name},
            "disk_rin": {"value": -1., "prior_min": -2., "prior_max": 0., 
            "Fixed": False, "unit": u.au, "log":True, "component": model_name},
            "disk_rout": {"value": 2., "prior_min": 1., "prior_max": 3., 
            "Fixed": False, "unit": u.au, "log":True, "component": model_name},
            "gamma": {"value": 1., "prior_min": 0., "prior_max": 2., 
            "Fixed": False,  "component": model_name},
            "h0": {"value": 0.05, "prior_min": 0.01, "prior_max": 0.3, 
            "Fixed": False, "unit": u.au, "component": model_name},
            "beta": {"value": 1., "prior_min": 0.5, "prior_max": 1.5, 
            "Fixed": False,  "component": model_name}
        }, orient='index')

    def __init__(parameters=None):

        self.parameters = self.model_defaults.copy()
        self.update_parameters(

    def update_parameters(self, updates=None, **kwargs):
        if updates:
            if isinstance(updates, dict):
                new_df = pd.DataFrame(updates, orient='index')
                self.parameters.update(new_df)
            elif isinstance(updates, pd.DataFrame):
                self.parameters.update(new_df)
        else:
            for param_name, value in kwargs.items():
                self.parameters.loc[param_name, 'value'] = value


