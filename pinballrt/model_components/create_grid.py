from .Parameter import Parameter
from .ParameterList import ParameterList

uniform_cartesian_params = ParameterList([
   Parameter(name='ncells', value=9, kind='grid'),
   Parameter(name='dx', value=1.0, unit=u.au, kind='grid'),
   Parameter(name='device', value='cpu', kind='grid')])

uniform_spherical_params = ParameterList([
   Parameter(name='ncells', value=9, kind='grid'),
   Parameter(name='dr', value=1.0, unit=u.au, kind='grid'),
   Parameter(name='mirror', value=True, kind='grid'),
   Parameter(name='device', value='cpu', kind='grid')])

log_uniform_spherical_params = ParameterList([
   Parameter(name='ncells', value=9, kind='grid'),
   Parameter(name='rmin', value=0.1, unit=u.au, kind='grid'),
   Parameter(name='rmax', value=4.5, unit=u.au, kind='grid'),
   Parameter(name='mirror', value=True, kind='grid'),
   Parameter(name='device', value='cpu', kind='grid')])

def update_grid_params(params):
    grid_type = params.grid_type

    if grid_type == "uniform_cartesian":
        for param_name in uniform_cartesian_params.keys():
            if param_name not in param.keys():
                params.add(uniform_cartesian_params[param_name])
    if grid_type == "uniform_spherical":
        for param_name in uniform_spherical_params.keys():
            if param_name not in param.keys():
                params.add(uniform_spherical_params[param_name])
    if grid_type == "log_uniform_spherical":
        for param_name in log_uniform_spherical_params.keys():
            if param_name not in param.keys():
                params.add(log_uniform_spherical_params[param_name])

    return params
