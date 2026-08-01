from .Fittable_Model import Fittable_Model
from .Parameter import Parameter
from .ParameterList import ParameterList
import numpy as np
import astropy.units as u


class Envelope(Fittable_Model):

    model_name = "envelope"
    default_params = ParameterList([
        Parameter("rho0",      2e-12, model_name, None, False, u.g/u.cm**3),
        Parameter("rin",       0.1,   model_name, None, False, u.au),
        Parameter("rout",      1000., model_name, None, False, u.au),
        Parameter("pl",        1.5,   model_name, None, False, None),
        Parameter("cavpl",     1.0,   model_name, None, False, None),
        Parameter("cavrrfact", 0.2,   model_name, None, False, None),
        Parameter("dusttogasratio", 0.01, "dust"),
        Parameter("amax", 1.0, "dust", unit=u.cm),
        Parameter("p", 3.0, "dust"),
        Parameter("dust_file", "diana_wice.dst", "dust"),
        Parameter("dust_abundances", (), "dust"),
        Parameter("gases", None, "gas"),
        Parameter("abundances", None, "gas"),
        Parameter("velocity", None, "gas"),
        Parameter("microturbulence", None, "gas")
    ])

    density_coordinates = "spherical"

    def density(self, r, theta, phi):
        rho0 = self["rho0", Envelope.model_name]
        rin  = self["rin",  Envelope.model_name]
        pl   = self["pl",   Envelope.model_name]

        rho = rho0 * (r / rin)**-pl
        return rho.to(u.g / u.cm**3)
