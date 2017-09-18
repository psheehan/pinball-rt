import cython

"""
from libcpp cimport bool
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np
"""

from mcrt3d cimport *

"""
cdef extern from "../src/params.h":
    cppclass Params:
        int nphot
        bool bw
        bool scattering
        double scattering_nu
        bool verbose
        bool use_mrw
        double mrw_gamma

        void set_nphot(int _nphot)
        void set_bw(bool _bw)
        void set_scattering(bool _scattering)
        void set_verbose(bool _verbose)
        void set_mrw(bool _use_mrw)
        void set_mrw_gamma(double _mrw_gamma)

cdef extern from "../src/dust.h":
    cppclass Dust:
        int nlam
        double *nu
        double *lam
        double *kabs
        double *ksca
        double *kext
        double *albedo
        double *dkextdnu
        double *dalbedodnu

        int ntemp
        double *temp
        double *planck_opacity
        double *dplanck_opacity_dT
        double *rosseland_extinction
        double *drosseland_extinction_dT
        double **random_nu_CPD
        double **random_nu_CPD_bw
        double **drandom_nu_CPD_dT
        double **drandom_nu_CPD_bw_dT

        Dust(int _nlam, double *_nu, double *_lam, double *_kabs, 
                double *_ksca, double *_kext, double *_albedo)
            
        void set_lookup_tables(int _ntemp, double *_temp, 
                double *_planck_opacity, double *_rosseland_extinction, 
                double *_dplanck_opacity_dT, double *_drosseland_extinction_dT,
                double *_dkextdnu, double *dalbedodnu, double *_random_nu_CPD, 
                double *_random_nu_CPD_bw, double *_drandom_nu_CPD_dT, 
                double *_drandom_nu_CPD_bw_dT)

cdef extern from "../src/isotropic_dust.h":
    cppclass IsotropicDust(Dust)

cdef extern from "../src/source.h":
    cppclass Source:
        double *nu
        double *Bnu
        int nnu

cdef extern from "../src/grid.h":
    cppclass Grid:
        int n1
        int n2
        int n3
        int nw1
        int nw2
        int nw3
        double *w1
        double *w2
        double *w3

        #std::vector<double***> dens
        #std::vector<double***> energy
        #std::vector<double***> temp
        #std::vector<double***> mass
        #std::vector<double****> scatt
        vector[double***] dens
        vector[double***] energy
        vector[double***] temp
        vector[double***] mass
        vector[double****] scatt
        double ***volume

        int nspecies
        #std::vector<Dust*> dust
        vector[Dust*] dust

        int nsources
        #std::vector<Source*> sources
        vector[Source*] sources
        double total_lum

        Params *Q

        int ny
        double *y
        double *f
        double *dydf

        Grid(int _n1, int _n2, int _n3, int _nw1, int _nw2, int _nw3, 
                double *_w1, double *_w2, double *_w3, double *_volume)

        void add_density(double *_dens, double *_temp, double *_mass, 
                Dust *D)
        void add_source(Source *S)
        void set_mrw_tables(double *y, double *f, double *dydf, int ny)

cdef extern from "../src/cartesian_grid.h":
    cppclass CartesianGrid(Grid)

cdef extern from "../src/cylindrical_grid.h":
    cppclass CylindricalGrid(Grid)

cdef extern from "../src/spherical_grid.h":
    cppclass SphericalGrid(Grid)

cdef extern from "../src/mcrt3d.h":
    cppclass MCRT:
        Grid *G
        Params *Q

        MCRT(Grid *_G, Params *_Q)

        void thermal_mc()
        void scattering_mc()
        void mc_iteration()
"""

# Define the Params and MCRT classes here because there isn't yet a better 
# place.

cdef class ParamsObj:
    cdef Params *obj

    def __init__(self):
        self.obj = new Params()
        if self.obj == NULL:
            raise MemoryError("Not enough memory.")

    def __del__(self):
        del self.obj

    property nphot:
        def __get__(self):
            return self.obj.nphot
        def __set__(self, int var):
            self.obj.set_nphot(var)

    property bw:
        def __get__(self):
            return self.obj.bw
        def __set__(self, bool var):
            self.obj.set_bw(var)

    property scattering:
        def __get__(self):
            return self.obj.scattering
        def __set__(self, bool var):
            self.obj.set_scattering(var)

    property verbose:
        def __get__(self):
            return self.obj.verbose
        def __set__(self, bool var):
            self.obj.set_verbose(var)

    property use_mrw:
        def __get__(self):
            return self.obj.use_mrw
        def __set__(self, bool var):
            self.obj.set_mrw(var)

    property mrw_gamma:
        def __get__(self):
            return self.obj.mrw_gamma
        def __set__(self, double var):
            self.obj.set_mrw_gamma(var)

"""
cdef class MCRTObj:
    cdef MCRT *obj

    def __init__(self, Grid *_G, Params *_Q):
        self.obj = new MCRT(_G, _Q)

    def run_thermal_mc(self):
        self.obj.thermal_mc()

    def run_scattering_mc(self):
        self.obj.scattering_mc()

    def run_mc_iteration(self):
        self.obj.mc_iteration()
"""