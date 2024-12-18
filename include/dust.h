#ifndef DUST_H
#define DUST_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <Kokkos_Random.hpp>

#include <stdlib.h>
#include <cmath>
#include "misc.h"
#include "photon.h"

namespace py = pybind11;

struct Dust {
    int nlam;
    Kokkos::View<double*> nu{"nu", 0};
    Kokkos::View<double*> lam{"lam", 0};
    Kokkos::View<double*> kabs{"kabs", 0};
    Kokkos::View<double*> ksca{"ksca", 0};
    Kokkos::View<double*> kext{"kext", 0};
    Kokkos::View<double*> albedo{"albedo", 0};
    Kokkos::View<double*> dkextdnu{"dkextdnu", 0};
    Kokkos::View<double*> dalbedodnu{"dalbedodnu", 0};

    Kokkos::Random_XorShift64_Pool<> *random_pool;

    int ntemp;
    Kokkos::View<double*> temp{"temp", 0};
    Kokkos::View<double*> planck_opacity{"planck_opacity", 0};
    Kokkos::View<double*> dplanck_opacity_dT{"dplanck_opacity_dT", 0};
    Kokkos::View<double*> rosseland_extinction{"rosseland_extinction", 0};
    Kokkos::View<double*> drosseland_extinction_dT{"drosseland_extinction_dT", 0};
    Kokkos::View<double**> random_nu_CPD{"random_nu_CPD", 0, 0};
    Kokkos::View<double**> random_nu_CPD_bw{"random_nu_CPD_bw", 0, 0};
    Kokkos::View<double**> drandom_nu_CPD_dT{"drandom_nu_CPD_dT", 0, 0};
    Kokkos::View<double**> drandom_nu_CPD_bw_dT{"drandom_nu_CPD_bw_dT", 0, 0};

    Dust();
    Dust(py::array_t<double> lam, py::array_t<double> kabs, py::array_t<double> ksca);

    void copy(Dust *D);

    void set_properties(py::array_t<double> lam, py::array_t<double> kabs, 
            py::array_t<double> ksca);
    void set_properties(Kokkos::View<double*> h_lam, Kokkos::View<double*> h_kabs,
        Kokkos::View<double*> h_ksca);

    /*Dust(int _nlam, double *_nu, double *_lam, double *_kabs, double *_ksca, 
            double *_kext, double *_albedo);*/

    ~Dust();
        
    void set_lookup_tables();

    virtual void scatter(Photon *P);
    void absorb(Photon *P, double T, bool bw);
    void absorb_mrw(Photon *P, double T, bool bw);

    double random_nu(double T, bool bw);
    double opacity(double freq);
    double albdo(double freq);
    double planck_mean_opacity(double T);
    double rosseland_mean_extinction(double T);
};

#endif
