#ifndef SOURCE_H
#define SOURCE_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stdlib.h>
#include "vector.h"
#include "dust.h"
#include "photon.h"
#include "misc.h"
#include "params.h"

namespace py = pybind11;

struct Source {
    Vector<double, 3> r;
    Kokkos::View<double*> nu{"nu", 0};
    Kokkos::View<double*> lam{"lam", 0};
    Kokkos::View<double*> Bnu{"Bnu", 0};
    int nnu;

    bool spectrum_set;

    virtual ~Source();

    virtual Photon *emit(int nphot);
    virtual Photon *emit(double _nu, double _dnu, int nphot);
    virtual Ray *emit_ray(Kokkos::View<double*> _nu, int _nnu, double _pixelsize, \
            Vector<double, 3> _n, int nphot);
    virtual Ray *emit_ray(Kokkos::View<double*> _nu, int _nnu, Vector<double, 3> _n, 
            int nphot);
    virtual double intercept_distance(Photon *P);
    virtual double random_nu();
    virtual double flux(double freq);

    virtual void reemit(Photon *P);
};

#endif
