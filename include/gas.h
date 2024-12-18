#ifndef GAS_H
#define GAS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stdlib.h>
#include <cmath>
#include "misc.h"

namespace py = pybind11;

struct Gas {
    double mu;

    int nlevels;
    Kokkos::View<int*> levels{"levels", 0};
    Kokkos::View<double*> energies{"energies", 0};
    Kokkos::View<double*> weights{"weights", 0};
    Kokkos::View<int*> J{"J", 0};

    int ntransitions;
    Kokkos::View<int*> transitions{"transitions", 0};
    Kokkos::View<int*> up{"up", 0};
    Kokkos::View<int*> low{"low", 0};
    Kokkos::View<double*> A{"A", 0};
    Kokkos::View<double*> nu{"nu", 0};
    Kokkos::View<double*> Eu{"Eu", 0};

    int ntemp;
    Kokkos::View<double*> temp{"temp", 0};
    Kokkos::View<double*> Z{"Z", 0};
    Kokkos::View<double*> dZdT{"dZdT", 0};

    Gas();
    Gas(double _mu, py::array_t<int> __levels, py::array_t<double> __energies, 
            py::array_t<double> __weights, py::array_t<int> __J, 
            py::array_t<int> __transitions, py::array_t<int> __up, 
            py::array_t<int> __low, py::array_t<double> __A, 
            py::array_t<double> __nu, py::array_t<double> __Eu);

    void set_properties(double _mu, py::array_t<int> __levels, py::array_t<double> __energies, 
            py::array_t<double> __weights, py::array_t<int> __J, 
            py::array_t<int> __transitions, py::array_t<int> __up, 
            py::array_t<int> __low, py::array_t<double> __A, 
            py::array_t<double> __nu, py::array_t<double> __Eu);

    void set_properties(double _mu, Kokkos::View<int*> __levels, Kokkos::View<double*> __energies, 
            Kokkos::View<double*> __weights, Kokkos::View<int*> __J, 
            Kokkos::View<int*> __transitions, Kokkos::View<int*> __up, 
            Kokkos::View<int*> __low, Kokkos::View<double*> __A, 
            Kokkos::View<double*> __nu, Kokkos::View<double*> __Eu);

    void copy(Gas *G);

    ~Gas();

    double partition_function(double T);
};

#endif
