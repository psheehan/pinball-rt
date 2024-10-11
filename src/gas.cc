#include "gas.h"

/* Functions to set up the dust. */

Gas::Gas(double _mu, py::array_t<int> __levels, py::array_t<double> __energies, 
        py::array_t<double> __weights, py::array_t<int> __J, 
        py::array_t<int> __transitions, py::array_t<int> __up, 
        py::array_t<int> __low, py::array_t<double> __A, 
        py::array_t<double> __nu, py::array_t<double> __Eu) {

    mu = _mu;

    /*_levels = __levels; _energies = __energies; _weights = __weights; _J = __J;
    _transitions = __transitions; _up = __up; _low = __low; _A = __A;
    _nu = __nu; _Eu = __Eu;

    // Load the array buffers to get the proper setup info.*/

    auto _levels_buf = __levels.request(); 
    /*auto _energies_buf = __energies.request();
    auto _weights_buf = __weights.request();
    auto _J_buf = __J.request();*/

    auto _transitions_buf = __transitions.request(); 
    /*auto _up_buf = __up.request();
    auto _low_buf = __low.request();
    auto _A_buf = __A.request();
    auto _nu_buf = __nu.request();
    auto _Eu_buf = __Eu.request();

    if (_levels_buf.ndim != 1 || _energies_buf.ndim != 1 || 
            _weights_buf.ndim != 1 || _J_buf.ndim != 1 || 
            _transitions_buf.ndim != 1 || _up_buf.ndim != 1 || 
            _low_buf.ndim != 1 || _A_buf.ndim != 1 || _nu_buf.ndim != 1 
            || _Eu_buf.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    // Now get the correct format.*/

    nlevels = _levels_buf.shape[0];
    /*levels = (int *) _levels_buf.ptr;
    energies = (double *) _energies_buf.ptr;
    weights = (double *) _weights_buf.ptr;
    J = (int *) _J_buf.ptr;*/

    ntransitions = _transitions_buf.shape[0];
    /*transitions = (int *) _transitions_buf.ptr;
    up = (int *) _up_buf.ptr;
    low = (int *) _low_buf.ptr;
    A = (double *) _A_buf.ptr;
    nu = (double *) _nu_buf.ptr;
    Eu = (double *) _Eu_buf.ptr;*/

    Kokkos::resize(levels, nlevels);
    Kokkos::deep_copy(levels, view_from_array(__levels));
    Kokkos::resize(energies, nlevels);
    Kokkos::deep_copy(energies, view_from_array(__energies));
    Kokkos::resize(weights, nlevels);
    Kokkos::deep_copy(weights, view_from_array(__weights));
    Kokkos::resize(J, nlevels);
    Kokkos::deep_copy(J, view_from_array(__J));

    Kokkos::resize(transitions, ntransitions);
    Kokkos::deep_copy(transitions, view_from_array(__transitions));
    Kokkos::resize(up, ntransitions);
    Kokkos::deep_copy(up, view_from_array(__up));
    Kokkos::resize(low, ntransitions);
    Kokkos::deep_copy(low, view_from_array(__low));
    Kokkos::resize(A, ntransitions);
    Kokkos::deep_copy(A, view_from_array(__A));
    Kokkos::resize(nu, ntransitions);
    Kokkos::deep_copy(nu, view_from_array(__nu));
    Kokkos::resize(Eu, ntransitions);
    Kokkos::deep_copy(Eu, view_from_array(__Eu));

    // Calculate the partition function.

    ntemp = 1000;
    Kokkos::resize(temp, ntemp);
    Kokkos::resize(Z, ntemp);

    Kokkos::View<double*>::HostMirror h_temp = Kokkos::create_mirror_view(temp);
    Kokkos::View<double*>::HostMirror h_Z = Kokkos::create_mirror_view(Z);
    Kokkos::View<double*>::HostMirror h_weights = Kokkos::create_mirror_view(weights);
    Kokkos::View<double*>::HostMirror h_energies = Kokkos::create_mirror_view(energies);

    Kokkos::deep_copy(h_temp, temp);
    Kokkos::deep_copy(h_Z, Z);
    Kokkos::deep_copy(h_weights, weights);
    Kokkos::deep_copy(h_energies, energies);

    for (size_t i = 0; i < ntemp; i++) {
        h_temp(i) = pow(10.,-1.+i*6./(ntemp-1));
        h_Z(i) = 0;

        for (int j = 0; j < nlevels; j++)
            h_Z(i) += h_weights(j)*exp(-h_p*c_l*h_energies(j) / (k_B * h_temp(i)));
    }

    Kokkos::deep_copy(temp, h_temp);
    Kokkos::deep_copy(Z, h_Z);
    Kokkos::deep_copy(weights, h_weights);
    Kokkos::deep_copy(energies, h_energies);

    Kokkos::resize(dZdT, ntemp-1);
    Kokkos::deep_copy(dZdT, derivative(Z, temp, ntemp));
}

Gas::~Gas() {
}

double Gas::partition_function(double T) {
    int n = find_in_arr(T,temp,ntemp);

    double partition_function = dZdT(n)*(T-temp(n))+Z(n);

    return partition_function;
}
