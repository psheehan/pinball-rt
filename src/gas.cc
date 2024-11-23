#include "gas.h"

/* Functions to set up the dust. */

Gas::Gas() {}

Gas::Gas(double _mu, py::array_t<int> __levels, py::array_t<double> __energies, 
        py::array_t<double> __weights, py::array_t<int> __J, 
        py::array_t<int> __transitions, py::array_t<int> __up, 
        py::array_t<int> __low, py::array_t<double> __A, 
        py::array_t<double> __nu, py::array_t<double> __Eu) {

    set_properties(_mu, __levels, __energies, __weights, __J, __transitions, __up, 
            __low, __A, __nu, __Eu);
}

void Gas::copy(Gas *G) {
    set_properties(G->mu, G->levels, G->energies, G->weights, G->J, G->transitions, G->up, 
            G->low, G->A, G->nu, G->Eu);
}

void Gas::set_properties(double _mu, py::array_t<int> __levels, py::array_t<double> __energies, 
        py::array_t<double> __weights, py::array_t<int> __J, 
        py::array_t<int> __transitions, py::array_t<int> __up, 
        py::array_t<int> __low, py::array_t<double> __A, 
        py::array_t<double> __nu, py::array_t<double> __Eu) {

    auto h_levels = view_from_array(__levels);
    auto h_energies = view_from_array(__energies);
    auto h_weights = view_from_array(__weights);
    auto h_J = view_from_array(__J);

    auto h_transitions = view_from_array(__transitions);
    auto h_up = view_from_array(__up);
    auto h_low = view_from_array(__low);
    auto h_A = view_from_array(__A);
    auto h_nu = view_from_array(__nu);
    auto h_Eu = view_from_array(__Eu);

    set_properties(_mu, h_levels, h_energies, h_weights, h_J, h_transitions, h_up, h_low, 
            h_A, h_nu, h_Eu);
}

void Gas::set_properties(double _mu, Kokkos::View<int*> h_levels, Kokkos::View<double*> h__energies, 
        Kokkos::View<double*> h__weights, Kokkos::View<int*> h_J, 
        Kokkos::View<int*> h_transitions, Kokkos::View<int*> h_up, 
        Kokkos::View<int*> h_low, Kokkos::View<double*> h_A, 
        Kokkos::View<double*> h_nu, Kokkos::View<double*> h_Eu) {

    mu = _mu;

    /*_levels = __levels; _energies = __energies; _weights = __weights; _J = __J;
    _transitions = __transitions; _up = __up; _low = __low; _A = __A;
    _nu = __nu; _Eu = __Eu;

    // Load the array buffers to get the proper setup info.*/

    /*auto _energies_buf = __energies.request();
    auto _weights_buf = __weights.request();
    auto _J_buf = __J.request();*/

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

    nlevels = (int) h_levels.extent(0);
    /*levels = (int *) _levels_buf.ptr;
    energies = (double *) _energies_buf.ptr;
    weights = (double *) _weights_buf.ptr;
    J = (int *) _J_buf.ptr;*/

    ntransitions = (int) h_transitions.extent(0);
    /*transitions = (int *) _transitions_buf.ptr;
    up = (int *) _up_buf.ptr;
    low = (int *) _low_buf.ptr;
    A = (double *) _A_buf.ptr;
    nu = (double *) _nu_buf.ptr;
    Eu = (double *) _Eu_buf.ptr;*/

    Kokkos::resize(levels, nlevels);
    Kokkos::deep_copy(levels, h_levels);
    Kokkos::resize(energies, nlevels);
    Kokkos::deep_copy(energies, h__energies);
    Kokkos::resize(weights, nlevels);
    Kokkos::deep_copy(weights, h__weights);
    Kokkos::resize(J, nlevels);
    Kokkos::deep_copy(J, h_J);

    Kokkos::resize(transitions, ntransitions);
    Kokkos::deep_copy(transitions, h_transitions);
    Kokkos::resize(up, ntransitions);
    Kokkos::deep_copy(up, h_up);
    Kokkos::resize(low, ntransitions);
    Kokkos::deep_copy(low, h_low);
    Kokkos::resize(A, ntransitions);
    Kokkos::deep_copy(A, h_A);
    Kokkos::resize(nu, ntransitions);
    Kokkos::deep_copy(nu, h_nu);
    Kokkos::resize(Eu, ntransitions);
    Kokkos::deep_copy(Eu, h_Eu);

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
