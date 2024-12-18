#ifndef GRID_H
#define GRID_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <signal.h>

#include <cmath>
#include <vector>
#include <algorithm>
#include "vector.h"
#include "dust.h"
#include "isotropic_dust.h"
#include "source.h"
#include "star.h"
#include "photon.h"
#include "misc.h"
#include "params.h"
#include "gas.h"

namespace py = pybind11;

struct Grid {
    int n1;
    int n2;
    int n3;
    int nw1;
    int nw2;
    int nw3;
    Kokkos::View<double*> w1{"w1", 0};
    Kokkos::View<double*> w2{"w2", 0};
    Kokkos::View<double*> w3{"w3", 0};

    Kokkos::Random_XorShift64_Pool<> *random_pool;

    bool mirror_symmetry;

    py::list _scatt;

    Kokkos::View<double****> dens{"density", 0, 0, 0, 0};
    Kokkos::View<double****> energy{"energy", 0, 0, 0, 0};
    Kokkos::View<double****> energy_mrw{"energy", 0, 0, 0, 0};
    Kokkos::View<double****> temp{"temperature", 0, 0, 0, 0};
    Kokkos::View<double****> mass{"mass", 0, 0, 0, 0};
    Kokkos::View<double****> rosseland_mean_extinction{"rosseland_mean_extinction", 0, 0, 0, 0};
    Kokkos::View<double****> planck_mean_opacity{"planck_mean_opacity", 0, 0, 0, 0};
    Kokkos::View<double****> luminosity{"luminosity", 0, 0, 0, 0};
    Kokkos::View<double*****> scatt{"scattering_phase_function", 0, 0, 0, 0, 0};
    Kokkos::View<double***> volume{"volume", 0, 0, 0};
    Kokkos::View<double***> uses_mrw{"uses_mrw", 0, 0, 0};

    Kokkos::View<double****> number_dens{"number_density", 0, 0, 0, 0};
    Kokkos::View<double****> gas_temp{"gas_temperature", 0, 0, 0, 0};
    Kokkos::View<double****[3]> velocity{"velocity", 0, 0, 0, 0};
    Kokkos::View<double****> microturbulence{"microturbulence", 0, 0, 0, 0};

    int nspecies;
    //Kokkos::View<Dust*> dust;
    Dust* dust;

    int ngases;
    //Kokkos::View<Gas*> gas;
    Gas* gas;

    Kokkos::View<int*> include_lines{"include_lines", 0};
    Kokkos::View<double****> level_populations{"level_populations", 0, 0, 0, 0};
    Kokkos::View<double****> alpha_line{"alpha_line", 0, 0, 0, 0};
    Kokkos::View<double****> inv_gamma_thermal{"inv_gamma_thermal", 0, 0, 0, 0};


    int nsources;
    //Kokkos::View<Star*> sources;
    Star* sources;
    double total_lum;

    Params *Q;

    Grid(py::array_t<double> w1, py::array_t<double> w2, 
            py::array_t<double> w3);

    /*Grid(int _n1, int _n2, int _n3, int _nw1, int _nw2, int _nw3, 
            double *_w1, double *_w2, double *_w3, double *_volume,
            bool _mirror_symmetry);*/

    ~Grid();

    //void add_density(double *_dens, double *_temp, double *_mass, 
    //        Dust *D);
    void add_density(py::array_t<double>, Dust *d);
    void add_number_density(py::array_t<double>, py::array_t<double>, 
            py::array_t<double>, Gas *g);

    //void add_source(Source *S);
    void add_star(double x, double y, double z, double _mass, double _radius, 
            double _temperature);

    //void add_scattering_array(py::array_t<double> _scatt, int nthreads);
    void initialize_scattering_array(int nthreads);
    void deallocate_scattering_array(int start);
    //void collapse_scattering_array();

    //void add_energy_arrays(int nthreads);
    //void deallocate_energy_arrays();

    void initialize_luminosity_array();
    void initialize_luminosity_array(double nu);
    void deallocate_luminosity_array();

    Photon *emit(int iphot);
    Photon *emit(double _nu, double _dnu, int photons_per_source);

    virtual Vector<double, 3> random_location_in_cell(int ix, int iy, int iz);

    virtual double next_wall_distance(Photon *P);
    virtual double outer_wall_distance(Photon *P);
    virtual double minimum_wall_distance(Photon *P);
    virtual double smallest_wall_size(Photon *P);
    virtual double smallest_wall_size(Ray *R);
    virtual double grid_size();

    void propagate_photon_full(Photon *P);
    void propagate_photon(Photon *P, double tau, bool absorb);
    void propagate_photon_scattering(Photon *P);
    void propagate_photon_mrw(Photon *P);
    void propagate_ray(Ray *R);
    void propagate_ray_from_source(Ray *R);

    void absorb(Photon *P, int idust);
    void absorb_mrw(Photon *P, int idust);
    void scatter(Photon *P, int idust);

    void random_dir_mrw(Photon *P);

    virtual Vector<int, 3> photon_loc(Photon *P);
    virtual void photon_loc_mrw(Photon *P);
    virtual bool in_grid(Photon *P);
    virtual bool on_and_parallel_to_wall(Photon *P);

    void update_grid(Vector<int, 3> l, int cell_index);
    void update_grid();

    double cell_lum(int idust, int i, int j, int k);
    double cell_lum(int idust, int i, int j, int k, double nu);

    Vector<double, 3> vector_velocity(int igas, Photon *P);
    double maximum_velocity(int igas);
    double maximum_gas_temperature(int igas);
    double maximum_microturbulence(int igas);
    double line_profile(int igas, int iline, int itrans, int i, int j, int k, 
            double nu);
    void calculate_level_populations(int igas, int iline);
    void set_tgas_eq_tdust();
    void select_lines(py::array_t<double> lam);
    void deselect_lines();
};

#endif
