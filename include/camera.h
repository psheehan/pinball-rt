#ifndef CAMERA_H
#define CAMERA_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <Kokkos_Random.hpp>

#include "vector.h"
#include "photon.h"
#include "grid.h"
#include "misc.h"
#include "params.h"
#include "misc.h"

namespace py = pybind11;

struct Image {
    Kokkos::View<double*> x{"x", 0};
    Kokkos::View<double*> y{"y", 0};
    Kokkos::View<double*> lam{"lam", 0};
    Kokkos::View<double*> nu{"nu", 0};
    Kokkos::View<double*> intensity{"intensity", 0};

    double pixel_size;
    int nx;
    int ny;
    int nnu;

    Image(int nx, int ny, double pixel_size, py::array_t<double> lam);

    ~Image();
};

struct UnstructuredImage {
    Kokkos::View<double*> x{"x", 0};
    Kokkos::View<double*> y{"y", 0};
    Kokkos::View<double*> lam{"lam", 0};
    Kokkos::View<double*> nu{"lam", 0};
    Kokkos::View<double**> intensity{"intensity", 0, 0};

    double pixel_size;
    int nx;
    int ny;
    int nnu;

    UnstructuredImage(int nx, int ny, double pixel_size, 
            py::array_t<double> lam);
    UnstructuredImage(int nr, int nphi, double rmin, double rmax, 
            py::array_t<double> lam);
};

struct Spectrum {
    Kokkos::View<double*> lam{"lam", 0};
    Kokkos::View<double*> nu{"nu", 0};
    Kokkos::View<double*> intensity{"intensity", 0};
    double pixel_size;
    int nnu;

    Spectrum(py::array_t<double> lam);
};

struct Camera {
    Grid* G;
    Params *Q;

    Kokkos::Random_XorShift64_Pool<> *random_pool;

    double r;
    double incl;
    double pa;

    Vector<double, 3> i;
    Vector<double, 3> ex;
    Vector<double, 3> ey;
    Vector<double, 3> ez;

    Camera(Grid *_G, Params *_Q);

    Image* make_image(int nx, int ny, double pixel_size, 
            py::array_t<double> lam, double incl, double pa, double dpc, 
            int nthreads);

    UnstructuredImage* make_unstructured_image(int nx, int ny, 
            double pixel_size, py::array_t<double> lam, double incl, 
            double pa, double dpc, int nthreads);

    UnstructuredImage* make_circular_image(int nr, int nphi, 
            py::array_t<double> lam, double incl, double pa, double dpc, 
            int nthreads);

    Spectrum* make_spectrum(py::array_t<double> lam, double incl, 
            double pa, double dpc, int nthreads);

    void set_orientation(double incl, double pa, double dpc);

    Ray* emit_ray(double x, double y, double pixel_size, Kokkos::View<double*> nu, int nnu);
    Kokkos::View<double*> raytrace_pixel(double x, double y, double pixel_size, Kokkos::View<double*> nu, 
            int nnu, int count);
    void raytrace_pixel(UnstructuredImage *image, int ix, double pixel_size); 
    Kokkos::View<double*> raytrace(double x, double y, double pixel_size, Kokkos::View<double*> nu, 
            int nnu, bool unstructured, bool *pixel_too_large);

    void raytrace_sources(Image *I, int nthreads);
    void raytrace_sources(UnstructuredImage *I);
};

#endif
