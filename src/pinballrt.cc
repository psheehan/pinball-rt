#include "pinballrt.h"

#include "params.cc"
#include "dust.cc"
#include "isotropic_dust.cc"
#include "grid.cc"
#include "cartesian_grid.cc"
#include "cylindrical_grid.cc"
#include "spherical_grid.cc"
#include "source.cc"
#include "star.cc"
#include "camera.cc"
#include "misc.cc"
#include "photon.cc"
#include "gas.cc"

#include "timer.c"

Model::Model() {
    settings.set_num_threads(1);
    Kokkos::initialize(settings);
    
    Q = new Params();
}

Model::~Model() {
    delete G; delete C; delete Q;

    Kokkos::finalize();
}

void Model::set_cartesian_grid(py::array_t<double> x, py::array_t<double> y,
        py::array_t<double> z) {
    G = new CartesianGrid(x, y, z);
    G->Q = Q;

    C = new Camera(G, Q);
}

void Model::set_cylindrical_grid(py::array_t<double> r, py::array_t<double> phi,
        py::array_t<double> z) {
    G = new CylindricalGrid(r, phi, z);
    G->Q = Q;

    C = new Camera(G, Q);
}

void Model::set_spherical_grid(py::array_t<double> r, 
        py::array_t<double> theta, py::array_t<double> phi) {
    G = new SphericalGrid(r, theta, phi);
    G->Q = Q;

    C = new Camera(G, Q);
}

/* Run a Monte Carlo simulation to calculate the temperature throughout the 
 * grid. */

void signalHandler(int signum) {
    exit(signum);
}

void Model::thermal_mc(int nphot, bool bw, bool use_mrw, double mrw_gamma, 
        bool verbose, int nthreads) {
    // Add a signal handler.
    signal(SIGINT, signalHandler);

    // Make sure parameters are set properly.

    Q->nphot = nphot; 
    Q->bw = bw; 
    Q->use_mrw = use_mrw; 
    Q->mrw_gamma = mrw_gamma;
    Q->verbose = verbose;
    Q->scattering = false;

    // Make sure the proper number of energy arrays are allocated for the
    // threads.
    //G->add_energy_arrays(nthreads);

    // Do the thermal calculation.
    if (Q->bw)
        mc_iteration(nthreads);
    else {
        Kokkos::View<double****> told, treallyold;

        int maxniter = 10;

        Kokkos::deep_copy(told, G->temp);

        int i = 1;
        while (i <= maxniter) {
            printf("Starting iteration # %i \n\n", i);

            Kokkos::deep_copy(treallyold, told);
            Kokkos::deep_copy(told, G->temp);

            mc_iteration(nthreads);

            for (int ithread=0; ithread < (int) G->energy.size(); ithread++) {
                for (int idust=0; idust < G->nspecies; idust++) {
                    for (int ix=0; ix < G->n1; ix++) {
                        for (int iy=0; iy < G->n2; iy++) {
                            for (int iz=0; iz < G->n3; iz++) {
                                G->energy(idust,ix,iy,iz) = 0.;
                                G->energy_mrw(idust,ix,iy,iz) = 0.;
                            }
                        }
                    }
                }
            }

            if (i > 2)
                if (converged(G->temp, told, treallyold, G->nspecies, G->n1, 
                            G->n2, G->n3))
                    i = maxniter;

            i++;
            printf("\n");
        }
    }

    // Clean up the energy arrays that were calculated.
    //G->deallocate_energy_arrays();
}

void Model::scattering_mc(py::array_t<double> __lam, int nphot, bool verbose, 
        bool save, int nthreads) {
    // Add a signal handler.
    signal(SIGINT, signalHandler);

    // Make sure parameters are set properly.
    Q->nphot = nphot; 
    Q->use_mrw = false; 
    Q->verbose = verbose;

    // Make sure we've turned the scattering simulation option on.
    bool old_scattering = Q->scattering;
    Q->scattering = true;

    // Set some parameters that are going to be needed.
    auto _lam_buf = __lam.request();
    if (_lam_buf.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");
    double *lam = (double *) _lam_buf.ptr;

    Q->nnu = _lam_buf.shape[0];
    Kokkos::resize(Q->scattering_nu, Q->nnu);

    for (int i = 0; i < Q->nnu; i++)
        Q->scattering_nu(i) = c_l / (lam[i]*1.0e-4);

    // Create a scattering array in Numpy.
    //if ((int) G->scatt.size() > 1) printf("Whoops, looks like the scattering array wasn't cleaned properly.\n");

    G->initialize_scattering_array(nthreads);

    // Run the simulation for every frequency bin.
    for (int inu=0; inu<Q->nnu; inu++) {
        printf("inu = %i\n", inu);
        // Set up the right parameters.
        Q->inu = inu;
        Q->nu = Q->scattering_nu(inu);
        Q->dnu = abs(Q->scattering_nu(inu+1) - Q->scattering_nu(inu));

        G->initialize_luminosity_array(Q->nu);
        mc_iteration(nthreads);
        G->deallocate_luminosity_array();
    }

    // Reset the scattering simulation to what it was before.
    Q->scattering = old_scattering;

    // If nthreads is >1, collapse the scattering array to a single value.
    //if (nthreads > 1) G->collapse_scattering_array();

    // Clean up the appropriate grid parameters.
    if (save) {
        std::vector<size_t> extents = {1, (size_t) G->n1, (size_t) G->n2, (size_t) G->n3, (size_t) Q->nnu};
        G->_scatt.append(array_from_view<double,double*****>(G->scatt, 5, extents));

        G->deallocate_scattering_array(0);

        Q->nnu = 0;
        Kokkos::resize(Q->scattering_nu, 0);
    }
}

void Model::mc_iteration(int nthreads) {
    Kokkos::View<double> event_average("event average");
    Kokkos::View<int> photon_count("photon count");

    event_average() = 0.;
    photon_count() = 0;

    //seed1 = int(time(NULL)) ^ omp_get_thread_num();
    //seed2 = int(time(NULL)) ^ omp_get_thread_num();
    seed1 = int(time(NULL));
    seed2 = int(time(NULL));

    //for (int i=0; i<Q->nphot; i++) {
    Kokkos::parallel_for(Q->nphot, 
        KL (const int64_t i) {
        Photon *P = G->emit(i);
        P->event_count = 0;
        P->ithread = 0;

        if (Q->verbose) {
            printf("Emitting photon # %i\n", i);
            printf("Emitted with direction: %f  %f  %f\n", P->n[0], P->n[1], 
                    P->n[2]);
            printf("Emitted from a cell with temperature: %f\n", 
                    G->temp(0,P->l[0],P->l[1],P->l[2]));
            printf("Emitted with frequency: %e\n", P->nu);
        }

        if (Q->scattering)
            G->propagate_photon_scattering(P);
        else
            G->propagate_photon_full(P);

        Kokkos::atomic_add(&event_average(), P->event_count);

        delete P;
        if (Q->verbose) printf("Photon has escaped the grid.\n\n");

        Kokkos::atomic_increment(&photon_count());

        if (Kokkos::fmod(photon_count(),Q->nphot/10) == 0) printf("%i\n", photon_count());
    });

    printf("Average number of abs/scat events per photon package = %f \n", 
            event_average() / Q->nphot);

    // Make sure all of the cells are properly updated.
    if (not Q->scattering) G->update_grid();
}

void Model::run_image(py::str name, py::array_t<double> __lam, int nx, int ny, 
        double pixel_size, int nphot, double incl, double pa, double dpc, 
        int nthreads, bool raytrace_dust, bool raytrace_gas) {
    // Add a signal handler.
    signal(SIGINT, signalHandler);

    // Set the appropriate parameters.
    Q->raytrace_dust = raytrace_dust;
    Q->raytrace_gas = raytrace_gas;

    // Run a scattering simulation.
    if (raytrace_dust) scattering_mc(__lam, nphot, false, false, nthreads);

    // Make sure the lines are properly set.
    if (raytrace_gas) {
        G->set_tgas_eq_tdust();
        G->select_lines(__lam);
    }

    // Now, run the image through the camera.
    TCREATE(moo); TCLEAR(moo); TSTART(moo);
    Image *I = C->make_image(nx, ny, pixel_size, __lam, incl, pa, dpc, 
            nthreads);
    TSTOP(moo);
    printf("Time to raytrace: %f \n", TGIVE(moo));

    images[name] = I;

    // Clean up the appropriate grid parameters.
    if (raytrace_dust) {
        G->deallocate_scattering_array(0);

        Q->nnu = 0;
        Kokkos::resize(Q->scattering_nu, 0);
    }

    if (raytrace_gas) G->deselect_lines();
}

void Model::run_unstructured_image(py::str name, py::array_t<double> __lam, 
        int nx, int ny, double pixel_size, int nphot, double incl, double pa, 
        double dpc, int nthreads, bool raytrace_dust, bool raytrace_gas) {
    // Add a signal handler.
    signal(SIGINT, signalHandler);

    // Set the appropriate parameters.
    Q->raytrace_dust = raytrace_dust;
    Q->raytrace_gas = raytrace_gas;

    // Run a scattering simulation.
    if (raytrace_dust) scattering_mc(__lam, nphot, false, false, nthreads);

    // Make sure the lines are properly set.
    if (raytrace_gas) {
        G->set_tgas_eq_tdust();
        G->select_lines(__lam);
    }

    // Now, run the image through the camera.
    TCREATE(moo); TCLEAR(moo); TSTART(moo);
    UnstructuredImage *I = C->make_unstructured_image(nx, ny, pixel_size, 
            __lam, incl, pa, dpc, nthreads);
    TSTOP(moo);
    printf("Time to raytrace: %f \n", TGIVE(moo));

    images[name] = I;

    // Clean up the appropriate grid parameters.
    if (raytrace_dust) {
        G->deallocate_scattering_array(0);

        Q->nnu = 0;
        Kokkos::resize(Q->scattering_nu, 0);
    }

    if (raytrace_gas) G->deselect_lines();
}

void Model::run_circular_image(py::str name, py::array_t<double> __lam, int nr, 
        int nphi, int nphot, double incl, double pa, double dpc, int nthreads, 
        bool raytrace_dust, bool raytrace_gas) {
    // Add a signal handler.
    signal(SIGINT, signalHandler);

    // Set the appropriate parameters.
    Q->raytrace_dust = raytrace_dust;
    Q->raytrace_gas = raytrace_gas;

    // Run a scattering simulation.
    if (raytrace_dust) scattering_mc(__lam, nphot, false, false, nthreads);

    // Make sure the lines are properly set.
    if (raytrace_gas) {
        G->set_tgas_eq_tdust();
        G->select_lines(__lam);
    }

    // Now, run the image through the camera.
    TCREATE(moo); TCLEAR(moo); TSTART(moo);
    UnstructuredImage *I = C->make_circular_image(nr, nphi, __lam, incl, 
            pa, dpc, nthreads);
    TSTOP(moo);
    printf("Time to raytrace: %f \n", TGIVE(moo));

    images[name] = I;

    // Clean up the appropriate grid parameters.
    if (raytrace_dust) {
        G->deallocate_scattering_array(0);

        Q->nnu = 0;
        Kokkos::resize(Q->scattering_nu, 0);
    }

    if (raytrace_gas) G->deselect_lines();
}

void Model::run_spectrum(py::str name, py::array_t<double> __lam, int nphot, 
        double incl, double pa, double dpc, int nthreads, bool raytrace_dust, 
        bool raytrace_gas) {
    // Add a signal handler.
    signal(SIGINT, signalHandler);

    // Set the appropriate parameters.
    Q->raytrace_dust = raytrace_dust;
    Q->raytrace_gas = raytrace_gas;

    // Run a scattering simulation.
    if (raytrace_dust) scattering_mc(__lam, nphot, false, false, nthreads);

    // Make sure the lines are properly set.
    if (raytrace_gas) {
        G->set_tgas_eq_tdust();
        G->select_lines(__lam);
    }

    // Now, run the image through the camera.
    TCREATE(moo); TCLEAR(moo); TSTART(moo);
    Spectrum *S = C->make_spectrum(__lam, incl, pa, dpc, nthreads);
    TSTOP(moo);
    printf("Time to raytrace: %f \n", TGIVE(moo));

    spectra[name] = S;

    // Clean up the appropriate grid parameters.
    if (raytrace_dust) {
        G->deallocate_scattering_array(0);

        Q->nnu = 0;
        Kokkos::resize(Q->scattering_nu, 0);
    }

    if (raytrace_gas) G->deselect_lines();
}

PYBIND11_MODULE(cpu, m) {
    py::class_<Dust>(m, "Dust")
        .def(py::init<py::array_t<double>, py::array_t<double>, 
                py::array_t<double>>())
        .def_property("lam", [](const Dust &D) {return array_from_view<double,double*>(D.lam);}, nullptr)
        .def_property("nu", [](const Dust &D) {return array_from_view<double,double*>(D.nu);}, nullptr)
        .def_property("kabs", [](const Dust &D) {return array_from_view<double,double*>(D.kabs);}, nullptr)
        .def_property("ksca", [](const Dust &D) {return array_from_view<double,double*>(D.ksca);}, nullptr)
        .def_property("kext", [](const Dust &D) {return array_from_view<double,double*>(D.kext);}, nullptr)
        .def_property("albedo", [](const Dust &D) {return array_from_view<double,double*>(D.albedo);}, nullptr)
        .def("random_nu", &Dust::random_nu);

    py::class_<IsotropicDust, Dust>(m, "IsotropicDust")
        .def(py::init<py::array_t<double>, py::array_t<double>, 
                py::array_t<double>>());

    py::class_<Gas>(m, "Gas")
        .def(py::init<double, py::array_t<int>, py::array_t<double>, 
                py::array_t<double>, py::array_t<int>, py::array_t<int>, 
                py::array_t<int>, py::array_t<int>, py::array_t<double>,
                py::array_t<double>, py::array_t<double>>())
        .def_property("levels", [](const Gas &G) {return array_from_view<int,int*>(G.levels);}, nullptr)
        .def_property("energies", [](const Gas &G){return array_from_view<double,double*>(G.energies);}, nullptr)
        .def_property("weights", [](const Gas &G){return array_from_view<double,double*>(G.weights);}, nullptr)
        .def_property("J", [](const Gas &G){return array_from_view<int,int*>(G.J);}, nullptr)
        .def_property("transitions", [](const Gas &G){return array_from_view<int,int*>(G.transitions);}, nullptr)
        .def_property("up", [](const Gas &G){return array_from_view<int,int*>(G.up);}, nullptr)
        .def_property("low", [](const Gas &G){return array_from_view<int,int*>(G.low);}, nullptr)
        .def_property("A", [](const Gas &G){return array_from_view<double,double*>(G.A);}, nullptr)
        .def_property("nu", [](const Gas &G){return array_from_view<double,double*>(G.nu);}, nullptr)
        .def_property("Eu", [](const Gas &G){return array_from_view<double,double*>(G.Eu);}, nullptr);

    py::class_<Source>(m, "Source")
        .def_property("lam", [](const Source &S) {return array_from_view<double,double*>(S.lam);}, nullptr)
        .def_property("nu", [](const Source &S) {return array_from_view<double,double*>(S.nu);}, nullptr)
        .def_property("flux", [](const Source &S) {return array_from_view<double,double*>(S.Bnu);}, nullptr);

    py::class_<Star, Source>(m, "Star")
        //.def(py::init<double, double, double, double, double, double>())
        .def("set_blackbody_spectrum", &Star::set_blackbody_spectrum, 
                "Set the spectrum of the star to be a blackbody.");

    py::class_<Grid>(m, "Grid")
        .def_property("density", [](const Grid &G) {
            return array_from_view<double,double****>(G.dens);}, nullptr)
        .def_property("temperature", [](const Grid &G) {
            return array_from_view<double,double****>(G.temp);}, nullptr)
        .def_property("gas_temperature", [](const Grid &G) {
            return array_from_view<double,double****>(G.gas_temp);}, nullptr)
        .def_property("number_density", [](const Grid &G) {
            return array_from_view<double,double****>(G.number_dens);}, nullptr)
        .def_property("microturbulence", [](const Grid &G) {
            return array_from_view<double,double****>(G.microturbulence);}, nullptr)
        .def_property("velocity", [](const Grid &G) {
            return array_from_view<double,double*****>(G.velocity);}, nullptr)
        //.def("dust", [](const Grid &G, int i) {return &(G.dust(i));}, py::return_value_policy::reference, py::arg("i"))
        //.def("gas", [](const Grid &G, int i) {return &(G.gas(i));}, py::return_value_policy::reference, py::arg("i"))
        .def_readonly("scatt", &Grid::_scatt)
        //.def("sources", [](const Grid &G, int i) {return &(G.sources(i));}, py::return_value_policy::reference, py::arg("i"))
        .def("sources", [](const Grid &G) {return G.sources;}, py::return_value_policy::reference)
        .def("add_density", &Grid::add_density, 
                "Add a density layer to the Grid.")
        .def("add_number_density", &Grid::add_number_density, 
                "Add a gas density layer to the Grid.")
        .def("add_star", &Grid::add_star, "Add a star to the Grid.", 
                py::arg("x")=0., py::arg("y")=0., py::arg("z")=0., 
                py::arg("mass")=1.989e33, py::arg("radius")=69.634e9, 
                py::arg("temperature")=4000.);

    py::class_<CartesianGrid, Grid>(m, "CartesianGrid")
        .def_readonly("x", &CartesianGrid::x)
        .def_readonly("y", &CartesianGrid::y)
        .def_readonly("z", &CartesianGrid::z);

    py::class_<CylindricalGrid, Grid>(m, "CylindricalGrid")
        .def_readonly("r", &CylindricalGrid::r)
        .def_readonly("phi", &CylindricalGrid::phi)
        .def_readonly("z", &CylindricalGrid::z);

    py::class_<SphericalGrid, Grid>(m, "SphericalGrid")
        .def_readonly("r", &SphericalGrid::r)
        .def_readonly("theta", &SphericalGrid::theta)
        .def_readonly("phi", &SphericalGrid::phi);

    py::class_<Image>(m, "Image")
        .def_property("x", [](const Image &I) {return array_from_view<double,double*>(I.x);}, nullptr)
        .def_property("y", [](const Image &I) {return array_from_view<double,double*>(I.y);}, nullptr)
        .def_property("intensity", [](const Image &I) {return array_from_view<double,double*>(I.intensity, 3, {(size_t) I.nx, (size_t) I.ny, (size_t) I.nnu});;}, nullptr)
        .def_property("nu", [](const Image &I) {return array_from_view<double,double*>(I.nu);}, nullptr)
        .def_property("lam", [](const Image &I) {return array_from_view<double,double*>(I.lam);}, nullptr);

    py::class_<UnstructuredImage>(m, "UnstructuredImage")
        .def_property("x", [](const UnstructuredImage &I) {return array_from_view<double,double*>(I.x);}, nullptr)
        .def_property("y", [](const UnstructuredImage &I) {return array_from_view<double,double*>(I.y);}, nullptr)
        .def_property("intensity", [](const UnstructuredImage &I) {return array_from_view<double,double**>(I.intensity);}, nullptr)
        .def_property("nu", [](const UnstructuredImage &I) {return array_from_view<double,double*>(I.nu);}, nullptr)
        .def_property("lam", [](const UnstructuredImage &I) {return array_from_view<double,double*>(I.lam);}, nullptr);

    py::class_<Spectrum>(m, "Spectrum")
        .def_property("intensity", [](const Spectrum &S) {return array_from_view<double,double*>(S.intensity);}, nullptr)
        .def_property("nu", [](const Spectrum &S) {return array_from_view<double,double*>(S.nu);}, nullptr)
        .def_property("lam", [](const Spectrum &S) {return array_from_view<double,double*>(S.lam);}, nullptr);

    py::class_<Model>(m, "Model")
        .def(py::init<>())
        .def_readonly("grid", &Model::G)
        .def_readonly("images", &Model::images)
        .def_readonly("spectra", &Model::spectra)
        .def("set_cartesian_grid", &Model::set_cartesian_grid,
                "Setup a grid in cartesian coordinates.")
        .def("set_cylindrical_grid", &Model::set_cylindrical_grid,
                "Setup a grid in cylindrical coordinates.")
        .def("set_spherical_grid", &Model::set_spherical_grid,
                "Setup a grid in spherical coordinates.")
        .def("thermal_mc", &Model::thermal_mc, 
                "Calculate the temperature throughout the grid.",
                py::arg("nphot")=1000000, py::arg("bw")=true, 
                py::arg("use_mrw")=false, py::arg("mrw_gamma")=4, 
                py::arg("verbose")=false, py::arg("nthreads")=1)
        .def("scattering_mc", &Model::scattering_mc, py::arg("lam"), 
                py::arg("nphot")=100000, py::arg("verbose")=false, 
                py::arg("save")=true, py::arg("nthreads")=1)
        .def("run_image", &Model::run_image, "Generate an image.", 
                py::arg("name"),
                py::arg("lam"), py::arg("nx")=256, py::arg("ny")=256, 
                py::arg("pixel_size")=0.1, py::arg("nphot")=100000, 
                py::arg("incl")=0., py::arg("pa")=0., py::arg("dpc")=1., 
                py::arg("nthreads")=1, py::arg("raytrace_dust")=true, 
                py::arg("raytrace_gas")=false)
        .def("run_unstructured_image", &Model::run_unstructured_image, 
                "Generate an unstructured image.", 
                py::arg("name"),
                py::arg("lam"), py::arg("nx")=25, py::arg("ny")=25, 
                py::arg("pixel_size")=1.0, py::arg("nphot")=100000, 
                py::arg("incl")=0., py::arg("pa")=0., py::arg("dpc")=1., 
                py::arg("nthreads")=1, py::arg("raytrace_dust")=true, 
                py::arg("raytrace_gas")=false)
        .def("run_circular_image", &Model::run_circular_image, 
                "Generate an unstructured image.", 
                py::arg("name"),
                py::arg("lam"), py::arg("nr")=128, py::arg("ny")=128, 
                py::arg("nphot")=100000, py::arg("incl")=0., py::arg("pa")=0., 
                py::arg("dpc")=1., py::arg("nthreads")=1, 
                py::arg("raytrace_dust")=true, py::arg("raytrace_gas")=false)
        .def("run_spectrum", &Model::run_spectrum, "Generate a spectrum.", 
                py::arg("name"),
                py::arg("lam"), py::arg("nphot")=10000, py::arg("incl")=0,
                py::arg("pa")=0, py::arg("dpc")=1., py::arg("nthreads")=1, 
                py::arg("raytrace_dust")=true, py::arg("raytrace_gas")=false);
}
