#include "star.h"

/* Functions to set up the sources. */

Star::Star() {
    random_pool = new Kokkos::Random_XorShift64_Pool<>(/*seed=*/12345);
}

void Star::set_properties(double x, double y, double z, double _mass, double _radius, \
        double _temperature) {

    r[0] = x;
    r[1] = y;
    r[2] = z;
    mass = _mass;
    radius = _radius;
    temperature = _temperature;

    spectrum_set = false;
}

Star::~Star() {
}

void Star::set_blackbody_spectrum(py::array_t<double> __lam) {

    // Load the array buffers to get the proper setup info.

    auto _lam_buf = __lam.request();

    if (_lam_buf.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    //lam = (double *) _lam_buf.ptr;
    nnu = _lam_buf.shape[0];

    Kokkos::resize(lam, nnu);
    Kokkos::deep_copy(lam, view_from_array(__lam));

    // Set up the frequency array.

    Kokkos::resize(nu, nnu);

    Kokkos::parallel_for(nnu, KL (const size_t i) {
        nu(i) = c_l / lam(i);
    });

    // And set up the spectrum.

    Kokkos::resize(Bnu, nnu);

    Kokkos::parallel_for(nnu, KL (const size_t i) {
        Bnu(i) = planck_function(nu(i), temperature);
    });

    luminosity = 4*pi*radius*radius*sigma*temperature*temperature*temperature*
            temperature;

    // Calculate the lookup array for the cumulative spectrum.

    Kokkos::resize(random_nu_CPD, nnu);
    Kokkos::deep_copy(random_nu_CPD, cumulative_integrate(Bnu, nu, nnu));

    // Indicate that the spectrum was calculated.

    spectrum_set = true;
}

/* Emit a photon from the source. */

Photon *Star::emit(int nphot) {
    Photon *P = new Photon();

    double theta = pi*random_number(random_pool);
    double phi = 2*pi*random_number(random_pool);

    P->r[0] = radius*sin(theta)*cos(phi);
    P->r[1] = radius*sin(theta)*sin(phi);
    P->r[2] = radius*cos(theta);

    Vector<double, 3> r_hat, theta_hat, phi_hat;

    r_hat[0] = sin(theta)*cos(phi);
    r_hat[1] = sin(theta)*sin(phi);
    r_hat[2] = cos(theta);
    theta_hat[0] = cos(theta)*cos(phi);
    theta_hat[1] = cos(theta)*sin(phi);
    theta_hat[2] = -sin(theta);
    phi_hat[0] = -sin(phi);
    phi_hat[1] = cos(phi);
    phi_hat[2] = 0;

    double cost = random_number(random_pool);
    double sint = sqrt(1-pow(cost,2));
    phi = 2*pi*random_number(random_pool);

    P->n = cost*r_hat + sint*cos(phi)*phi_hat + sint*sin(phi)*theta_hat;

    P->invn[0] = 1.0/P->n[0];
    P->invn[1] = 1.0/P->n[1];
    P->invn[2] = 1.0/P->n[2];
    P->l[0] = -1;
    P->l[1] = -1;
    P->l[2] = -1;

    P->energy = luminosity / nphot;

    P->nu = random_nu();

    return P;
};

Photon *Star::emit(double _nu, double _dnu, int nphot) {
    Photon *P = new Photon();

    double theta = pi*random_number(random_pool);
    double phi = 2*pi*random_number(random_pool);

    P->r[0] = radius*sin(theta)*cos(phi);
    P->r[1] = radius*sin(theta)*sin(phi);
    P->r[2] = radius*cos(theta);

    Vector<double, 3> r_hat, theta_hat, phi_hat;

    r_hat[0] = sin(theta)*cos(phi);
    r_hat[1] = sin(theta)*sin(phi);
    r_hat[2] = cos(theta);
    theta_hat[0] = cos(theta)*cos(phi);
    theta_hat[1] = cos(theta)*sin(phi);
    theta_hat[2] = -sin(theta);
    phi_hat[0] = -sin(phi);
    phi_hat[1] = cos(phi);
    phi_hat[2] = 0;

    double cost = random_number(random_pool);
    double sint = sqrt(1-pow(cost,2));
    phi = 2*pi*random_number(random_pool);

    P->n = cost*r_hat + sint*cos(phi)*phi_hat + sint*sin(phi)*theta_hat;

    P->invn[0] = 1.0/P->n[0];
    P->invn[1] = 1.0/P->n[1];
    P->invn[2] = 1.0/P->n[2];
    P->l[0] = -1;
    P->l[1] = -1;
    P->l[2] = -1;

    P->nu = _nu;

    P->energy = 4*pi*pi*radius*radius*flux(_nu) / nphot;

    return P;
};

Ray *Star::emit_ray(Kokkos::View<double*> _nu, int _nnu, double _pixelsize, \
        Vector<double, 3> _n, int nphot) {
    Ray *R = new Ray();

    double theta = pi*random_number(random_pool);
    double phi = 2*pi*random_number(random_pool);

    R->r[0] = radius*sin(theta)*cos(phi);
    R->r[1] = radius*sin(theta)*sin(phi);
    R->r[2] = radius*cos(theta);

    R->n = -_n;

    R->invn[0] = 1.0/R->n[0];
    R->invn[1] = 1.0/R->n[1];
    R->invn[2] = 1.0/R->n[2];
    R->l[0] = -1;
    R->l[1] = -1;
    R->l[2] = -1;

    // Now get the total energy of the photon.
    R->nnu = _nnu;
    Kokkos::resize(R->nu, _nnu);
    Kokkos::resize(R->tau, _nnu);
    Kokkos::resize(R->intensity, _nnu);
    for (int i = 0; i < _nnu; i++) {
        R->nu(i) = _nu(i);
        R->tau(i) = 0;
        R->intensity(i) = flux(_nu[i]) * pi*radius*radius / 
                (_pixelsize * _pixelsize) / nphot;
    }

    R->pixel_size = _pixelsize;
    R->pixel_too_large = false;

    return R;
};

Ray *Star::emit_ray(Kokkos::View<double*> _nu, int _nnu, Vector<double, 3> _n, int nphot) {
    Ray *R = new Ray();

    double theta = pi*random_number(random_pool);
    double phi = 2*pi*random_number(random_pool);

    R->r[0] = radius*sin(theta)*cos(phi);
    R->r[1] = radius*sin(theta)*sin(phi);
    R->r[2] = radius*cos(theta);

    R->n = -_n;

    R->invn[0] = 1.0/R->n[0];
    R->invn[1] = 1.0/R->n[1];
    R->invn[2] = 1.0/R->n[2];
    R->l[0] = -1;
    R->l[1] = -1;
    R->l[2] = -1;

    // Now get the total energy of the photon.
    R->nnu = _nnu;
    Kokkos::resize(R->nu, _nnu);
    Kokkos::resize(R->tau, _nnu);
    Kokkos::resize(R->intensity, _nnu);
    for (int i = 0; i < _nnu; i++) {
        R->nu(i) = _nu(i);
        R->tau(i) = 0;
        R->intensity(i) = flux(_nu[i]) * pi*radius*radius / nphot;
    }

    R->pixel_size = 0.0;
    R->pixel_too_large = false;

    return R;
};

/* Get a random frequency drawn from the spectrum of the source. */

double Star::random_nu() {
    double freq;
    double ksi = random_number(random_pool);

    for (int i=1; i<nnu; i++) {
        if (random_nu_CPD(i) > ksi) {
            freq = random_number(random_pool) * (nu(i) - nu(i-1)) + nu(i-1);
            break;
        }
    }

    return freq;
};

double Star::intercept_distance(Photon *P) {
    double s = HUGE_VAL;

    double r = P->r.norm();

    if (!equal(r, radius, EPSILON)) {
        double b = P->r*P->n;
        double c = r*r - radius*radius;
        double d = b*b - c;

        if (d >= 0) {
            double sr1 = -b + sqrt(d);
            if ((sr1 < s) && (sr1 > 0)) s = sr1;
            double sr2 = -b - sqrt(d);
            if ((sr2 < s) && (sr2 > 0)) s = sr2;
        }
    }

    return s;
}

double Star::flux(double freq) {
    int l = find_in_arr(freq, nu, nnu);

    double dBnudnu = (Bnu(l+1) - Bnu(l))/(nu(l+1) - nu(l));

    double flux = dBnudnu*(freq-nu(l))+Bnu(l);

    return flux;
};

void Star::reemit(Photon *P) {
    double theta = pi*random_number(random_pool);
    double phi = 2*pi*random_number(random_pool);

    P->r[0] = radius*sin(theta)*cos(phi);
    P->r[1] = radius*sin(theta)*sin(phi);
    P->r[2] = radius*cos(theta);

    Vector<double, 3> r_hat, theta_hat, phi_hat;

    r_hat[0] = sin(theta)*cos(phi);
    r_hat[1] = sin(theta)*sin(phi);
    r_hat[2] = cos(theta);
    theta_hat[0] = cos(theta)*cos(phi);
    theta_hat[1] = cos(theta)*sin(phi);
    theta_hat[2] = -sin(theta);
    phi_hat[0] = -sin(phi);
    phi_hat[1] = cos(phi);
    phi_hat[2] = 0;

    double cost = random_number(random_pool);
    double sint = sqrt(1-pow(cost,2));
    phi = 2*pi*random_number(random_pool);

    P->n = cost*r_hat + sint*cos(phi)*phi_hat + sint*sin(phi)*theta_hat;

    P->invn[0] = 1.0/P->n[0];
    P->invn[1] = 1.0/P->n[1];
    P->invn[2] = 1.0/P->n[2];
    P->l[0] = -1;
    P->l[1] = -1;
    P->l[2] = -1;

    P->nu = random_nu();
};
