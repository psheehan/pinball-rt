#ifndef SOURCES_CC
#define SOURCES_CC

#include <stdlib.h>
#include "vector.cc"
#include "dust.cc"
#include "photon.cc"
#include "misc.cc"

struct Source {
    double mass;
    double radius;
    double temperature;
    double luminosity;
    Vector<double, 3> r;
    double *nu;
    double *Bnu;
    int nnu;

    Photon *emit(int nphot, int nspecies, Dust *species);
    double intercept_distance(Photon *P);
    double random_nu();
};

/* Emit a photon from the source. */

Photon *Source::emit(int nphot, int nspecies, Dust *species) {
    Photon *P = new Photon();

    double theta = pi*random_number();
    double phi = 2*pi*random_number();

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

    double cost = random_number();
    double sint = sqrt(1-pow(cost,2));
    phi = 2*pi*random_number();

    P->n = cost*r_hat + sint*cos(phi)*phi_hat + sint*sin(phi)*theta_hat;

    P->invn[0] = 1.0/P->n[0];
    P->invn[1] = 1.0/P->n[1];
    P->invn[2] = 1.0/P->n[2];
    P->l[0] = -1;
    P->l[1] = -1;
    P->l[2] = -1;

    P->energy = luminosity / nphot;

    P->nu = random_nu();

    double *current_kext = new double[nspecies];
    double *current_albedo = new double[nspecies];

    for (int i=0; i<nspecies; i++) {
        current_kext[i] = species[i].opacity(P->nu);
        current_albedo[i] = species[i].albdo(P->nu);
    }

    P->current_kext = current_kext;
    P->current_albedo = current_albedo;

    return P;
};

/* Get a random frequency drawn from the spectrum of the source. */

double Source::random_nu() {
    double freq;
    double norm = -pi/(sigma*pow(temperature,4));
    double ksi = random_number();

    double tot = 0.0;
    for (int i=0; i<nnu-2; i++) {
        tot += 0.5*(nu[nnu-i-1]-nu[nnu-i-2])*(Bnu[nnu-i-1]+Bnu[nnu-i-2]);

        double Prob = tot*norm;

        if (Prob > ksi) {
            freq = random_number()*(nu[nnu-i-1]-nu[nnu-i-2])+nu[nnu-i-2];
            break;
        }
    }

    return freq;
};

double Source::intercept_distance(Photon *P) {
    double s = HUGE_VAL;

    double r = P->r.norm();

    if (!equal(r, radius, 1.0e-6)) {
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

#endif
