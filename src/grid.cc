#include "grid.h"

// Initialize the grid from numpy arrays directly.

Grid::Grid(py::array_t<double> __w1, py::array_t<double> __w2,
            py::array_t<double> __w3) {

    random_pool = new Kokkos::Random_XorShift64_Pool<>(/*seed=*/12345);

    // Load the array buffers to get the proper setup info.

    auto _w1_buf = __w1.unchecked<1>(); auto _w2_buf = __w2.unchecked<1>(); 
    auto _w3_buf = __w3.unchecked<1>();

    if (_w1_buf.ndim() != 1 || _w2_buf.ndim() != 1 || _w3_buf.ndim() != 1)
        throw std::runtime_error("Number of dimensions must be one");

    // Now get the correct format.
    
    nw1 = _w1_buf.shape(0), nw2 = _w2_buf.shape(0), nw3 = _w3_buf.shape(0);
    n1 = _w1_buf.shape(0)-1, n2 = _w2_buf.shape(0)-1, n3 = _w3_buf.shape(0)-1;
    
    Kokkos::resize(w1, nw1);
    Kokkos::deep_copy(w1, view_from_array(__w1));
    Kokkos::resize(w2, nw2);
    Kokkos::deep_copy(w2, view_from_array(__w2));
    Kokkos::resize(w3, nw3);
    Kokkos::deep_copy(w3, view_from_array(__w3));

    // Set up the volume of each cell.

    Kokkos::resize(volume, n1, n2, n3);

    // Make sure the number of species and sources are set correctly.

    nspecies = 0; nsources = 0; ngases = 0;

    // Initialize the uses_mrw array.

    Kokkos::resize(uses_mrw, n1, n2, n3);
    Kokkos::deep_copy(uses_mrw, -1);
    //for (int i=0; i<n1; i++)
    //    for (int j=0; j < n2; j++)
    //        for (int k=0; k < n3; k++)
    //            uses_mrw(i,j,k) = -1;
}

/* Functions to set up the grid. */

/*Grid::Grid(int _n1, int _n2, int _n3, int _nw1, int _nw2, int _nw3, 
        double *_w1, double *_w2, double *_w3, double *_volume, 
        bool _mirror_symmetry) {

    n1 = _n1;
    n2 = _n2;
    n3 = _n3;
    nw1 = _nw1;
    nw2 = _nw2;
    nw3 = _nw3;
    w1 = _w1;
    w2 = _w2;
    w3 = _w3;

    mirror_symmetry = _mirror_symmetry;

    volume = _volume;
}*/

Grid::~Grid() {
    // Free the physical parameters;

    //dust.clear();

    // Clear the gas.
    
    //gas.clear();

    // Clear the sources.

    //sources.clear();
    delete sources;
}

//void Grid::add_density(double *_dens, double *_temp, double *_mass, 
//        Dust *D) {
void Grid::add_density(py::array_t<double> ___dens, Dust *D) {

    int idust = dens.extent(0);
    Kokkos::resize(dens, dens.extent(0)+1, n1, n2, n3);
    Kokkos::resize(temp, temp.extent(0)+1, n1, n2, n3);
    Kokkos::resize(energy, energy.extent(0)+1, n1, n2, n3);
    Kokkos::resize(energy_mrw, energy_mrw.extent(0)+1, n1, n2, n3);
    Kokkos::resize(mass, mass.extent(0)+1, n1, n2, n3);
    Kokkos::resize(rosseland_mean_extinction, rosseland_mean_extinction.extent(0)+1, n1, n2, n3);
    Kokkos::resize(planck_mean_opacity, planck_mean_opacity.extent(0)+1, n1, n2, n3);

    auto ___dens_access = ___dens.unchecked<3>();
    Kokkos::View<double****>::HostMirror h_dens = Kokkos::create_mirror_view(dens);
    Kokkos::deep_copy(h_dens, dens);
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < n3; k++) {
                h_dens(idust,i,j,k) = ___dens_access(i,j,k);
                if (h_dens(nspecies,i,j,k) < 1.0e-99) h_dens(nspecies,i,j,k) = 1.0e-99;
            }
        }
    }
    Kokkos::deep_copy(dens, h_dens);

    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{n1,n2,n3}), 
            KL (const size_t i, const size_t j, const size_t k) {
                temp(idust,i,j,k) = 0.1;
                mass(nspecies,i,j,k) = dens(nspecies,i,j,k) * volume(i,j,k);
                rosseland_mean_extinction(nspecies,i,j,k) = 
                        D->rosseland_mean_extinction(temp(nspecies,i,j,k));
                planck_mean_opacity(nspecies,i,j,k) = D->planck_mean_opacity(
                        temp(nspecies,i,j,k));
                energy(nspecies,i,j,k) = 0;
                energy_mrw(nspecies,i,j,k) = 0;
            });

    //std::vector<size_t> extents = {(size_t) nspecies+1, (size_t) n1, (size_t) n2, (size_t) n3};
    //_temp = array_from_view<double,double****>(temp, 4, extents);

    // Add the dust to the list of dust classes.

    //dust.push_back(D);
    //Kokkos::resize(dust, dust.extent(0)+1);
    //dust(nspecies).copy(D);
    dust = D;
    nspecies++;
}

void Grid::add_number_density(py::array_t<double> ___number_dens, 
        py::array_t<double> ___velocity, py::array_t<double> ___microturbulence,
        Gas *G) {

    Kokkos::resize(number_dens, number_dens.extent(0)+1, n1, n2, n3);
    Kokkos::resize(gas_temp, gas_temp.extent(0)+1, n1, n2, n3);
    Kokkos::resize(velocity, velocity.extent(0)+1, n1, n2, n3);
    Kokkos::resize(microturbulence, microturbulence.extent(0)+1, n1, n2, n3);

    Kokkos::View<double****>::HostMirror h_number_dens = Kokkos::create_mirror_view(number_dens);
    Kokkos::View<double****>::HostMirror h_gas_temp = Kokkos::create_mirror_view(gas_temp);
    Kokkos::View<double*****>::HostMirror h_velocity = Kokkos::create_mirror_view(velocity);
    Kokkos::View<double****>::HostMirror h_microturbulence = Kokkos::create_mirror_view(microturbulence);

    Kokkos::deep_copy(h_number_dens, number_dens);
    Kokkos::deep_copy(h_gas_temp, gas_temp);
    Kokkos::deep_copy(h_velocity, velocity);
    Kokkos::deep_copy(h_microturbulence, microturbulence);

    auto number_dens_access = ___number_dens.unchecked<3>();
    auto velocity_access = ___velocity.unchecked<4>();
    auto microturbulence_access = ___microturbulence.unchecked<3>();
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < n3; k++) {
                int icell = i*n2*n3 + j*n3 + k;
                h_number_dens(ngases,i,j,k) = number_dens_access(i,j,k);
                h_gas_temp(ngases,i,j,k) = 0.;
                h_microturbulence(ngases,i,j,k) = microturbulence_access(i,j,k);
                h_velocity(ngases,0,i,j,k) = velocity_access(0,i,j,k);
                h_velocity(ngases,1,i,j,k) = velocity_access(1,i,j,k);
                h_velocity(ngases,2,i,j,k) = velocity_access(2,i,j,k);
            }
        }
    }

    Kokkos::deep_copy(number_dens, h_number_dens);
    Kokkos::deep_copy(gas_temp, h_gas_temp);
    Kokkos::deep_copy(velocity, h_velocity);
    Kokkos::deep_copy(microturbulence, h_microturbulence);

    // Add the dust to the list of dust classes.

    //Kokkos::resize(gas, gas.extent(0)+1);
    //gas(ngases).copy(G);
    gas = G;
    ngases++;
}

//void Grid::add_source(Source *S) {
void Grid::add_star(double x, double y, double z, double _mass, double _radius, 
            double _temperature) {
    //TODO: calculate the lookup tables somehow!
    
    //Kokkos::resize(sources, sources.extent(0)+1);
    //sources(nsources).set_properties(x, y, z, _mass, _radius, _temperature);
    sources = new Star();
    sources->set_properties(x, y, z, _mass, _radius, _temperature);
    nsources++;
}

/* Functions to manage the scattering array. */

/*void Grid::add_scattering_array(py::array_t<double> ___scatt, int nthreads) {
    // Add the array to the Python-connected list of scattering phase
    // functions.

    _scatt.append(___scatt);

    // Get the buffer and create an array useful for C++.

    auto _scatt_buf = ___scatt.request();

    scatt.push_back((double *) _scatt_buf.ptr);
}*/

void Grid::initialize_scattering_array(int nthreads) {
    // Create a 4D scattering array for each of the separate threads.
    Kokkos::resize(scatt, 1, n1, n2, n3, Q->nnu);
}

/*void Grid::collapse_scattering_array() {
    for (int ithread = 1; ithread < (int) scatt.size(); ithread++)
        for (int icell=0; icell < n1*n2*n3*Q->nnu; icell++)
            scatt[0][icell] += scatt[ithread][icell];

    deallocate_scattering_array(1);
}*/

void Grid::deallocate_scattering_array(int start) {
    Kokkos::resize(scatt, 0, 0, 0, 0, 0);
}

/* Functions to manage the energy array. */

/*void Grid::add_energy_arrays(int nthreads) {
    // Create a 3D energy array for each of the separate threads.

    for (int ithread = 1; ithread < nthreads; ithread++) {
        std::vector<double*> _energy = {};
        energy.push_back(_energy);
        for (int idust = 0; idust < nspecies; idust++) {
            energy[ithread].push_back(new double[n1*n2*n3]);
            for (int icell = 0; icell < n1*n2*n3; icell++)
                energy[ithread][idust][icell] = 0.;
        }
    }
}*/

/*void Grid::deallocate_energy_arrays() {
    for (int ithread = 1; ithread < (int) energy.size(); ithread++) {
        for (int idust = 0; idust < nspecies; idust++)
            delete[] energy[ithread][idust];
        energy[ithread].clear();
    }
    energy.erase(energy.begin()+1, energy.end());
}*/

/* Functions to manage the luminosity array. */

void Grid::initialize_luminosity_array() {
    total_lum = 0.;

    Kokkos::resize(luminosity, nspecies, n1*n2*n3);

    for (int idust = 0; idust<nspecies; idust++) {
        for (int i = 0; i<n1; i++) {
            for (int j = 0; j<n2; j++) {
                for (int k = 0; k<n3; k++) {
                    luminosity(idust,i,j,k) = cell_lum(idust, i, j, k);

                    total_lum += luminosity(idust,i,j,k);
                }
            }
        }
    }
}

void Grid::initialize_luminosity_array(double nu) {
    total_lum = 0.;

    Kokkos::resize(luminosity, nspecies, n1, n2, n3);

    for (int idust = 0; idust<nspecies; idust++) {
        for (int i = 0; i<n1; i++) {
            for (int j = 0; j<n2; j++) {
                for (int k = 0; k<n3; k++) {
                    luminosity(idust,i,j,k) = cell_lum(idust, i, j, k, nu);

                    total_lum += luminosity(idust,i,j,k);
                }
            }
        }
    }
}

void Grid::deallocate_luminosity_array() {
    Kokkos::resize(luminosity, 0, 0, 0, 0);
}

/* Emit a photon from the grid. */

Photon *Grid::emit(int iphot) {
    /* Cycle through the various stars, having them emit photons one after 
     * another. This way each source will get the same nuber of photons 
     * +/- 1. */
    int isource = 0;
    int photons_per_source = Q->nphot;
    if (Q->scattering) {
        isource = fmod(iphot, nsources+1);
        photons_per_source = int(Q->nphot/(nsources+1));
    }
    else if (nsources > 1) {
        isource = fmod(iphot, nsources);
        photons_per_source = int(Q->nphot/nsources);
    }

    Photon *P;
    if (Q->scattering) {
        if (isource == nsources)
            P = emit(Q->scattering_nu(Q->inu), Q->dnu, photons_per_source);
        else
            //P = sources(isource).emit(Q->scattering_nu(Q->inu), Q->dnu, 
            P = sources->emit(Q->scattering_nu(Q->inu), Q->dnu, 
                    photons_per_source);
    }
    else
        //P = sources(isource).emit(photons_per_source);
        P = sources->emit(photons_per_source);

    /* Calculate kext and albedo at the photon's current frequency for all
     * dust species. */

    /*Kokkos::resize(P->current_kext, nspecies);
    Kokkos::resize(P->current_albedo, nspecies);
    for (int i=0; i<nspecies; i++) {
        P->current_kext(i) = dust(i).opacity(P->nu);
        P->current_albedo(i) = dust(i).albdo(P->nu);
    }*/

    /* Check the photon's location in the grid. */
    P->l = photon_loc(P);

    return P;
}

Photon *Grid::emit(double _nu, double _dnu, int nphot) {
    Photon *P = new Photon();

    int idust, ix, iy, iz;

    // Get the cell that randomly emits.

    double rand = random_number(random_pool);
    double cum_lum = 0.;

    for (idust = 0; idust<nspecies; idust++) {
        for (ix = 0; ix<n1; ix++) {
            for (iy = 0; iy<n2; iy++) {
                for (iz = 0; iz<n3; iz++) {
                    cum_lum += luminosity(idust,ix,iy,iz) / total_lum;

                    if (cum_lum >= rand)
                        break;
                }
                if (cum_lum >= rand)
                    break;
            }
            if (cum_lum >= rand)
                break;
        }
        if (cum_lum >= rand)
            break;
    }

    // Now set up the photon.

    P->r = random_location_in_cell(ix, iy, iz);

    double theta = pi*random_number(random_pool);
    double phi = 2*pi*random_number(random_pool);

    P->n[0] = sin(theta)*cos(phi);
    P->n[1] = sin(theta)*sin(phi);
    P->n[2] = cos(theta);

    P->invn[0] = 1.0/P->n[0];
    P->invn[1] = 1.0/P->n[1];
    P->invn[2] = 1.0/P->n[2];
    P->l[0] = -1;
    P->l[1] = -1;
    P->l[2] = -1;

    P->nu = _nu;

    P->energy = total_lum / nphot;

    return P;
}

/* Linker function to the dust absorb function. */

void Grid::absorb(Photon *P, int idust) {
    // If we're doing a Bjorkman & Wood simulation, update the cell to
    // find its new temperature before continuing.
    if (Q->bw) {
        update_grid(P->l, P->cell_index);
    }


    // Determine which dust species should do the absorption.
    idust = 0;
    if (nspecies == 1) {
        idust = 0;
    }
    else {
        double kabs_tot = 0;
        double *kabs_cum = new double[nspecies];
        for (int i=0; i<nspecies; i++) {
            //kabs_tot += (1 - P->current_albedo(i))*P->current_kext(i)*
            kabs_tot += (1 - dust->albdo(P->nu))*dust->opacity(P->nu)*
                dens(i,P->l[0],P->l[1],P->l[2]);
            kabs_cum[i] = kabs_tot;
        }

        double rand = random_number(random_pool);
        for (int i=0; i<nspecies; i++) {
            if (rand < kabs_cum[i] / kabs_tot) {
                idust = i;
                break;
            }
        }
        delete[] kabs_cum;
    }

    // Now do the absorption.

    //dust(idust).absorb(P, temp(idust,P->l[0],P->l[1],P->l[2]), Q->bw);
    dust->absorb(P, temp(idust,P->l[0],P->l[1],P->l[2]), Q->bw);

    // Update the photon's arrays of kext and albedo since P->nu has changed
    // upon absorption.
    /*for (int i=0; i<nspecies; i++) {
        P->current_kext(i) = dust(i).opacity(P->nu);
        P->current_albedo(i) = dust(i).albdo(P->nu);
    }*/

    // Check the photon's location again because there's a small chance that 
    // the photon was absorbed on a wall, and if it was we may need to update
    // which cell it is in if the direction has changed.
    P->l = photon_loc(P);
}

void Grid::absorb_mrw(Photon *P, int idust) {
    // Update the temperature after absorbing all that energy.
    if (Q->bw) update_grid(P->l, P->cell_index);

    // Determine which dust species should do the absorption.
    idust = 0;
    if (nspecies == 1) {
        idust = 0;
    }
    else {
        double kabs_tot = 0;
        double *kabs_cum = new double[nspecies];
        for (int i=0; i<nspecies; i++) {
            //kabs_tot += (1 - P->current_albedo(i))*P->current_kext(i)*
            kabs_tot += (1 - dust->albdo(P->nu))*dust->opacity(P->nu)*
                dens(i,P->l[0],P->l[1],P->l[2]);
            kabs_cum[i] = kabs_tot;
        }

        double rand = random_number(random_pool);
        for (int i=0; i<nspecies; i++) {
            if (rand < kabs_cum[i] / kabs_tot) {
                idust = i;
                break;
            }
        }
        delete[] kabs_cum;
    }

    // Now do the absorption.

    //dust(idust).absorb_mrw(P, temp(idust,P->l[0],P->l[1],P->l[2]), Q->bw);
    dust->absorb_mrw(P, temp(idust,P->l[0],P->l[1],P->l[2]), Q->bw);

    // Update the photon's arrays of kext and albedo since P->nu has changed
    // upon absorption.
    /*for (int i=0; i<nspecies; i++) {
        P->current_kext(i) = dust(i).opacity(P->nu);
        P->current_albedo(i) = dust(i).albdo(P->nu);
    }*/

    // Check the photon's location again because there's a small chance that 
    // the photon was absorbed on a wall, and if it was we may need to update
    // which cell it is in if the direction has changed.
    P->l = photon_loc(P);
}

/* Linker function to the dust scatter function. */

void Grid::scatter(Photon *P, int idust) {
    // Determine which dust species should do the absorption.
    idust = 0;
    if (nspecies == 1) {
        idust = 0;
    }
    else {
        double ksca_tot = 0;
        double *ksca_cum = new double[nspecies];
        for (int i=0; i<nspecies; i++) {
            //ksca_tot += P->current_albedo(i)*P->current_kext(i)*
            ksca_tot += dust->albdo(P->nu)*dust->opacity(P->nu)*
                dens(i,P->l[0],P->l[1],P->l[2]);
            ksca_cum[i] = ksca_tot;
        }

        double rand = random_number(random_pool);
        for (int i=0; i<nspecies; i++) {
            if (rand < ksca_cum[i] / ksca_tot) {
                idust = i;
                break;
            }
        }
        delete[] ksca_cum;
    }

    // Now do the absorption.

    //dust(idust).scatter(P);
    dust->scatter(P);

    // Check the photon's location again because there's a small chance that 
    // the photon was absorbed on a wall, and if it was we may need to update
    // which cell it is in if the direction has changed.
    P->l = photon_loc(P);
}

/* Propagate a photon through the grid until it escapes. */

void Grid::propagate_photon_full(Photon *P) {
    P->same_cell_count = 0;

    while (in_grid(P)) {
        // Determin the optical depth that the photon can travel until it's
        // next interaction.
        double tau = -log(1-random_number(random_pool));

        // Figure out what that next action is, absorption or scattering. This
        // is figured out early for the sake of the continuous absorption
        // method.
        double albedo;
        if (nspecies == 1) {
            //albedo = P->current_albedo(0);
            albedo = dust->albdo(P->nu);
        }
        else {
            double ksca_tot = 0;
            double kext_tot = 0;
            for (int i=0; i<nspecies; i++) {
                //ksca_tot += P->current_albedo(i)*P->current_kext(i)*
                ksca_tot += dust->albdo(P->nu)*dust->opacity(P->nu)*
                    dens(i,P->l[0],P->l[1],P->l[2]);
                //kext_tot += P->current_kext(i)*
                kext_tot += dust->opacity(P->nu)*
                    dens(i,P->l[0],P->l[1],P->l[2]);
            }
            albedo = ksca_tot / kext_tot;
        }

        bool absorb_photon = random_number(random_pool) > albedo;

        // Move the photon to the point of it's next interaction.
        if (Q-> scattering)
            propagate_photon(P, tau, false);
        else
            propagate_photon(P, tau, absorb_photon);

        // If the photon is still in the grid when it reaches it's 
        // destination...
        if (in_grid(P)) {
            // Increase the event counter.
            P->event_count += 1;

            // If the next interaction is absorption...
            if (absorb_photon) {
                absorb(P, 0);
                // If we've asked for verbose output, print some info.
                if (Q->verbose) {
                    printf("Absorbing photon at %i  %i  %i\n", P->l[0],
                            P->l[1], P->l[2]);
                    printf("Absorbed in a cell with temperature: %f\n",
                            temp(0,P->l[0],P->l[1],P->l[2]));
                    printf("Re-emitted with direction: %f  %f  %f\n",
                            P->n[0], P->n[1], P->n[2]);
                    printf("Re-emitted with frequency: %e\n", P->nu);
                }
            }
            // Otherwise, scatter the photon.
            else {
                // Now scatter the photon.
                scatter(P, 0);
                // If we've asked for verbose output, print some info.
                if (Q->verbose) {
                    printf("Scattering photon at cell  %i  %i  %i\n",
                            P->l[0], P->l[1], P->l[2]);
                    printf("Scattered with direction: %f  %f  %f\n",
                            P->n[0], P->n[1], P->n[2]);
                }
            }

            // If we've enabled MRW, try running an MRW step.
            // Need to do the in_grid check again because the absorption
            // or scattering may have been on the outermost wall, putting
            // it outside of the grid.
            if (Q->use_mrw && in_grid(P) && ((P->same_cell_count > 400) || 
                        (uses_mrw(P->l[0],P->l[1],P->l[2]) > 0))) {
                // First check whether we need to update the cell properties,
                // i.e. the last event was scattering.
                if (Q->bw and not absorb_photon) update_grid(P->l, P->cell_index);

                // Check whether the cell is thick enough for MRW.

                double alpha = 0.0;
                for (int idust = 0; idust<nspecies; idust++)
                    alpha += rosseland_mean_extinction(idust,P->l[0],P->l[1],P->l[2]) * 
                        dens(idust,P->l[0],P->l[1],P->l[2]);

                double dmin = smallest_wall_size(P);

                if (alpha * dmin > 20) {
                    uses_mrw(P->l[0],P->l[1],P->l[2]) = 1.0;

                    // Propagate the photon.
                    propagate_photon_mrw(P);

                    // Absorb the photon
                    absorb_mrw(P, 0);
                }
            }
        }
    }
}

/* Propagate a photon through the grid a distance equivalent to tau. */

void Grid::propagate_photon(Photon *P, double tau, bool absorb) {

    bool absorbed_by_source = false;
    int i = 0;
    while ((tau > EPSILON) && (in_grid(P))) {
        // Calculate the distance to the next wall.
        double s1 = next_wall_distance(P);

        // Calculate how far the photon can go with the current tau.
        double alpha = 0;
        for (int idust = 0; idust<nspecies; idust++)
            //alpha += P->current_kext(idust)*
            alpha += dust->opacity(P->nu)*
                dens(idust,P->l[0],P->l[1],P->l[2]);

        double s2 = tau/alpha;

        // Determine whether to move to the next wall or to the end of tau.
        double s = s1;
        if (s2 < s) {
            s = s2;
            P->same_cell_count++;
        } else
            P->same_cell_count = 0;

        // Calculate how far the photon can go before running into a source.
        int isource_intercept;
        for (int isource=0; isource<nsources; isource++) {
            //double s3 = sources(isource).intercept_distance(P);
            double s3 = sources->intercept_distance(P);

            if (s3 < s) {
                s = s3;
                absorbed_by_source = true;
                isource_intercept = isource;
            }
        }

        // Continuously absorb the photon's energy, if the end result of the
        // current trajectory is absorption.
        if (absorb) {
            for (int idust=0; idust<nspecies; idust++)
                Kokkos::atomic_add(&energy(idust,P->l[0],P->l[1],P->l[2]),
                    //P->energy*s*P->current_kext(idust)*
                    P->energy*s*dust->opacity(P->nu)*
                    dens(idust,P->l[0],P->l[1],P->l[2]));
        }

        // Remvove the tau we've used up with this stepl
        tau -= s*alpha;

        // Move the photon to it's new position.
        P->move(s);

        // If the photon moved to the next cell, update it's location.
        if (s1 < s2) P->l = photon_loc(P);
        i++;

        // If we've asked for verbose, print some information out.
        if (Q->verbose) {
            printf("%2i  %7.4f  %i  %7.4f  %7.4f  %7.4f\n", i, tau, P->l[0],
                    P->r[0]/au, s1*P->n[0]/au, s2*P->n[0]/au);
            printf("%14i  %7.4f  %7.4f  %7.4f\n", P->l[1], P->r[1]/au, 
                    s1*P->n[1]/au, s2*P->n[1]/au);
            printf("%14i  %7.4f  %7.4f  %7.4f\n", P->l[2], P->r[2]/au, 
                    s1*P->n[2]/au, s2*P->n[2]/au);
        }

        // If the distance to the star is the shortest distance, kill the 
        // photon.
        if (absorbed_by_source) {
            //sources(isource_intercept).reemit(P);
            sources->reemit(P);
            P->l = photon_loc(P);
            absorbed_by_source = false;
        }

        // Kill the photon if it bounces around too many times...
        if (i > 1000) {
            tau = -1.0;
            printf("!!!!!!! ERROR - Killing photon because it seems to be stuck.\n");
        }
    }
}

/* Propagate a photon through the grid for a scattering simulation. */

void Grid::propagate_photon_scattering(Photon *P) {
    double total_tau_abs = 0.;

    //while (in_grid(P) && P->energy > EPSILON) {
    while (in_grid(P) && total_tau_abs < 30.) {
        // Determin the optical depth that the photon can travel until it's
        // next interaction.
        double tau = -log(1-random_number(random_pool));

        int i = 0;

        // Move the photon to the point of it's next interaction.
        while ((tau > EPSILON) && (in_grid(P))) {
            // Calculate the distance to the next wall.
            double s1 = next_wall_distance(P);

            // Calculate how far the photon can go with the current tau.
            double alpha_abs = 0;
            double alpha_scat = 0;
            for (int idust = 0; idust<nspecies; idust++) {
                //alpha_abs += P->current_kext(idust)* 
                alpha_abs += dust->opacity(P->nu)* 
                    //(1.-P->current_albedo(idust))*
                    (1.-dust->albdo(P->nu))*
                    dens(idust,P->l[0],P->l[1],P->l[2]);
                //alpha_scat += P->current_kext(idust)*P->current_albedo(idust)*
                alpha_scat += dust->opacity(P->nu)*dust->albdo(P->nu)*
                    dens(idust,P->l[0],P->l[1],P->l[2]);
            }

            double s2 = tau/alpha_scat;

            // Determine whether to move to the next wall or to the end of tau.
            double s = s1;
            if (s2 < s) s = s2;

            // Calculate the relevant optical depths.
            double tau_abs = s * alpha_abs;
            double tau_scat = s * alpha_scat;

            // Calculate the average energy over the course of the cell.
            double average_energy = (1.0 - exp(-tau_abs)) / tau_abs *
                P->energy;
            if (tau_abs < EPSILON)
                average_energy = (1.0 - 0.5*tau_abs) * P->energy;

            // Add some of the energy to the scattering array.
            Kokkos::atomic_add(&scatt(0,P->l[0],P->l[1],P->l[2],Q->inu), 
                    average_energy * s / (4*pi * 
                    volume(P->l[0],P->l[1],P->l[2])));

            // Absorb some of the photon's energy.
            P->energy *= exp(-tau_abs);

            // Remvove the tau we've used up with this stepl
            tau -= tau_scat;

            // Add to the total absorbed tau.
            total_tau_abs += tau_abs;

            // Move the photon to it's new position.
            P->move(s);

            // If the photon moved to the next cell, update it's location.
            if (s1 < s2) P->l = photon_loc(P);

            i++;
            // Kill the photon if it bounces around too many times...
            if (i > 1000) {
                tau = -1.0;
                printf("!!!!!!! ERROR - Killing photon because it seems to be stuck.\n");
            }
        }

        // If the photon is still in the grid when it reaches it's 
        // destination...
        if (in_grid(P)) {
            P->event_count += 1;
            // Now scatter the photon.
            scatter(P, 0);
        }
    }
}

/* Get a random direction for the photon. */

void Grid::random_dir_mrw(Photon *P) {
    // Adjust the photon's direction if we are doing MRW.
    double cost = -1+2*random_number(random_pool);
    double sint = sqrt(1-pow(cost,2));
    double phi = 2*pi*random_number(random_pool);

    P->n[0] = sint*cos(phi);
    P->n[1] = sint*sin(phi);
    P->n[2] = cost;
    P->invn[0] = 1.0/P->n[0];
    P->invn[1] = 1.0/P->n[1];
    P->invn[2] = 1.0/P->n[2];
}

/* Propagate a photon using the MRW method for high optical depths. */

void Grid::propagate_photon_mrw(Photon *P) {
    // Calculate the Rosseland mean opacity.
    double alpha = 0.0;
    for (int idust = 0; idust<nspecies; idust++)
        alpha += rosseland_mean_extinction(idust,P->l[0],P->l[1],P->l[2]) * 
                dens(idust,P->l[0],P->l[1],P->l[2]);

    // Calculate the energy threshold at which to stop and re-calculate the
    // Rosseland mean opacity.
    double energy_threshold = 0.;
    for (int idust = 0; idust<nspecies; idust++) {
        // NEEDS TO BE ATOMIC? Because energy could be changing as this is
        // beinc calculated?
        if (temp(idust,P->l[0],P->l[1],P->l[2]) > 3.) {
            for (int ithread = 0; ithread < (int) energy.size(); ithread++)
                energy_threshold += energy(idust,P->l[0],P->l[1],P->l[2]) + 
                    energy_mrw(idust,P->l[0],P->l[1],P->l[2]);
        } else {
            energy_threshold = HUGE_VAL;
            break;
        }
        // TO HERE?
    }
    energy_threshold *= 0.3;

    /* Now calculate how that translates into path traveled. */
    double path_threshold = energy_threshold / (alpha*P->energy);

    // Start walking the photon.

    double path_accumulated = 0.;

    while (true) {
        double dmin = minimum_wall_distance(P);

        if (alpha * dmin > Q->mrw_gamma) {
            if (Q->verbose) {
                printf("Doing MRW step...\n");
            }

            // Calculate the radius of the sphere that we will move the photon 
            // to somewhere on.
            double R_0 = 0.99 * dmin;
            if (Q->verbose) {
                printf("%f %f\n", dmin/au, R_0/au);
            }

            // Calculate the actual distance that the photon traveled through 
            // diffusion.
            double s = 0.5 * R_0 * R_0 * alpha; //Per RADMC-3D, this is the 
                                                //average distance.

            if (Q->verbose) {
                printf("%f\n", s/au);
            }
            // Add the energy absorbed into the grid.
            path_accumulated += s;

            // Move the photon to the edge of the sphere.
            P->move(R_0);

            // Pick a random direction to be traveling in next.
            random_dir_mrw(P);

        } else {
            double tau = -log(1-random_number(random_pool));

            // Calculate the distance to the next wall.
            double s1 = next_wall_distance(P);

            // Calculate how far the photon can go with the current tau.
            double s2 = tau/alpha;

            // Determine whether to move to the next wall or to the end of tau.
            double s = s1;
            if (s2 < s) s = s2;

            // Continuously absorb the photon's energy, if the end result of the
            // current trajectory is absorption.
            path_accumulated += s;

            // Move the photon to it's new position.
            P->move(s);

            // If the photon moved to the next cell, update it's location.
            if (s1 < s2) {
                P->same_cell_count = 0;
                break;
            } else
                // "Absorb" to get a new direction.
                random_dir_mrw(P);
        }

        /* Make sure we update the rad, theta, phi values for P. */
        photon_loc_mrw(P);

        if (path_accumulated > path_threshold)
            break;
    }

    // Add the energy absorbed into the grid.
    for (int idust=0; idust<nspecies; idust++)
        Kokkos::atomic_add(&energy_mrw(idust,P->l[0],P->l[1],P->l[2]),
                P->energy*path_accumulated*
                planck_mean_opacity(idust,P->l[0],P->l[1],P->l[2])*
                dens(idust,P->l[0],P->l[1],P->l[2]));
}

/* Propagate a ray through the grid for raytracing. */

void Grid::propagate_ray(Ray *R) {

    int i=0;
    do {
        double s = next_wall_distance(R);

        if (smallest_wall_size(R) < R->pixel_size)
            R->pixel_too_large = true;

        for (int inu = 0; inu < R->nnu; inu++) {
            double tau_cell = 0;
            double intensity_abs = 0;
            double alpha_ext = 0;
            double alpha_sca = 0;

            if (Q->raytrace_dust) {
                for (int idust=0; idust<nspecies; idust++) {
                    //tau_cell += s*R->current_kext(idust,inu) *
                    tau_cell += s*dust->opacity(R->nu(inu)) *
                            dens(idust,R->l[0],R->l[1],R->l[2]);

                    //alpha_ext += R->current_kext(idust,inu) *
                    alpha_ext += dust->opacity(R->nu(inu)) *
                            dens(idust,R->l[0],R->l[1],R->l[2]);
                    //alpha_sca += R->current_kext(idust,inu) *
                    alpha_sca += dust->opacity(R->nu(inu)) *
                            //R->current_albedo(idust,inu) *
                            dust->albdo(R->nu(inu)) *
                            dens(idust,R->l[0],R->l[1],R->l[2]);

                    //intensity_abs += R->current_kext(idust,inu) * 
                    intensity_abs += dust->opacity(R->nu(inu)) * 
                            //(1 - R->current_albedo(idust,inu)) *
                            (1 - dust->albdo(R->nu(inu))) *
                            dens(idust,R->l[0],R->l[1],R->l[2])
                            * planck_function(R->nu(inu), 
                            temp(idust,R->l[0],R->l[1],R->l[2]));
                }
            }

            double intensity_line = 0;
            if (Q->raytrace_gas) {
                for (int itrans=0; itrans < include_lines.size(); itrans+=2) {
                    int igas = include_lines(itrans);
                    int iline = include_lines(itrans+1);

                    double alpha_this_line = 
                            alpha_line(itrans/2,R->l[0],R->l[1],R->l[2]) *
                            line_profile(igas, iline, itrans/2, R->l[0],R->l[1],R->l[2], 
                            R->nu(inu) * (1 - -R->nframe.dot(
                            vector_velocity(igas, R)) / c_l));

                    tau_cell += s*alpha_this_line;
                    alpha_ext += alpha_this_line;

                    intensity_line += alpha_this_line * 
                            planck_function(R->nu(inu),
                            gas_temp(igas,R->l[0],R->l[1],R->l[2]));
                }
            }

            double albedo = alpha_sca / alpha_ext;

            if (alpha_ext > 0) {
                intensity_abs *= (1.0-exp(-tau_cell)) / alpha_ext;
                intensity_line *= (1.0-exp(-tau_cell)) / alpha_ext;
            }

            double intensity_sca = 0;
            if (Q->raytrace_dust) intensity_sca = (1.0-exp(-tau_cell)) * 
                    albedo * scatt(0,R->l[0],R->l[1],R->l[2],inu);

            double intensity_cell = intensity_abs + intensity_sca + 
                intensity_line;

            if (Q->verbose) {
                printf("%2i  %7.5f  %i  %7.4f  %7.4f\n", i, tau_cell, 
                        R->l[0], R->r[0]/au, s*R->n[0]/au);
                printf("%11.1e  %i  %7.4f  %7.4f\n", R->intensity(inu), R->l[1],
                        R->r[1]/au, s*R->n[1]/au);
                printf("%11.5f  %i  %7.4f  %7.4f\n", R->tau(inu), R->l[2], 
                        R->r[2]/au, s*R->n[2]/au);
            }

            R->intensity(inu) += intensity_cell*exp(-R->tau(inu));
            R->tau(inu) += tau_cell;
        }

        R->move(s);

        R->l = photon_loc(R);

        i++;
    } while (in_grid(R));
}

/* Propagate a ray through the grid for raytracing. */

void Grid::propagate_ray_from_source(Ray *R) {

    int i=0;
    do {
        double s = next_wall_distance(R);

        for (int inu = 0; inu < R->nnu; inu++) {

            double tau_cell = 0;
            for (int idust=0; idust<nspecies; idust++)
                //tau_cell += s*R->current_kext(idust,inu)*
                tau_cell += s*dust->opacity(R->nu(inu))*
                    dens(idust,R->l[0],R->l[1],R->l[2]);

            if (Q->verbose) {
                printf("%2i  %7.5f  %i  %7.4f  %7.4f\n", i, tau_cell, 
                        R->l[0], R->r[0]/au, s*R->n[0]/au);
                printf("%11.1e  %i  %7.4f  %7.4f\n", R->intensity(inu), R->l[1],
                        R->r[1]/au, s*R->n[1]/au);
                printf("%11.5f  %i  %7.4f  %7.4f\n", R->tau(inu), R->l[2], 
                        R->r[2]/au, s*R->n[2]/au);
            }

            R->intensity(inu) *= exp(-tau_cell);
        }

        R->move(s);

        R->l = photon_loc(R);

        i++;
    } while (in_grid(R));
}

/* Calculate the distance between the photon and the nearest wall. */

double Grid::next_wall_distance(Photon *P) {
    return 0.0;
}

/* Calculate the distance between the photon and the outermost wall. */

double Grid::outer_wall_distance(Photon *P) {
    return 0.0;
}

/* Calculate the smallest absolute distance to the nearest wall. */

double Grid::minimum_wall_distance(Photon *P) {
    return 0.0;
}

/* Calculate the smallest distance across the cell. */

double Grid::smallest_wall_size(Photon *P) {
    return 0.0;
}

double Grid::smallest_wall_size(Ray *R) {
    return 0.0;
}

/* Calculate the size of the grid. */

double Grid::grid_size() {
    return 0.0;
}

/* Determine which cell the photon is in. */

Vector<int, 3> Grid::photon_loc(Photon *P) {
    return Vector<int, 3>();
}

/* Update extra position parameters like rad and theta during MRW. */

void Grid::photon_loc_mrw(Photon *P) {
}

/* Randomly generate a photon location within a cell. */
 
Vector<double, 3> Grid::random_location_in_cell(int ix, int iy, int iz) {
    return Vector<double, 3>();
}

/* Check whether a photon is on a wall and going parallel to it. */

bool Grid::on_and_parallel_to_wall(Photon *P) {
    return false;
}

/* Check whether a photon is in the boundaries of the grid. */

bool Grid::in_grid(Photon *P) {
    return true;
}

/* Update the temperature in a cell given the number of photons that have 
 * been absorbed in the cell. */

void Grid::update_grid(Vector<int, 3> l, int cell_index) {
    for (int idust=0; idust<nspecies; idust++) {
        bool not_converged = true;

        /*double total_energy = 0;
        for (int ithread = 0; ithread < (int) energy.size(); ithread++)
            total_energy += energy(idust,cell_index) + 
                energy_mrw(idust,cell_index);*/
        double total_energy = energy(idust,l[0],l[1],l[2]) + energy_mrw(idust,l[0],l[1],l[2]);

        while (not_converged) {
            double T_old = temp(idust,l[0],l[1],l[2]);

            temp(idust,l[0],l[1],l[2])=pow(total_energy/
                //(4*sigma*dust(idust).
                (4*sigma*dust->
                planck_mean_opacity(temp(idust,l[0],l[1],l[2]))*
                mass(idust,l[0],l[1],l[2])),0.25);

            // Make sure that there is a minimum temperature that the grid can
            // get to.
            if (temp(idust,l[0],l[1],l[2]) < 0.1) 
                temp(idust,l[0],l[1],l[2]) = 0.1;

            if ((fabs(T_old-temp(idust,l[0],l[1],l[2]))/T_old < 1.0e-2))
                not_converged = false;
        }

        if (Q->use_mrw) {
            rosseland_mean_extinction(idust,l[0],l[1],l[2]) = 
                    //dust(idust).rosseland_mean_extinction(
                    dust->rosseland_mean_extinction(
                    temp(idust,l[0],l[1],l[2]));
            planck_mean_opacity(idust,l[0],l[1],l[2]) = 
                    //dust(idust).planck_mean_opacity(
                    dust->planck_mean_opacity(
                    temp(idust,l[0],l[1],l[2]));
        }
    }
}

void Grid::update_grid() {
    for (int i=0; i<nw1-1; i++)
        for (int j=0; j<nw2-1; j++)
            for (int k=0; k<nw3-1; k++)
                update_grid(Vector<int, 3>(i,j,k), i*n2*n3 + j*n3 + k);
}

/* Calculate the luminosity of the cell indicated by l. */

/*double Grid::cell_lum(Vector<int, 3> l) {
    return 4*mass[0][l[0]][l[1]][l[2]]*dust[0]->
        planck_mean_opacity(temp[0][l[0]][l[1]][l[2]])*sigma*
        pow(temp[0][l[0]][l[1]][l[2]],4);
}*/

double Grid::cell_lum(int idust, int i, int j, int k) {
    //return 4*mass(idust,i,j,k)*dust(idust).
    return 4*mass(idust,i,j,k)*dust->
        planck_mean_opacity(temp(idust,i,j,k))*sigma*
        pow(temp(idust,i,j,k),4);
}

/*double Grid::cell_lum(Vector<int, 3> l, double nu) {
    return 4*pi*mass[0][l[0]][l[1]][l[2]]*
        dust[0]->opacity(nu)*(1. - dust[0]->albdo(nu))*
        planck_function(nu, temp[0][l[0]][l[1]][l[2]]);
}*/

double Grid::cell_lum(int idust, int i, int j, int k, double nu) {
    return 4*pi*mass(idust,i,j,k)*
        //dust(idust).opacity(nu)*(1. - dust(idust).albdo(nu))*
        dust->opacity(nu)*(1. - dust->albdo(nu))*
        planck_function(nu, temp(idust,i,j,k));
}

/* Calculate the line profile of a spectral line. */

Vector<double, 3> Grid::vector_velocity(int igas, Photon *P) {
    return Vector<double, 3>(velocity(igas,P->l[0],P->l[1],P->l[2],0), 
            velocity(igas,P->l[0],P->l[1],P->l[2],1), 
            velocity(igas,P->l[0],P->l[1],P->l[2],2));
}

double Grid::maximum_velocity(int igas) {
    double vmax = 0;

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < n3; k++) {
                Vector<double, 3> vcell(velocity(igas,i,j,k,0),
                        velocity(igas,i,j,k,1), 
                        velocity(igas,i,j,k,2));
                double vnorm = vcell.norm();

                if (vnorm > vmax) vmax = vnorm;
            }
        }
    }

    return vmax;
}

double Grid::maximum_gas_temperature(int igas) {
    double Tmax = 0;

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < n3; k++) {
                if (gas_temp(igas,i,j,k) > Tmax) Tmax = 
                        gas_temp(igas,i,j,k);
            }
        }
    }

    return Tmax;
}

double Grid::maximum_microturbulence(int igas) {
    double Mmax = 0;

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < n3; k++) {
                if (microturbulence(igas,i,j,k) > Mmax) Mmax = 
                        microturbulence(igas,i,j,k);
            }
        }
    }

    return Mmax;
}

double Grid::line_profile(int igas, int iline, int itrans, int ix, int iy, int iz, 
        double nu) {
    double inv_gammat = inv_gamma_thermal(itrans,ix,iy,iz);

    //double profile = exp(-(nu - gas(igas).nu(iline))*(nu - 
    //        gas(igas).nu(iline)) * (inv_gammat * inv_gammat));
    double profile = exp(-(nu - gas->nu(iline))*(nu - 
            gas->nu(iline)) * (inv_gammat * inv_gammat));

    return profile;
}

void Grid::set_tgas_eq_tdust() {
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < n3; k++) {
                double dust_temperature = 0;
                double dust_density = 0;

                for (int idust = 0; idust < nspecies; idust++) {
                    dust_temperature += dens(idust,i,j,k) * 
                        temp(idust,i,j,k);
                    dust_density += dens(idust,i,j,k);
                }

                if (dust_density > 0)
                    dust_temperature /= dust_density;
                else
                    dust_temperature = 0.1;

                for (int igas = 0; igas < ngases; igas++)
                    gas_temp(igas,i,j,k) = dust_temperature;
            }
        }
    }
}

void Grid::select_lines(py::array_t<double> _lam) {
    // Load the array buffers to get the proper setup info.

    auto _lam_buf = _lam.request();

    // Now get the correct format.
    
    int nlam = _lam_buf.shape[0];
    double *lam = (double *) _lam_buf.ptr; 
    
    // Find the max and min frequencies.

    double max_nu = 0.;
    double min_nu = HUGE_VAL;
    for (int ilam = 0; ilam < nlam; ilam++) {
        double nu = c_l / (lam[ilam]*1.0e-4);

        if (nu > max_nu) max_nu = nu;
        if (nu < min_nu) min_nu = nu;
    }

    // Now, figure out which lines fall in that range.

    for (int igas = 0; igas < ngases; igas++) {
        double max_v = maximum_velocity(igas);
        double a_thermal = 3*sqrt(2 * k_B * maximum_gas_temperature(igas) / 
                //(gas(igas).mu * m_p)); // 3-sigma width
                (gas->mu * m_p)); // 3-sigma width
        double a_microturb = maximum_microturbulence(igas);

        max_v += a_thermal + a_microturb;

        //for (int iline = 0; iline < gas(igas).ntransitions; iline++) {
        //    double max_frequency = gas(igas).nu(iline) * (1. + max_v / c_l);
        //    double min_frequency = gas(igas).nu(iline) * (1. - max_v / c_l);
        for (int iline = 0; iline < gas->ntransitions; iline++) {
            double max_frequency = gas->nu(iline) * (1. + max_v / c_l);
            double min_frequency = gas->nu(iline) * (1. - max_v / c_l);

            if (((min_nu < min_frequency) && (min_frequency < max_nu)) || 
                    ((min_nu < max_frequency) && (max_frequency < max_nu)) ||
                    ((min_frequency < min_nu) && (max_nu < max_frequency))) {
                printf("Including gas %d transition at %f GHz \n", igas, \
                        //gas(igas).nu(iline)/1e9);
                        gas->nu(iline)/1e9);

                Kokkos::resize(include_lines, include_lines.extent(0)+2);
                include_lines(include_lines.extent(0)-2) = igas;
                include_lines(include_lines.extent(0)-1) = iline;

                calculate_level_populations(igas, iline);
            }
        }
    }
}

void Grid::calculate_level_populations(int igas, int iline) {
    size_t i_up = level_populations.extent(0);
    size_t i_low = level_populations.extent(0) + 1;
    Kokkos::resize(level_populations, level_populations.extent(0)+2, n1, n2, n3);
    Kokkos::resize(alpha_line, alpha_line.extent(0)+1, n1, n2, n3);
    Kokkos::resize(inv_gamma_thermal, inv_gamma_thermal.extent(0)+1, n1, n2, n3);

    //int level_up = gas(igas).up(iline)-1;
    //int level_low = gas(igas).low(iline)-1;
    int level_up = gas->up(iline)-1;
    int level_low = gas->low(iline)-1;

    for (int ix = 0; ix < n1; ix++) {
        for (int iy = 0; iy < n2; iy++) {
            for (int iz = 0; iz < n3; iz++) {
                //level_populations(i_up,ix,iy,iz) = gas(igas).weights(level_up) * 
                level_populations(i_up,ix,iy,iz) = gas->weights(level_up) * 
                    //exp(-h_p * c_l * gas(igas).energies(level_up) / (k_B *
                    exp(-h_p * c_l * gas->energies(level_up) / (k_B *
                    gas_temp(igas,ix,iy,iz))) / 
                    //gas(igas).partition_function(gas_temp(igas,ix,iy,iz));
                    gas->partition_function(gas_temp(igas,ix,iy,iz));

                //level_populations(i_low,ix,iy,iz) = gas(igas).weights(level_low) * 
                level_populations(i_low,ix,iy,iz) = gas->weights(level_low) * 
                    //exp(-h_p * c_l * gas(igas).energies(level_low) / (k_B *
                    exp(-h_p * c_l * gas->energies(level_low) / (k_B *
                    gas_temp(igas,ix,iy,iz))) / 
                    //gas(igas).partition_function(gas_temp(igas,ix,iy,iz));
                    gas->partition_function(gas_temp(igas,ix,iy,iz));

                // Also calculate inv_gamma_thermal so we don't have to do it
                // on the fly.

                double a_thermal = sqrt(2 * k_B * gas_temp(igas,ix,iy,iz) / 
                        //(gas(igas).mu * m_p));
                        (gas->mu * m_p));
                double a_microturb = microturbulence(igas,ix,iy,iz);

                //inv_gamma_thermal(inv_gamma_thermal.extent(0), ix, iy, iz) = 1./(gas(igas).nu(iline) / c_l * 
                inv_gamma_thermal(inv_gamma_thermal.extent(0), ix, iy, iz) = 1./(gas->nu(iline) / c_l * 
                        sqrt(a_thermal*a_thermal + a_microturb*a_microturb));

                // Calculate the alpha value for the line in this cell, so this
                // doesn't have to be done on the fly, either.

                alpha_line(alpha_line.extent(0), ix, iy, iz) = c_l*c_l / (8*pi*
                        //gas(igas).nu(iline)*gas(igas).nu(iline)) * 
                        gas->nu(iline)*gas->nu(iline)) * 
                        //gas(igas).A(iline) * 
                        gas->A(iline) * 
                        number_dens(igas,ix,iy,iz) * 
                        (level_populations(i_low,ix,iy,iz)*
                        //gas(igas).weights(level_up) / 
                        gas->weights(level_up) / 
                        //gas(igas).weights(level_low) - 
                        gas->weights(level_low) - 
                        level_populations(i_up,ix,iy,iz)) *
                        inv_gamma_thermal(inv_gamma_thermal.extent(0), ix,iy,iz) / sqrt(pi);
            }
        }
    }
}

void Grid::deselect_lines() {
    Kokkos::resize(level_populations, 0, 0);
    Kokkos::resize(inv_gamma_thermal, 0, 0);
    Kokkos::resize(alpha_line, 0, 0);
    Kokkos::resize(include_lines, 0);
}
