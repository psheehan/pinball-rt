#include "spherical_grid.h"

SphericalGrid::SphericalGrid(py::array_t<double> _r, 
            py::array_t<double> _theta, py::array_t<double> _phi) : 
        Grid(_r, _theta, _phi) {

    // Set up the x, y, and z arrays.

    r = py::array_t<double>(n1);
    theta = py::array_t<double>(n2);
    phi = py::array_t<double>(n3);

    // Load the array buffers to get the proper setup info.

    auto r_buf = r.request(); auto theta_buf = theta.request(); 
    auto phi_buf = phi.request();

    // Now get the correct values.

    double *__r = (double *) r_buf.ptr;
    for (int i = 0; i < n1; i++) __r[i] = 0.5 * (w1(i+1) + w1(i));

    double *__theta = (double *) theta_buf.ptr;
    for (int i = 0; i < n2; i++) __theta[i] = 0.5 * (w2(i+1) + w2(i));

    double *__phi = (double *) phi_buf.ptr;
    for (int i = 0; i < n3; i++) __phi[i] = 0.5 * (w3(i+1) + w3(i));
    
    // Also precompute trig functions of the walls.

    Kokkos::resize(sin_w2, nw2);
    Kokkos::resize(cos_w2, nw2);
    Kokkos::resize(neg_mu, nw2);
    Kokkos::resize(tan_w2, nw2);

    Kokkos::resize(sin_tol_w2, nw2);
    Kokkos::resize(cos_tol_w2, nw2);

    for (int iy = 0; iy < nw2; iy++) {
        sin_w2(iy) = sin(w2(iy));
        cos_w2(iy) = cos(w2(iy));
        tan_w2(iy) = tan(w2(iy));

        neg_mu(iy) = -cos_w2(iy);

        sin_tol_w2(iy) = fabs(sin(w2(iy) * (1.0 - EPSILON)) - sin(w2(iy)));
        cos_tol_w2(iy) = fabs(cos(w2(iy) * (1.0 - EPSILON)) - cos(w2(iy)));
    }

    Kokkos::resize(sin_w3, nw3);
    Kokkos::resize(cos_w3, nw3);

    for (int iz = 0; iz < nw3; iz++) {
        sin_w3(iz) = sin(w3(iz));
        cos_w3(iz) = cos(w3(iz));
    }

    // Check for mirror symmetry.

    int volume_scale = 1;
    if (equal_zero(cos_w2(nw2-1), EPSILON))
    {
        mirror_symmetry = true;
        volume_scale = 2;
    }
    else
        mirror_symmetry = false;

    // Set up the volume of each cell.

    for (int i = 0; i < n1; i++)
        for (int j = 0; j < n2; j++)
            for (int k = 0; k < n3; k++)
                volume(i,j,k) = (w1(i+1)*w1(i+1)*w1(i+1) - 
                        w1(i)*w1(i)*w1(i)) * (cos_w2(j) - cos_w2(j+1)) * 
                        (w3(k+1) - w3(k)) / 3 * volume_scale;
}

SphericalGrid::~SphericalGrid() {
}

/* Calculate the distance between the photon and the nearest wall. */

double SphericalGrid::next_wall_distance(Photon *P) {
    //double r = P->r.norm();
    double r = P->rad;

    // Calculate the distance to the intersection with the next radial wall.
    
    double b = P->r*P->n;

    double s = HUGE_VAL;
    for (int i=P->l[0]; i <= P->l[0]+1; i++) {
        if (r == w1(i)) {
            double sr1 = -b + fabs(b);
            if ((sr1 < s) && (sr1 > 0) && (not equal_zero(sr1/
                    (P->rad*(w2(P->l[1]+1)-w2(P->l[1]))),EPSILON))) s = sr1;
            double sr2 = -b - fabs(b);
            if ((sr2 < s) && (sr2 > 0) && (not equal_zero(sr2/
                    (P->rad*(w2(P->l[1]+1)-w2(P->l[1]))),EPSILON))) s = sr2;
        }
        else {
            double c = r*r - w1(i)*w1(i);
            double d = b*b - c;

            if (d >= 0) {
                double sr1 = -b + sqrt(d);
                if ((sr1 < s) && (sr1 > 0)) s = sr1;
                double sr2 = -b - sqrt(d);
                if ((sr2 < s) && (sr2 > 0)) s = sr2;
            }
        }
    }

    // Calculate the distance to the intersection with the next theta wall.
    
    if (nw2 != 2) {
        double theta = P->theta;
        
        for (int i=P->l[1]; i <= P->l[1]+1; i++) {
            if (equal_zero(cos_w2(i),1.0e-10)) {
                double st1 = -P->r[2]*P->invn[2];
                if (equal_zero(st1/(P->rad*(w2(P->l[1]+1)-w2(P->l[1]))),
                        EPSILON)) st1 = 0;
                if ((st1 < s) && (st1 > 0)) s = st1;
            }
            else {
                double a = P->n[0]*P->n[0]+P->n[1]*P->n[1]-P->n[2]*P->n[2]*
                    tan_w2(i)*tan_w2(i);
                double b = 2*(P->r[0]*P->n[0]+P->r[1]*P->n[1]-P->r[2]*P->n[2]*
                    tan_w2(i)*tan_w2(i));

                //if (theta == w2(i)) {
                if (equal(P->sin_theta,sin_w2(i),sin_tol_w2(i))) {
                    double st1 = (-b + fabs(b))/(2*a);
                    if ((st1 < s) && (st1 > 0)) s = st1;
                    double st2 = (-b - fabs(b))/(2*a);
                    if ((st2 < s) && (st2 > 0)) s = st2;
                }
                else {
                    double c = P->r[0]*P->r[0]+P->r[1]*P->r[1]-P->r[2]*P->r[2]*
                        tan_w2(i)*tan_w2(i);
                    double d = b*b-4*a*c;

                    if (d >= 0) {
                        double st1 = (-b + sqrt(d))/(2*a);
                        if ((st1 < s) && (st1 > 0)) s = st1;
                        double st2 = (-b - sqrt(d))/(2*a);
                        if ((st2 < s) && (st2 > 0)) s = st2;
                    }
                }
            }
        }
    }

    // Calculate the distance to intersection with the nearest phi wall.
    
    if (nw3 != 2) {
        double phi = P->phi;
        
        for (int i=P->l[2]; i <= P->l[2]+1; i++) {
            if (phi != w3(i)) {
                double c = P->r[0]*sin_w3(i)-P->r[1]*cos_w3(i);
                double d = P->n[0]*sin_w3(i)-P->n[1]*cos_w3(i);

                double sp = -c/d;

                if ((sp < s) && (sp > 0)) s = sp;
            }
        }
    }

    return s;
}

/* Calculate the distance between the photon and the outermost wall. */

double SphericalGrid::outer_wall_distance(Photon *P) {

    double r = P->r.norm();

    double s = HUGE_VAL;

    double b = P->r*P->n;
    double c = r*r - w1(nw1-1)*w1(nw1-1);
    double d = b*b - c;

    if (d >= 0) {
        double sr1 = -b + sqrt(d);
        if ((sr1 < s) && (sr1 > 0)) s = sr1;
        double sr2 = -b - sqrt(d);
        if ((sr2 < s) && (sr2 > 0)) s = sr2;
    }

    return s;
}

/* Calculate the smallest absolute distance to the nearest wall. */

double SphericalGrid::minimum_wall_distance(Photon *P) {
    // Calculate the distance to the nearest radial wall.
    
    double s = HUGE_VAL;
    for (int i=P->l[0]; i <= P->l[0]+1; i++) {
        double sr = fabs(P->rad - w1(i));
        if (sr < s) s = sr;
    }

    // Calculate the distance to the nearest theta wall.
    
    if (nw2 != 2) {
        for (int i=P->l[1]; i <= P->l[1]+1; i++) {
            Vector<double, 3> r_hat(sin_w2(i)*P->cos_phi, 
                    sin_w2(i)*P->sin_phi, cos_w2(i));

            double rho = P->r.dot(r_hat);

            double st = (rho*r_hat - P->r).norm();
            if (st < s) s = st;
        }
    }

    // Calculate the distance to the nearest phi wall.
    
    if (nw3 != 2) {
        for (int i=P->l[2]; i <= P->l[2]+1; i++) {
            Vector<double, 3> r_hat = Vector<double, 3>(cos_w3(i),
                    sin_w3(i), 0);
            Vector<double, 3> z_hat = Vector<double, 3>(0.,0.,1.);

            double rho = P->r.dot(r_hat);

            double sp = (rho*r_hat+P->r[2]*z_hat - P->r).norm();
            if (sp < s) s = sp;
        }
    }
    
    return s;
}

/* Calculate the smallest distance across the cell. */

double SphericalGrid::smallest_wall_size(Photon *P) {

    double s = fabs(w1(P->l[0]+1) - w1(P->l[0]));

    if (nw2 != 2) {
        double r = w1(P->l[0]);
        if (w1(P->l[0]) == 0)
            r = w1(P->l[0]+1)*0.5;

        double st = fabs(r*(w2(P->l[1]+1) - w2(P->l[1])));
        if (st < s) s = st;
    }
    
    if (nw3 != 2) {
        double r = w1(P->l[0]);
        if (w1(P->l[0]) == 0)
            r = w1(P->l[0]+1)*0.5;

        double sint = fmin(sin_w2(P->l[1]), sin_w2(P->l[1]+1));
        if (equal_zero(sint, EPSILON))
            sint = sin(0.5*(w2(P->l[1]) + w2(P->l[1]+1)));

        double sp = fabs(r * sint * (w3(P->l[2]+1) - w3(P->l[2])));
        if (sp < s) s = sp;
    }
    
    return s;
}

double SphericalGrid::smallest_wall_size(Ray *R) {

    // Use the cell volume as an estimator of the average size of a cell.

    double cell_volume = volume(R->l[0],R->l[1],R->l[2]);

    // Scale by the size in the theta, if theta width > 0.3
    
    double theta_scale = fmin(0.3 / (w2(R->l[1]+1) - w2(R->l[1])), 1.);
    double phi_scale = fmin(0.3 / (w3(R->l[2]+1) - w3(R->l[2])), 1.);

    double s = pow(cell_volume*theta_scale*phi_scale, 1./3);

    return s;
}

/* Calculate the size of the grid. */

double SphericalGrid::grid_size() {
    return 2*w1(nw1-1);
}

/* Determine which cell the photon is in. */

Vector<int, 3> SphericalGrid::photon_loc(Photon *P) {
    Vector<int, 3> l;

    double pi = 3.14159265;
    P->rad = P->r.norm();
    /* If P->rad = 0 we need to be careful about how we calculate theta and 
     * phi. */
    if (P->rad == 0) {
        //P->theta = pi - P->theta;
        P->cos_theta *= -1;
        P->l[1] = -1;
        if (nw3 != 2) P->phi = fmod(P->phi + pi, 2*pi);
        P->l[2] = -1;
    }
    else {
        double R = sqrt(P->r[0]*P->r[0] + P->r[1]*P->r[1]);
        //P->theta = acos(P->r[2]/P->rad);
        if (nw3 != 2) P->phi = fmod(atan2(P->r[1],P->r[0])+2*pi,2*pi);

        P->cos_theta = P->r[2] / P->rad;
        P->sin_theta = R / P->rad;
        if (R == 0) {
            P->cos_phi = 1.0;
            P->sin_phi = 0.0;
        } 
        else {
            P->cos_phi = P->r[0] / R;
            P->sin_phi = P->r[1] / R;
        }
    }
    double r = P->rad;
    //double theta = P->theta;
    double phi = P->phi;

    double cos_theta = P->cos_theta;
    double sin_theta = P->sin_theta;
    double cos_phi = P->cos_phi;
    double sin_phi = P->sin_phi;

    // Check if we are using mirror symmetry and we're in the southern
    // hemisphere. If so, we need to flip.
    
    if (mirror_symmetry) {
        if (cos_theta < 0) {
            //theta = pi - theta;
            P->n[2] *= -1;
            cos_theta *= -1;
        }

        if (equal_zero(cos_theta, EPSILON) and P->n[2] < 0)
            P->n[2] *= -1;
    }

    // Find the location in the radial grid.
    
    double gnx = sin_theta*cos_phi;
    double gny = sin_theta*sin_phi;
    double gnz = cos_theta;
    if (equal_zero(gnx, EPSILON)) gnx = 0.;
    if (equal_zero(gny, EPSILON)) gny = 0.;
    if (equal_zero(gnz, EPSILON)) gnz = 0.;
    
    if (r >= w1(nw1-1))
        l[0] = n1-1;
    else if (r <= w1(0))
        l[0] = 0;
    else {
        if (P->l[0] == -1)
            l[0] = find_in_arr(r,w1,nw1);
        else {
            int lower = P->l[0]-1;
            if (lower < 0) lower = 0;
            int upper = P->l[0]+1;
            if (upper > n1-1) upper = n1;
            
            l[0] = find_in_arr(r,w1,lower,upper);
        }
    }
    
    /* Because of floating point errors it may be the case that the photon 
     * should be on the wall exactly, but is not exactly on the wall. We
     * need to put the photon exactly on the wall. */

    if (equal(r,w1(l[0]),EPSILON))
        r = w1(l[0]);
    else if (equal(r,w1(l[0]+1),EPSILON))
        r = w1(l[0]+1);

    /* Finally, update which cell the photon is in based on the direction it
     * is going. */
    if ((r == w1(l[0])) && (P->n[0]*gnx+P->n[1]*gny+P->n[2]*gnz < 0))
        l[0] -= 1;
    else if ((r == w1(l[0]+1)) && (P->n[0]*gnx+P->n[1]*gny+P->n[2]*gnz >= 0))
        l[0] += 1;

    // Find the location in the theta grid.
    
    if (nw2 == 2)
        l[1] = 0;
    else {
        if (-cos_theta >= neg_mu(nw2-1))
            l[1] = n2-1;
        else if (-cos_theta <= neg_mu(0))
            l[1] = 0;
        else {
            if (P->l[1] == -1)
                l[1] = find_in_arr(-cos_theta,neg_mu,nw2);
            else {
                int lower = P->l[1]-1;
                if (lower < 0) lower = 0;
                int upper = P->l[1]+1;
                if (upper > n2-1) upper = n2-1;
                
                l[1] = find_in_arr(-cos_theta,neg_mu,lower,upper);
            }
            if (l[1] == n2) l[1] = n2-1;
        }

        /* Because of floating point errors it may be the case that the photon 
         * should be on the wall exactly, but is not exactly on the wall. We
         * need to put the photon exactly on the wall. */

        if (equal(cos_theta,cos_w2(l[1]),cos_tol_w2(l[1]))) {
            //theta = w2(l[1]);
            cos_theta = cos_w2(l[1]);
            sin_theta = sin_w2(l[1]);
        }
        else if (equal(cos_theta,cos_w2(l[1]+1),cos_tol_w2(l[1]+1))) {
            //theta = w2(l[1]+1);
            cos_theta = cos_w2(l[1]+1);
            sin_theta = sin_w2(l[1]+1);
        }

        /* Update which cell the photon is in based on the direction it
         * is going. */

        double gnx = cos_theta*cos_phi;
        double gny = cos_theta*sin_phi;
        double gnz = -sin_theta;
        if (equal_zero(gnx, EPSILON)) gnx = 0.;
        if (equal_zero(gny, EPSILON)) gny = 0.;
        if (equal_zero(gnz, EPSILON)) gnz = 0.;
        
        if ((cos_theta == cos_w2(l[1])) && (P->n[0]*gnx+P->n[1]*gny+P->n[2]*gnz < 0))
            l[1] -= 1;
        else if ((cos_theta == cos_w2(l[1]+1)) && (P->n[0]*gnx+P->n[1]*gny+P->n[2]*gnz >= 0))
            l[1] += 1;

        /* Finally, if you somehow end up with l[1] = -1 or l[1] = n2, change
         * those to the correct values because you can't escape the grid in the
         * theta direction. */

        if (l[1] == -1) l[1] = 0;
        if (l[1] == n2) l[1] = n2-1;
    }

    // Find the location in the phi grid.
    
    if (nw3 == 2)
        l[2] = 0;
    else {
        if (P->l[2] == -1)
            l[2] = find_in_arr(phi,w3,nw3);
        else
            l[2] = find_in_periodic_arr(phi,w3,n3,P->l[2]-1,P->l[2]+1);

        if (l[2] == -1) find_in_arr(phi,w3,nw3);

        /* Check whether the photon is supposed to be exactly on the cell
         * wall. Floating point errors may keep it from being exactly on the
         * wall, and we need to fix that. */

        if (equal(phi,w3(l[2]),EPSILON)) {
            phi = w3(l[2]);
            sin_phi = sin_w3(l[2]);
            cos_phi = cos_w3(l[2]);
        }
        else if (equal(phi,w3(l[2]+1),EPSILON)) {
            phi = w3(l[2]+1);
            sin_phi = sin_w3(l[2]+1);
            cos_phi = cos_w3(l[2]+1);
        }

        /* Update which cell the photon is in depending on the 
         * direction it is going. */

        double gnx = -sin_phi;
        double gny = cos_phi;
        double gnz = 0.0;
        if (equal_zero(gnx, EPSILON)) gnx = 0.;
        if (equal_zero(gny, EPSILON)) gny = 0.;
        if (equal_zero(gnz, EPSILON)) gnz = 0.;
        
        if ((phi == w3(l[2])) && (P->n[0]*gnx+P->n[1]*gny <= 0))
            l[2] -= 1;
        else if ((phi == w3(l[2]+1)) && (P->n[0]*gnx+P->n[1]*gny >= 0))
            l[2] += 1;
        l[2] = (l[2]+n3)%(n3);

        /* Finally, if you are at phi = 0, but going towards negative phi, 
         * you should set phi = 2*pi. */

        if ((phi == 0) && (l[2] == n3-1))
            phi = w3(l[2]+1);
    }

    /* Since we may have updated r, theta and phi to be exactly on the grid 
     * cell walls, change the photon position slightly to reflect this. */

    P->r[0] = r * sin_theta * cos_phi;
    P->r[1] = r * sin_theta * sin_phi;
    P->r[2] = r * cos_theta;
    P->rad = r;
    //P->theta = theta;
    P->phi = phi;

    P->sin_theta = sin_theta;
    P->cos_theta = cos_theta;
    P->sin_phi = sin_phi;
    P->cos_phi = cos_phi;
    
    /* Also calculate n in the coordinate system frame. */

    Vector<double, 3> xhat(sin_theta*cos_phi, cos_theta*cos_phi, -sin_phi);
    Vector<double, 3> yhat(sin_theta*sin_phi, cos_theta*sin_phi, cos_phi);
    Vector<double, 3> zhat(cos_theta, -sin_theta, 0.);

    P->nframe = P->n[0]*xhat + P->n[1]*yhat + P->n[2]*zhat;

    P->cell_index = l[0]*n2*n3 + l[1]*n3 + l[2];

    return l;
}

/* Update extra position parameters like rad and theta during MRW. */

void SphericalGrid::photon_loc_mrw(Photon *P) {
    /* Calculate the radial location of the photon. */
    P->rad = P->r.norm();

    /* If P->rad = 0 we need to be careful about how we calculate theta and 
     * phi. */
    if (P->rad == 0) {
        //P->theta = pi - P->theta;
        P->cos_theta *= -1;
        P->l[1] = -1;
        if (nw3 != 2) P->phi = fmod(P->phi + pi, 2*pi);
        P->l[2] = -1;
    }
    else {
        double R = sqrt(P->r[0]*P->r[0] + P->r[1]*P->r[1]);
        //P->theta = acos(P->r[2]/P->rad);
        if (nw3 != 2) P->phi = fmod(atan2(P->r[1],P->r[0])+2*pi,2*pi);

        P->cos_theta = P->r[2] / P->rad;
        P->sin_theta = R / P->rad;
        P->cos_phi = P->r[0] / R;
        P->sin_phi = P->r[1] / R;
    }
}

/* Randomly generate a photon location within a cell. */
 
Vector<double, 3> SphericalGrid::random_location_in_cell(int ix, int iy, 
        int iz) {
    double r = w1(ix) + random_number(random_pool) * (w1(ix+1) - w1(ix));
    double theta = w2(iy) + random_number(random_pool) * (w2(iy+1) - w2(iy));
    double phi = w3(iz) + random_number(random_pool) * (w3(iz+1) - w3(iz));

    double x = r * sin(theta) * cos(phi);
    double y = r * sin(theta) * sin(phi);
    double z = r * cos(theta);

    return Vector<double, 3>(x, y, z);
}

/* Check whether a photon is in the boundaries of the grid. */

bool SphericalGrid::in_grid(Photon *P) {

    /*double r = P->r.norm();

    if ((r >= w1[nw1-1]) || (equal(r,w1[nw1-1],EPSILON)))
        return false;
    else if ((r <= w1[0]) || (equal(r,w1[0],EPSILON)))
        return false; */
    if ((P->l[0] >= n1) || (P->l[0] < 0))
        return false;
    else
        return true;
}
