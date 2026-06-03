#ifndef TIMING_ND_H
#define TIMING_ND_H

#include "timing.h"
#include "stars.h"

struct ND_params{
    struct star_system sys;
    double toa;
    int op;
};

gsl_matrix *design_matrix_ND(int n, double toa[], struct star_system sys);
gsl_matrix *Fisher_white_from_design_ND(int n, double toa[], double sigma_TOA[], struct star_system sys);


#endif