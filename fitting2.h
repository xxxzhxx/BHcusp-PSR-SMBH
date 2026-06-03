#ifndef FITTING2_H
#define FITTING2_H

#include <gsl/gsl_matrix.h>
#include "stars.h"

struct star_system fitting_no_abb_GR(int n,double toa[],double N[],double sigma_TOA[],struct star_system sys);

#endif