#ifndef TIMING_H
#define TIMING_H

#include <gsl/gsl_matrix.h>

#include "stars.h"
#include "noise.h"

struct toa_output{
    double t;
    double N;
    double T;
    double Delta_E;
    double Delta_R;
    double Delta_S_1PN;
    double Delta_S_DPN;
    double Delta_S_PPN;
    double Delta_FD;
    double Delta_A1;
    double Delta_A2;
    double Delta_L;
    double toa;
};

void toa_generator(struct star_system star_sys,int n,double t[],struct toa_output out[]);

void residual_cal(struct star_system star_sys,int n, double toa[], struct toa_output out[]);

double chi2_white(int n, double toa[], double N[], double sigma_TOA[], struct star_system sys);

gsl_matrix *design_matrix(int n, double toa[], struct star_system sys);

// 这个老的Fisher矩阵计算我们就不再使用与更新了，我们采用design matrix进行计算
// gsl_matrix *Fisher_white(int n, double toa[], double sigma_TOA[], struct star_system sys);

// 应当尽量提供一致的接口，因此
gsl_matrix *Fisher_white_from_design(int n, double toa[], double sigma_TOA[], struct star_system sys);

gsl_matrix *Fisher_to_Cov(gsl_matrix *fisher);

gsl_matrix *residual_covariance(int n, double toa[], double sigma_TOA[], struct red_noise_params rp);

#endif