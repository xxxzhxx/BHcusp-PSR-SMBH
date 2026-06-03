#ifndef MOTION_H
#define MOTION_H

#include "stars.h"

struct mot_output{
    double t;
    double rv[6];
    double Delta_E;
    double t0;
};

/*这个函数仅供timing模组构建inverse timing model使用*/
int func_mot_evolve(double t, const double rv[], double f[], void *params);
void mot_evolve(struct star_system star_sys, int n, double t[], struct mot_output out[]);

void mot_evolve_Newton(struct star_system star_sys, int n, double t[], struct mot_output out[]);
void mot_evolve_T1PN(struct star_system star_sys, int n, double t[], struct mot_output out[]);
void mot_evolve_T2PN(struct star_system star_sys, int n, double t[], struct mot_output out[]);

#endif