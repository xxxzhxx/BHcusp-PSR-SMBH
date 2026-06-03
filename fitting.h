#ifndef FITTING_H
#define FITTING_H

#include <gsl/gsl_matrix.h>

#include "stars.h"

struct fitting_white_params{
    int n;
    double *toa;
    double *N;
    double *sigma_TOA;
    struct star_system sys;
};

struct fitting_disconnect_out{
    double M;
    double chi;
    double q;
    double lambda;
    double eta;
    double chi2;
};

/*显然应该让他返回一个含有fitting结果的sys*/
struct star_system fitting_white_nmsimplex2rand(int n, double toa[], double N[], double sigma_TOA[], struct star_system sys);
// struct star_system fitting_white_bfgs2(int n, double toa[], double N[], double sigma_TOA[], struct star_system sys);

/*以下考虑一些使用线性化模型的近似*/
/*这次返回的结果是每个参数的变动，因此不拟合的应该置0*/
/*输入值这次考虑是toa和noise分开，尽管这可能不是实际观测中的操作，但在实际观测中，可以理解为toa是猜测值,noise是pre-fit residual*/
/*考虑到矩阵计算的成本，反复调用这个函数会很愚蠢，因此我们还是把输入和输出设定为矩阵会比较好*/
/*输入输出的矩阵都按列存储数据*/
gsl_matrix *fitting_white_design_matrix(int n, double toa[], int n_real, double noise[n][n_real], double sigma_TOA[], struct star_system sys);

/*理想很美好，但现实是我们并无法存下太多的dt，那太耗费空间了，而且noise里写的随机数种子也不是很好*/
/*我想需要整合一个程序，也许放在noise中*/

struct star_system fitting_MSQ(int n, double toa[], double N[], double sigma_TOA[], struct star_system sys);

struct fitting_disconnect_out fitting_disconnect(int n_peri,int n,double toa[n_peri][n], double N[n_peri][n],double sigma_TOA[n_peri][n], struct star_system sys, double N_fit[n_peri][n]);


#endif