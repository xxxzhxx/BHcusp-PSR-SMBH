#ifndef NOISE_H
#define NOISE_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include "stars.h"

/*总的噪声就是白噪声和红噪声的和?*/

void white_noise_realization(int n, unsigned long seed, double sigma_TOA[], double delta_TOA[]);

/*生成n_real组由白噪声导致的参数的变动，并且存在一个文件里*/
void white_noise_dxi(unsigned long seed, int n, double toa[], double sigma_TOA[], struct star_system sys, int n_real, char *file_name);

/*先写在这里:红噪声的关联应该用T来计算,而非toa或t*/
/*但是简化起见，我们就假设是在noise里有这么一个component?否则感觉转换起来会很奇怪。。。*/

/*由于存在关联，这个的计算需要额外给出计算的时间点，用toa标记*/
void red_noise_realization(int n, unsigned long seed, double A, double fc, double alpha, double toa[], double delta_TOA[]);

/*那么下面就将是一些能产生结果的程序了，都放在这里因为比较相关*/

struct red_noise_params{
    double A;
    double fc;
    double alpha;
};

/*首先希望能产生一个画图用的程序，生成一组白噪声、红噪声、拟合后残差*/
/*默认输出一组fit2的结果以供分析*/
void noise_gen_wfit(unsigned long seed, int n, double toa[], double sigma_TOA[], struct red_noise_params rp, struct star_system sys,\
char *file_name);

/*产出一些结果用的程序：主要可以用来看白噪声拟合是否存在bias以及rescale的Fisher误差估计是否准确*/
/*对改变红噪声A进行分析*/
void noise_wfit_statistic_A(unsigned long seed, int n, double toa[], double sigma_TOA[], struct red_noise_params rp, \
struct star_system sys, int n_A, double A_range[], int n_real, char *file_name);

//因为希望对比两种方法，所以这个函数两者都算，尽管原理上其中一者我们已经算过一次了
//生成n_real组噪声，分别用白噪声拟合和红噪声拟合给出一些统计，因为是无偏的，这次就不给中值了
void noise_Cfit_statistic_A(unsigned long seed, int n, double toa[], double sigma_TOA[], struct red_noise_params rp, \
struct star_system sys, int n_A, double A_range[], int n_real, char *file_name);


struct noise_params_fit_func_params{
    int n;
    int n_params;
    int k;
    double T_obs;
    gsl_vector *delta_t;
    double sigma_TOA;
    gsl_matrix *F;
    gsl_matrix *FF;
    gsl_matrix *U;
    //我觉得工作空间也应该被反复使用，避免反复分配
    gsl_matrix *phi_inv;
    gsl_matrix *Sigma;
    gsl_matrix *Sigma_inv;
    gsl_matrix *Y;
    gsl_matrix *FS;
    gsl_matrix *FSF;
    gsl_matrix *YU;
    gsl_matrix *UYU;
    gsl_matrix *UYU_inv;
    gsl_matrix *YU_UYU;
    gsl_matrix *YU_UYU_UY;
    gsl_matrix *X;
    gsl_vector *Xt;
};
//k是指定使用傅里叶基的个数
void noise_params_fit_gen(unsigned long seed, int N_r, int n, double toa[], double sigma_TOA[], struct red_noise_params rp, int k, \
struct star_system sys, char *file_name);

#endif