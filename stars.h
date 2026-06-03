#ifndef STARS_H
#define STARS_H

#include <stdbool.h>

struct SMBH{
    double M;
    double chi;
    double lambda;
    double eta;
    double q;
    double mu_alpha; //自行，不应该有单位，nmd
    double mu_delta; 
};

struct options{
    bool Newton;
    bool Test_1PN;
    bool Test_S;
    bool Test_Q;
    bool Test_2PN;
    bool Delta_S_2PN;
    bool Delta_FD;
    bool Delta_E_2PN;
    bool proper_motion;
    bool Delta_A1;
    bool Delta_A2;
    bool Delta_L;
    bool IMBH;
    bool star_cluster;
    bool No_hair;//这个打开的话我们期待忽略所有四级矩设定，默认到q=-chi2
    bool My_cluster;
};

struct IMBH{
    double m;
    double oe[6];
};

struct PSR{
    // double m;
    struct options op;
    double oe[6]; /*oe[6]={Pb,e,omega,Omega,inc,T0}*/
    double rv[6];
    double Delta_E_norm;
    double N0;
    double nu;
    double dnu;
    double d2nu;
    double lambda_p; //自旋方向
    double eta_p;
    double d3nu;
    double d4nu;
};

struct star_cluster{
    int n;
    struct IMBH *stars;
};

struct star_system{
    struct SMBH BH;
    struct PSR psr;
    struct IMBH IM;
    //顺序为:（新增参数应该总是夹在最后，这样不影响老代码结构
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu,mu_alpha,mu_delta,lambda_p,\
    eta_p,m_I,Pb_I,e_I,omega_I,Omega_I,i_I,t0_I
    bool fit_op[28];
    struct star_cluster Sstars;
    struct star_cluster My_cluster;
};

/*只是通过oe生成rv，同时参数可选初始化op（全置为0）*/
/*oe[6]={Pb,e,omega,Omega,inc,T0}*/
/*clear_op选1则会将op全置为0*/
/*还需要给定Delta_E_norm，虽然想设计成自选默认或自由给定，但还是硬写成默认好了，减少修改接口*/
void star_PSR_rv_ini(struct PSR *psr, struct SMBH BH, bool clear_op);
void oe_to_rv(double M, double t, double oe[6], double rv[6]);

double star_u2v(double u,double e);

#endif