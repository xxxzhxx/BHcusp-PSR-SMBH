#include "IMBH.h"

#include "stars.h"
#include "const.h"
#include "timing.h"
#include "fitting2.h"

#include <math.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

//会忽略sys里的IMBH部分重新设定
//用之前请确保拟合那个函数没有问题
//要给定IMBH的a和M
//在给定随机个数，随机种子
//这里离心率的随机其实是有点说法的，但是我暂时不知道这个说法，我们先写成在0.1-0.9均匀
double IMBH_detectability(struct star_system sys,double a_IMBH, double m_IMBH,int n_samp,int ran_seed){
    //在并行外生成随机数列
    double e_list[n_samp],omega_list[n_samp],Omega_list[n_samp],inc_list[n_samp],T0_list[n_samp];

    //首先根据a算个Pb
    double Pb_IMBH=sqrt(4.0*pi*pi*a_IMBH*a_IMBH*a_IMBH/G/sys.BH.M);

    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup();
    T=gsl_rng_default;
    r=gsl_rng_alloc(T);
    gsl_rng_set(r,ran_seed);

    for(int i=0;i<n_samp;i++){
        e_list[i]=gsl_rng_uniform(r)*0.8+0.1;
        omega_list[i]=2.0*pi*gsl_rng_uniform(r);
        Omega_list[i]=2.0*pi*gsl_rng_uniform(r);
        inc_list[i]=acos(2.0*gsl_rng_uniform(r)-1.0);
        T0_list[i]=gsl_rng_uniform(r)*Pb_IMBH;
    }

    gsl_rng_free(r);

    int n=1000;//取略多的toa确保点够密集结果比较光滑
    double t[n];
    for(int i=0;i<n;i++){
        t[i]=(i+1.0)/n*5.0*yr;
    }

    //拟合后最大residual记在这
    double max_R[n_samp];
    #pragma omp parallel for schedule(dynamic,1)
    for(int i_ran=0;i_ran<n_samp;i_ran++){
        struct star_system sys_p=sys;
        sys_p.IM.m=m_IMBH;
        sys_p.IM.oe[0]=Pb_IMBH;
        sys_p.IM.oe[1]=e_list[i_ran];
        sys_p.IM.oe[2]=omega_list[i_ran];
        sys_p.IM.oe[3]=Omega_list[i_ran];
        sys_p.IM.oe[4]=inc_list[i_ran];
        sys_p.IM.oe[5]=T0_list[i_ran];

        sys_p.psr.op.IMBH=1;
        struct toa_output out1[n],out2[n];
        toa_generator(sys_p,n,t,out1);
        double toa[n],N[n],sigma_TOA[n];
        for(int i=0;i<n;i++){
            toa[i]=out1[i].toa;
            N[i]=out1[i].N;
            sigma_TOA[i]=1e-3*sec;
        }
        sys_p.psr.op.IMBH=0;
        struct star_system sys_fit=fitting_no_abb_GR(n,toa,N,sigma_TOA,sys_p);//拟合系统//修改程序使其不拟合0-4
        residual_cal(sys_fit,n,toa,out2);//拟合后残差

        double max_res=0;
        for(int i=0;i<n;i++){
            if(max_res<fabs(out2[i].N-out1[i].N)/sys.psr.nu){
                max_res=fabs(out2[i].N-out1[i].N)/sys.psr.nu;
            }
        }
        max_R[i_ran]=max_res;
    }

    double n_det=0;
    for(int i_ran=0;i_ran<n_samp;i_ran++){
        if(max_R[i_ran]>1.0*sec){
            n_det++;
        }
    }
    return n_det/n_samp;
}