#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "const.h"
#include "stars.h"
#include "motion.h"
#include "timing.h"
#include "noise.h"
#include "fitting.h"
#include "timing_ND.h"
#include "fitting2.h"
#include "IMBH.h"

int main(int argc, char *argv[]){
    struct SMBH SgrA;
    SgrA.M=4.3e6*M_sun;
    SgrA.chi=0.6;
    SgrA.lambda=pi/6.0;
    SgrA.eta=5.0*pi/9.0;
    SgrA.q=-0.36;
    SgrA.mu_alpha=-3.2*mas/yr;
    SgrA.mu_delta=-5.6*mas/yr;

    struct PSR psr;
    double oe[]={0.5*yr,0.8,pi*5.0/7.0,0.0,pi/5.0,0.5*yr/3.0};
    for(int i=0;i<6;i++){
        psr.oe[i]=oe[i];
    }
    star_PSR_rv_ini(&psr,SgrA,1);

    psr.op.Newton=1;
    psr.op.Test_1PN=0;
    psr.op.Test_2PN=0;
    psr.op.Test_S=1;
    psr.op.Test_Q=0;
    psr.op.No_hair=0;//这个会忽略自旋输入
    psr.op.Delta_S_2PN=0;
    psr.op.Delta_FD=0;
    psr.op.Delta_E_2PN=0;
    psr.op.proper_motion=0;
    psr.op.Delta_A1=0;
    psr.op.Delta_A2=0;
    psr.op.Delta_L=0;
    psr.op.IMBH=0;
    psr.op.star_cluster=0;
    psr.op.My_cluster=0;


    psr.N0=0.5;
    psr.nu=1.0/sec;
    psr.dnu=1e-15/sec/sec;
    psr.d2nu=0.0;
    psr.lambda_p=pi/5.0;
    psr.eta_p=0.0;
    psr.d3nu=0.0;
    psr.d4nu=0.0;

    double total_t=0;
    clock_t start_t,finish_t;

    start_t=clock();

    // 设置好系统
    struct star_system sys;
    sys.BH=SgrA;
    sys.psr=psr;
    sys.IM.m=1000.0*M_sun;
    sys.IM.oe[0]=50*yr;
    sys.IM.oe[1]=0.3;
    sys.IM.oe[2]=1.0;
    sys.IM.oe[3]=2.0;
    sys.IM.oe[4]=0.5;
    sys.IM.oe[5]=0.0; //t0

    for(int i=0;i<28;i++){
        sys.fit_op[i]=0;
    }
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu,mu_alpha,mu_delta,lambda_p,eta_p, d3nu, d4nu, m_I
    //0,1  ,2,3     ,4  ,5 ,6,7    ,8    ,9  ,10,11,12,13 ,14  ,15      ,16      ,17      ,18    ,19   ,20   ,21
    //Pb_I,e_I,omega_I,Omega_I,i_I,t0_I
    //22  ,23 ,24.    ,25.    ,26 ,27
    sys.fit_op[0]=1;
    sys.fit_op[5]=1;
    sys.fit_op[6]=1;
    sys.fit_op[7]=1;
    sys.fit_op[9]=1;
    sys.fit_op[10]=1;
    sys.fit_op[11]=1;
    sys.fit_op[12]=1;
    sys.fit_op[13]=1;


    int n=1000;
    double t[n],toa[n],sigma_toa[n];
    for(int i=0;i<n;i++){
        t[i]=5.0*yr*i/(n-1.0);
        toa[i]=t[i];
        sigma_toa[i]=1e-3*sec;
    }

    // gsl_matrix *F=Fisher_white_from_design(n,toa,sigma_toa,sys);
    // gsl_matrix *Cov=Fisher_to_Cov(F);
    // printf("dM/M = %.16g de/e = %.16g\n",sqrt(gsl_matrix_get(Cov,0,0))/sys.BH.M,sqrt(gsl_matrix_get(Cov,2,2)));

    struct mot_output out[n];
    mot_evolve(sys,n,t,out);
    FILE *fp=fopen("/Users/yuntian/Documents/VScode/Timing-3body/data/old_orbit.txt","wt");
    for(int i=0;i<n;i++){
        fprintf(fp,"%.16g %.16g %.16g\n",out[i].rv[0],out[i].rv[1],out[i].rv[2]);
    }

    fclose(fp);

    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu,mu_alpha,mu_delta,lambda_p,eta_p, d3nu, d4nu, m_I
    //0,1  ,2,3     ,4  ,5 ,6,7    ,8    ,9  ,10,11,12,13 ,14  ,15      ,16      ,17      ,18    ,19   ,20   ,21
    //Pb_I,e_I,omega_I,Omega_I,i_I,t0_I
    //22  ,23 ,24.    ,25.    ,26 ,27
    // sys.fit_op[2]=0;//在打开no-hair的时候显然不能估计q//但话说回来，我们这里不看Fisher，Fisher全看新程序的
    // sys.fit_op[8]=0;
    // sys.fit_op[14]=0;
    // sys.fit_op[15]=0;
    // sys.fit_op[16]=0;
    // sys.fit_op[17]=0;
    // sys.fit_op[18]=0;
    // sys.fit_op[19]=0;
    // sys.fit_op[20]=0;
    // sys.fit_op[21]=0;
    // sys.fit_op[22]=0;
    // sys.fit_op[23]=0;
    // sys.fit_op[24]=0;
    // sys.fit_op[25]=0;
    // sys.fit_op[26]=0;
    // sys.fit_op[27]=0;

    // FILE *fp=fopen("data2026/FD_residual.txt","wt");
    // int n_p=10000;
    // double t_p[n_p];
    // for(int i=0;i<n_p;i++){
    //     t_p[i]=(i+1.0)/n_p*5.0*yr;
    // }
    // struct mot_output mot_p[n_p];
    // mot_evolve(sys,n_p,t_p,mot_p);
    // double r_p[n_p];
    // for(int i=0;i<n_p;i++){
    //     r_p[i]=sqrt(mot_p[i].rv[0]*mot_p[i].rv[0]+mot_p[i].rv[1]*mot_p[i].rv[1]+mot_p[i].rv[2]*mot_p[i].rv[2]);
    // }

    // double t_peri[10];
    // int i_p=0;
    // for(int i=1;i<n_p-1;i++){
    //     if(r_p[i]<=r_p[i-1] && r_p[i]<=r_p[i+1]){
    //         t_peri[i_p]=t_p[i];
    //         // printf("%.16g\n",t_peri[i_p]);
    //         i_p++;
    //     }
    // }

    // int n=1000;
    // double t[n];
    // for(int i=0;i<n;i++){
    //     t[i]=t_peri[0]+(-9.0+18.0*i/(n-1.0))/365.25*yr;
    // }
    // struct toa_output out1[n],out2[n],out3[n],out4[n];
    // toa_generator(sys,n,t,out1);
    // double toa[n],N[n],sigma_TOA[n];
    // for(int i=0;i<n;i++){
    //     toa[i]=out1[i].toa;
    //     N[i]=out1[i].N;
    //     sigma_TOA[i]=1e-3*sec;
    // }
    // residual_cal(sys,n,toa,out2);
    // sys.psr.op.Delta_FD=0;//关闭扰动
    // // sys.psr.op.Test_S=0;
    // // sys.psr.op.Test_Q=0;
    // residual_cal(sys,n,toa,out3);//未拟合的原始残差
    // struct star_system sys_fit=fitting_no_abb_GR(n,toa,N,sigma_TOA,sys);//拟合系统包括黑洞，不包括q
    // residual_cal(sys_fit,n,toa,out4);//拟合后残差

    // for(int i=0;i<n;i++){
    //     fprintf(fp,"%.16g %.16g %.16g %.16g %.16g\n",(t[i]-t_peri[0])*365.25/yr,out1[i].N,out2[i].N,out3[i].N,out4[i].N);
    // }
    // fclose(fp);


    // int ran_seed=42;//这个需要改成每次不一样的之后
    // const gsl_rng_type *T;
    // gsl_rng *r;
    // gsl_rng_env_setup();
    // T=gsl_rng_default;
    // r=gsl_rng_alloc(T);
    // gsl_rng_set(r,ran_seed);


    // double r0_S2=9.4*mpc;
    // int N_rand=1000;
    // int n_cluster=100;
    // double m_star=10.0*M_sun;
    // double r_max=10.0*r0_S2;
    // double e_max=0.9;
    // FILE *fp=fopen("data2026/one_peri_10M_100.txt","wt");

    // sys.My_cluster.n=n_cluster;
    // sys.My_cluster.stars=malloc(n_cluster*sizeof(struct IMBH));
    // for(int i=0;i<n_cluster;i++){
    //     sys.My_cluster.stars[i].m=m_star;
    //     double a_rand=gsl_rng_uniform(r)*r_max;
    //     sys.My_cluster.stars[i].oe[0]=sqrt(a_rand*a_rand*a_rand*4.0*pi*pi/G/sys.BH.M);
    //     sys.My_cluster.stars[i].oe[1]=sqrt(gsl_rng_uniform(r))*e_max;
    //     sys.My_cluster.stars[i].oe[2]=2.0*pi*gsl_rng_uniform(r);
    //     sys.My_cluster.stars[i].oe[3]=2.0*pi*gsl_rng_uniform(r);
    //     sys.My_cluster.stars[i].oe[4]=acos(2.0*gsl_rng_uniform(r)-1.0);
    //     sys.My_cluster.stars[i].oe[5]=gsl_rng_uniform(r)*sqrt(a_rand*a_rand*a_rand*4.0*pi*pi/G/sys.BH.M);
    // }

    // int n_p=10000;
    // double t_p[n_p];
    // for(int i=0;i<n_p;i++){
    //     t_p[i]=(i+1.0)/n_p*5.0*yr;
    // }
    // sys.psr.op.My_cluster=1;
    // struct mot_output mot_p[n_p];
    // mot_evolve(sys,n_p,t_p,mot_p);
    // double r_p[n_p];
    // for(int i=0;i<n_p;i++){
    //     r_p[i]=sqrt(mot_p[i].rv[0]*mot_p[i].rv[0]+mot_p[i].rv[1]*mot_p[i].rv[1]+mot_p[i].rv[2]*mot_p[i].rv[2]);
    // }

    // double t_peri[10];
    // int i_p=0;
    // for(int i=1;i<n_p-1;i++){
    //     if(r_p[i]<=r_p[i-1] && r_p[i]<=r_p[i+1]){
    //         t_peri[i_p]=t_p[i];
    //         // printf("%.16g\n",t_peri[i_p]);
    //         i_p++;
    //     }
    // }

    // int n=10*18*3;
    // double t[n];
    // for(int i=0;i<10;i++){
    //     for(int j=0;j<18*3;j++){
    //         t[18*3*i+j]=t_peri[i]+(-9.0+j/3.0)/365.25*yr;
    //     }
    // }
    // struct toa_output out1[n],out2[n],out3[n],out4[n];
    // toa_generator(sys,n,t,out1);
    // double toa[n],N[n],sigma_TOA[n];
    // for(int i=0;i<n;i++){
    //     toa[i]=out1[i].toa;
    //     N[i]=out1[i].N;
    //     sigma_TOA[i]=1e-3*sec;
    // }
    // residual_cal(sys,n,toa,out2);
    // sys.psr.op.My_cluster=0;//关闭扰动
    // residual_cal(sys,n,toa,out3);//未拟合的原始残差
    // struct star_system sys_fit=fitting_no_abb_GR(n,toa,N,sigma_TOA,sys);//拟合系统包括黑洞，包括q
    // residual_cal(sys_fit,n,toa,out4);//拟合后残差

    // for(int i=0;i<n;i++){
    //     fprintf(fp,"%.16g %.16g %.16g %.16g %.16g\n",t[i],out1[i].N,out2[i].N,out3[i].N,out4[i].N);
    // }
    // fclose(fp);

    // for(int i=0;i<1000;i++){
    //     fprintf(fp,"%.16g ",(double)(-9.0+18.0*(i)/(1000.0-1.0)/365.25*yr));
    // }
    // // fprintf(fp,"\n");
    // // fflush(fp);
    // // //来设计任务并行
    // // //内存应该够用所以我们直接生成完在parallel for
    // struct star_cluster All_Cluster[N_rand];
    // for(int i_rand=0;i_rand<N_rand;i_rand++){
    //     All_Cluster[i_rand].n=n_cluster;
    //     All_Cluster[i_rand].stars=malloc(n_cluster*sizeof(struct IMBH));
    //     if(All_Cluster[i_rand].stars==NULL){
    //         printf("malloc error\n");
    //         fflush(stdout);
    //     }
    //     for(int i=0;i<n_cluster;i++){
    //         All_Cluster[i_rand].stars[i].m=m_star;
    //         double a_rand=gsl_rng_uniform(r)*r_max;
    //         All_Cluster[i_rand].stars[i].oe[0]=sqrt(a_rand*a_rand*a_rand*4.0*pi*pi/G/sys.BH.M);
    //         All_Cluster[i_rand].stars[i].oe[1]=sqrt(gsl_rng_uniform(r))*e_max;
    //         All_Cluster[i_rand].stars[i].oe[2]=2.0*pi*gsl_rng_uniform(r);
    //         All_Cluster[i_rand].stars[i].oe[3]=2.0*pi*gsl_rng_uniform(r);
    //         All_Cluster[i_rand].stars[i].oe[4]=acos(2.0*gsl_rng_uniform(r)-1.0);
    //         All_Cluster[i_rand].stars[i].oe[5]=gsl_rng_uniform(r)*sqrt(a_rand*a_rand*a_rand*4.0*pi*pi/G/sys.BH.M);
    //     }
    // }
    // #pragma omp parallel for schedule(dynamic,1)
    // for(int i_rand=0;i_rand<N_rand;i_rand++){
    //     struct star_system sys_pri=sys;
    //     sys_pri.My_cluster=All_Cluster[i_rand];
    //     printf("thread %d\n", omp_get_thread_num());
    //     fflush(stdout);

    //     int n_p=10000;
    //     double t_p[n_p];
    //     for(int i=0;i<n_p;i++){
    //         t_p[i]=(i+1.0)/n_p*5.0*yr;
    //     }
    //     sys.psr.op.My_cluster=1;
    //     struct mot_output mot_p[n_p];
    //     mot_evolve(sys,n_p,t_p,mot_p);
    //     double r_p[n_p];
    //     for(int i=0;i<n_p;i++){
    //         r_p[i]=sqrt(mot_p[i].rv[0]*mot_p[i].rv[0]+mot_p[i].rv[1]*mot_p[i].rv[1]+mot_p[i].rv[2]*mot_p[i].rv[2]);
    //     }

    //     double t_peri[10];
    //     int i_p=0;
    //     for(int i=1;i<n_p-1;i++){
    //         if(r_p[i]<=r_p[i-1] && r_p[i]<=r_p[i+1]){
    //             t_peri[i_p]=t_p[i];
    //             // printf("%.16g\n",t_peri[i_p]);
    //             i_p++;
    //         }
    //     }

    //     int n=1000;
    //     double t[n];
    //     for(int i=0;i<n;i++){
    //         t[i]=t_peri[0]+(-9.0+18.0*i/(n-1.0))/365.25*yr;
    //     }
    //     sys_pri.psr.op.My_cluster=1;
    //     struct toa_output out1[n],out2[n],out3[n],out4[n];
    //     toa_generator(sys_pri,n,t,out1);
    //     double toa[n],N[n],sigma_TOA[n];
    //     for(int i=0;i<n;i++){
    //         toa[i]=out1[i].toa;
    //         N[i]=out1[i].N;
    //         sigma_TOA[i]=1e-3*sec;
    //     }

    //     // residual_cal(sys_pri,n,toa,out2);//检查用
    //     sys_pri.psr.op.My_cluster=0;//关闭扰动
    //     // residual_cal(sys_pri,n,toa,out3);//未拟合的原始残差
    //     struct star_system sys_fit=fitting_no_abb_GR(n,toa,N,sigma_TOA,sys_pri);//拟合系统//修改程序使其不拟合0-4
    //     residual_cal(sys_fit,n,toa,out4);//拟合后残差
    //     #pragma omp critical
    //     {
    //         for(int i=0;i<n;i++){
    //             fprintf(fp,"%.16g ",out1[i].N-out4[i].N);
    //         }
    //         fprintf(fp,"\n");
    //         fflush(fp);
    //     }
    // }



//     //我们现在要随机生成cluster：
//     sys.My_cluster.n=1000; //对于10M, [0,2r0]
//     sys.My_cluster.stars=malloc(sys.My_cluster.n*sizeof(struct IMBH));
//     if(sys.My_cluster.stars==NULL){
//         printf("malloc error\n");
//     }

//     int ran_seed=42;//这个需要改成每次不一样的之后
//     const gsl_rng_type *T;
//     gsl_rng *r;
//     gsl_rng_env_setup();
//     T=gsl_rng_default;
//     r=gsl_rng_alloc(T);
//     gsl_rng_set(r,ran_seed);
//     for(int i=0;i<sys.My_cluster.n;i++){
//         sys.My_cluster.stars[i].m=10.0*M_sun;
//         double a_rand=gsl_rng_uniform(r)*10.0*r0_S2;
//         sys.My_cluster.stars[i].oe[0]=sqrt(a_rand*a_rand*a_rand*4.0*pi*pi/G/sys.BH.M);
//         sys.My_cluster.stars[i].oe[1]=sqrt(gsl_rng_uniform(r))*0.95;
//         sys.My_cluster.stars[i].oe[2]=2.0*pi*gsl_rng_uniform(r);
//         sys.My_cluster.stars[i].oe[3]=2.0*pi*gsl_rng_uniform(r);
//         sys.My_cluster.stars[i].oe[4]=acos(2.0*gsl_rng_uniform(r)-1.0);
//         sys.My_cluster.stars[i].oe[5]=gsl_rng_uniform(r)*sqrt(a_rand*a_rand*a_rand*4.0*pi*pi/G/sys.BH.M);
//     }
//     gsl_rng_free(r);
// printf("%.16g %.16g\n",sys.My_cluster.stars[48].oe[0],sys.My_cluster.stars[48].oe[1]);
// fflush(stdout);

//     // 生成带扰动的toa
//     int n=1000;
//     double t[n];
//     for(int i=0;i<n;i++){
//         t[i]=(i+1.0)/n*5.0*yr;
//     }
//     sys.psr.op.My_cluster=1;
//     struct toa_output out1[n],out2[n],out3[n],out4[n];
//     toa_generator(sys,n,t,out1);
//     double toa[n],N[n],sigma_TOA[n];
//     for(int i=0;i<n;i++){
//         toa[i]=out1[i].toa;
//         N[i]=out1[i].N;
//         sigma_TOA[i]=1e-3*sec;
//     }

//     residual_cal(sys,n,toa,out2);//检查用
//     sys.psr.op.My_cluster=0;//关闭扰动
//     residual_cal(sys,n,toa,out3);//未拟合的原始残差
//     struct star_system sys_fit=fitting_no_abb_GR(n,toa,N,sigma_TOA,sys);//拟合系统//修改程序使其不拟合0-4
//     residual_cal(sys_fit,n,toa,out4);//拟合后残差

//     FILE *fp=fopen("data2026/MyCluster_try.txt","wt");
//     for(int i=0;i<n;i++){
//         fprintf(fp,"%.16g %.16g %.16g %.16g %.16g\n",t[i],out1[i].N,out2[i].N,out3[i].N,out4[i].N);
//     }
//     fclose(fp);


//     free(sys.My_cluster.stars);

    //来实行我们的想法
    //我们把对给定a和给定M的东西写进函数里方便后续打靶的封装
    // double fracM=IMBH_detectability(sys,10.0*mpc,2000.0*M_sun,100,42);
    // printf("faction of detect = %.16g\n",fracM);

    //下面我们要写一个二分查找的东西，给定a找到m_IMBH，并记录过程
    //再套一个循环即可,
    // FILE *fp=fopen("data2026/try_find_M","wt");
    // FILE *fp2=fopen("data2026/final_M","wt");
    // for(int i_a=0;i_a<20;i_a++){
    // double a_IMBH=pow(10.0,-2.0+4.0*(i_a)/(20.0-1.0))*mpc;
    // double rate=0.68; //以68%为例
    // int n_samp=1000;
    // int ran_seed=42;
    // //我们总是从m_IMBH=1000.0*M_sun开始，首先找到二分的两个边界
    // double m_now=1000.0*M_sun;
    // double m_low,m_high;
    // double frac=IMBH_detectability(sys,a_IMBH,m_now,n_samp,42);
    // fprintf(fp,"%.16g %.16g %.16g\n",a_IMBH/mpc,m_now/M_sun,frac);
    // fflush(fp);

    // int ready_low=0,ready_high=0;
    // if(frac<=rate){
    //     ready_low=1.0;
    //     m_low=m_now;
    //     m_now*=10.0;
    // }else{
    //     ready_high=1;
    //     m_high=m_now;
    //     m_now/=10.0;
    // }
    // while(ready_low==0 || ready_high==0){
    //     frac=IMBH_detectability(sys,a_IMBH,m_now,n_samp,42);
    //     fprintf(fp,"%.16g %.16g %.16g\n",a_IMBH/mpc,m_now/M_sun,frac);
    //     fflush(fp);
    //     if(frac<=rate){
    //         ready_low=1.0;
    //         m_low=m_now;
    //         m_now*=10.0;
    //     }else{
    //         ready_high=1;
    //         m_high=m_now;
    //         m_now/=10.0;
    //     }
    // }
    //然后开始二分查找
    //由于上面m_low和m_high始终差一个量级，我们查找目标是1%的精度
    //我们总是二分10次
    // for(int i=0;i<10;i++){
    //     m_now=(m_low+m_high)/2.0;
    //     frac=IMBH_detectability(sys,a_IMBH,m_now,n_samp,42);
    //     fprintf(fp,"%.16g %.16g %.16g\n",a_IMBH/mpc,m_now/M_sun,frac);
    //     fflush(fp);
    //     if(frac<=rate){
    //         m_low=m_now;
    //     }else{
    //         m_high=m_now;
    //     }
    // }
    // fprintf(fp2,"%.16g %.16g\n",a_IMBH/mpc,(m_low+m_high)/2.0/M_sun);
    // }
    // fclose(fp);
    // fclose(fp2);

    //我们要来看下在固定Mchiq（以及方向）情况下的phase connect拟合效果
    //生成带扰动的toa
    // int n=260;
    // double t[n];
    // for(int i=0;i<n;i++){
    //     t[i]=(i+1.0)/n*5.0*yr;
    // }
    // // sys.psr.op.star_cluster=1;//带扰动
    // sys.psr.op.IMBH=1;
    // struct toa_output out1[n],out2[n],out3[n],out4[n];
    // toa_generator(sys,n,t,out1);
    // double toa[n],N[n],sigma_TOA[n];
    // for(int i=0;i<n;i++){
    //     toa[i]=out1[i].toa;
    //     N[i]=out1[i].N;
    //     sigma_TOA[i]=1e-3*sec;
    // }
    // residual_cal(sys,n,toa,out2);//检查用
    // // sys.psr.op.star_cluster=0;//关闭扰动
    // sys.psr.op.IMBH=0;
    // residual_cal(sys,n,toa,out3);//未拟合的原始残差
    // struct star_system sys_fit=fitting_no_abb_GR(n,toa,N,sigma_TOA,sys);//拟合系统//修改程序使其不拟合0-4
    // residual_cal(sys_fit,n,toa,out4);//拟合后残差

    // FILE *fp=fopen("data2026/IMBH_nospin_residual_100M.txt","wt");

    // for(int i=0;i<n;i++){
    //     fprintf(fp,"%.16g %.16g %.16g %.16g %.16g\n",t[i],out1[i].N,out2[i].N,out3[i].N,out4[i].N);
    // }

    // fclose(fp);


    finish_t=clock();
    total_t=(double)(finish_t-start_t)/CLOCKS_PER_SEC;
    printf("num calculation in %g sec\n",total_t);



    return 0;
}

