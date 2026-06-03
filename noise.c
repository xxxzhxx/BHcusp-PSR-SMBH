#include "noise.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_multimin.h>

#include "const.h"
#include "stars.h"
#include "timing.h"

void white_noise_realization(int n, unsigned long seed,double sigma_TOA[], double delta_TOA[]){
    const gsl_rng_type *T;
    gsl_rng *r;

    gsl_rng_env_setup();
    T=gsl_rng_default;
    r=gsl_rng_alloc(T);
    gsl_rng_set(r,seed);

    for(int i=0;i<n;i++){
        delta_TOA[i]=gsl_ran_gaussian(r,sigma_TOA[i]);
    }

    gsl_rng_free(r);
}

void white_noise_dxi(unsigned long seed, int n, double toa[], double sigma_TOA[], struct star_system sys, int n_real, char *file_name){
    //初始化随机数生成器
    const gsl_rng_type *T;
    gsl_rng *r;

    gsl_rng_env_setup();
    T=gsl_rng_default;
    r=gsl_rng_alloc(T);
    gsl_rng_set(r,seed);

    //以下是计算反复用到的最小二乘矩阵
    gsl_matrix *V=gsl_matrix_calloc(n,n);
    for(int i=0;i<n;i++){
        gsl_matrix_set(V,i,i,1.0/sigma_TOA[i]/sigma_TOA[i]);
    }

    //创建design matrix
    gsl_matrix *M=design_matrix(n,toa,sys);
    int dim=M->size2;

    gsl_matrix *VM=gsl_matrix_alloc(n,dim);
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,V,M,0.0,VM);

    gsl_matrix *MVM=gsl_matrix_alloc(dim,dim);
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,M,VM,0.0,MVM);

    //求逆，且是对称正定的
    gsl_linalg_cholesky_decomp1(MVM);
    gsl_linalg_cholesky_invert(MVM);

    gsl_matrix *factor=gsl_matrix_alloc(dim,n);
    gsl_blas_dgemm(CblasNoTrans,CblasTrans,-1.0,MVM,VM,0.0,factor);

    FILE *fp=fopen(file_name,"wt");
    gsl_vector *dt=gsl_vector_alloc(n);
    gsl_vector *dxi=gsl_vector_alloc(dim);

    for(int i=0;i<n_real;i++){
        for(int j=0;j<n;j++){
            gsl_vector_set(dt,j,gsl_ran_gaussian(r,sigma_TOA[j]));
        }
        gsl_blas_dgemv(CblasNoTrans,1.0,factor,dt,0.0,dxi);
        for(int j=0;j<dim;j++){
            fprintf(fp,"%.16g ",gsl_vector_get(dxi,j));
        }
        fprintf(fp,"\n");
    }
    fclose(fp);

    gsl_matrix_free(V);
    gsl_matrix_free(M);
    gsl_matrix_free(VM);
    gsl_matrix_free(MVM);
    gsl_matrix_free(factor);
    gsl_vector_free(dt);
    gsl_vector_free(dxi);
}

void red_noise_realization(int n, unsigned long seed, double A, double fc, double alpha, double toa[], double delta_TOA[]){
    //由于delta_TOA并不一定均匀，需要使用插值的方法来获得
    double T=(toa[n-1]-toa[0])*2.0;
    //这个地方有些疑问，在生成红噪声时，我频率在上下的扩展有意义吗，有必要吗？
    //我们将采取一个简单些的方法，上下各扩展2倍左右
    int n_fft=(int)pow(2.0,ceil(log2(n)))*4;
    double dt=T/(double)n_fft;

    double data[n_fft];

    const gsl_rng_type *Type;
    gsl_rng *r;

    gsl_rng_env_setup();
    Type=gsl_rng_default;
    r=gsl_rng_alloc(Type);
    gsl_rng_set(r,seed);

    for(int i=1;i<n_fft-1;i++){
        double rand_phase=2.0*pi*gsl_rng_uniform(r);
        double ci=sqrt(A/pow(1.0+((i+1.0)/2.0/T/fc)*((i+1.0)/2.0/T/fc),alpha/2.0)*n_fft/dt);
        data[i]=ci*cos(rand_phase);
        i++;
        data[i]=ci*sin(rand_phase);
    }
    data[0]=sqrt(A*n_fft/dt);
    data[n_fft-1]=sqrt(A/pow(1.0+(1.0/2.0/dt/fc)*(1.0/2.0/dt/fc),alpha/2.0)*n_fft/dt);
    gsl_rng_free(r);

    gsl_fft_halfcomplex_wavetable *hc=gsl_fft_halfcomplex_wavetable_alloc(n_fft);
    gsl_fft_real_workspace *work=gsl_fft_real_workspace_alloc(n_fft);

    gsl_fft_halfcomplex_inverse(data,1,n_fft,hc,work);

    gsl_fft_halfcomplex_wavetable_free(hc);
    gsl_fft_real_workspace_free(work);

    double fft_time[n_fft];
    for(int i=0;i<n_fft;i++){
        fft_time[i]=i*dt;
    }
    gsl_interp_accel *acc=gsl_interp_accel_alloc();
    gsl_spline *spline=gsl_spline_alloc(gsl_interp_steffen,n_fft);
    gsl_spline_init(spline,fft_time,data,n_fft);

    for(int i=0;i<n;i++){
        delta_TOA[i]=gsl_spline_eval(spline,toa[i]-toa[0],acc);
    }
    
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
}

void noise_gen_wfit(unsigned long seed, int n, double toa[], double sigma_TOA[], struct red_noise_params rp, struct star_system sys,\
char *file_name){
    const gsl_rng_type *T;
    gsl_rng *r;

    gsl_rng_env_setup();
    T=gsl_rng_default;
    r=gsl_rng_alloc(T);
    gsl_rng_set(r,seed);

    //生成白噪声
    double white[n];
    for(int i=0;i<n;i++){
        white[i]=gsl_ran_gaussian(r,sigma_TOA[i]);
    }

    //生成红噪声
    double T_fft=(toa[n-1]-toa[0])*2.0;
    int n_fft=(int)pow(2.0,ceil(log2(n)))*4;
    double dt=T_fft/(double)n_fft;

    double A=rp.A;
    double fc=rp.fc;
    double alpha=rp.alpha;

    double data[n_fft];
    for(int i=1;i<n_fft-1;i++){
        double rand_phase=2.0*pi*gsl_rng_uniform(r);
        double ci=sqrt(A/pow(1.0+((i+1.0)/2.0/T_fft/fc)*((i+1.0)/2.0/T_fft/fc),alpha/2.0)*n_fft/dt);
        double rand_ci=gsl_ran_gaussian(r,ci);
        data[i]=rand_ci*cos(rand_phase);
        i++;
        data[i]=rand_ci*sin(rand_phase);
    }
    data[0]=sqrt(A*n_fft/dt);
    data[n_fft-1]=sqrt(A/pow(1.0+(1.0/2.0/dt/fc)*(1.0/2.0/dt/fc),alpha/2.0)*n_fft/dt);

    gsl_fft_halfcomplex_wavetable *hc=gsl_fft_halfcomplex_wavetable_alloc(n_fft);
    gsl_fft_real_workspace *work=gsl_fft_real_workspace_alloc(n_fft);
    gsl_fft_halfcomplex_inverse(data,1,n_fft,hc,work);
    gsl_fft_halfcomplex_wavetable_free(hc);
    gsl_fft_real_workspace_free(work);

    double fft_time[n_fft];
    for(int i=0;i<n_fft;i++){
        fft_time[i]=i*dt;
    }
    gsl_interp_accel *acc=gsl_interp_accel_alloc();
    gsl_spline *spline=gsl_spline_alloc(gsl_interp_steffen,n_fft);
    gsl_spline_init(spline,fft_time,data,n_fft);

    double red[n];
    for(int i=0;i<n;i++){
        red[i]=gsl_spline_eval(spline,toa[i]-toa[0],acc);
    }
    
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);

    gsl_rng_free(r);

    //以下是计算反复用到的最小二乘矩阵
    //按该函数输入构建
    //创建design matrix
    gsl_matrix *M=design_matrix(n,toa,sys);
    int dim=M->size2;

    gsl_matrix *MVM=gsl_matrix_alloc(dim,dim);
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,M,M,0.0,MVM);

    //求逆，且是对称正定的
    gsl_linalg_cholesky_decomp1(MVM);
    gsl_linalg_cholesky_invert(MVM);

    gsl_matrix *factor=gsl_matrix_alloc(dim,n);
    gsl_blas_dgemm(CblasNoTrans,CblasTrans,-1.0,MVM,M,0.0,factor);

    gsl_vector *delta_t=gsl_vector_alloc(n);
    gsl_vector *delta_t2=gsl_vector_alloc(n);//用于计算2阶拟合
    for(int i=0;i<n;i++){
        gsl_vector_set(delta_t,i,white[i]+red[i]);
        gsl_vector_set(delta_t2,i,white[i]+red[i]);
    }
    gsl_vector *dxi=gsl_vector_alloc(dim);
    gsl_blas_dgemv(CblasNoTrans,1.0,factor,delta_t,0.0,dxi);

    //计算拟合后残差
    gsl_blas_dgemv(CblasNoTrans,1.0,M,dxi,1.0,delta_t);

    gsl_vector_free(dxi);
    gsl_matrix_free(M);
    gsl_matrix_free(MVM);
    gsl_matrix_free(factor);

    //需要重新算design matrix
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu,mu_alpha,mu_delta,lambda_p,eta_p
    //0,1  ,2,3     ,4  ,5 ,6,7    ,8    ,9  ,10,11,12,13 ,14  ,15      ,16      ,17      ,18
    bool fit_2[19]={0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0};
    for(int i=0;i<19;i++){
        sys.fit_op[i]=fit_2[i];
    }
    M=design_matrix(n,toa,sys);
    dim=M->size2;

    MVM=gsl_matrix_alloc(dim,dim);
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,M,M,0.0,MVM);

    //求逆，且是对称正定的
    gsl_linalg_cholesky_decomp1(MVM);
    gsl_linalg_cholesky_invert(MVM);

    factor=gsl_matrix_alloc(dim,n);
    gsl_blas_dgemm(CblasNoTrans,CblasTrans,-1.0,MVM,M,0.0,factor);

    dxi=gsl_vector_alloc(dim);
    gsl_blas_dgemv(CblasNoTrans,1.0,factor,delta_t2,0.0,dxi);

    gsl_blas_dgemv(CblasNoTrans,1.0,M,dxi,1.0,delta_t2);

    gsl_vector_free(dxi);
    gsl_matrix_free(M);
    gsl_matrix_free(MVM);
    gsl_matrix_free(factor);

    FILE *fp=fopen(file_name,"wt");
    for(int i=0;i<n;i++){
        // printf("%.16g %.16g %.16g %.16g %.16g\n",toa[i],white[i],red[i],gsl_vector_get(delta_t2,i),gsl_vector_get(delta_t,i));
        fprintf(fp,"%.16g %.16g %.16g %.16g %.16g\n",toa[i],white[i],red[i],gsl_vector_get(delta_t2,i),gsl_vector_get(delta_t,i));
    }
    fclose(fp);

    gsl_vector_free(delta_t);
    gsl_vector_free(delta_t2);
}

void noise_wfit_statistic_A(unsigned long seed, int n, double toa[], double sigma_TOA[], struct red_noise_params rp, \
struct star_system sys, int n_A, double A_range[], int n_real, char *file_name){
    const gsl_rng_type *T;
    gsl_rng *r;

    gsl_rng_env_setup();
    T=gsl_rng_default;
    r=gsl_rng_alloc(T);
    gsl_rng_set(r,seed);

    FILE *fp=fopen(file_name,"wt");

    gsl_matrix *M=design_matrix(n,toa,sys);
    int dim=M->size2;
    gsl_matrix *MVM=gsl_matrix_alloc(dim,dim);
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,M,M,0.0,MVM);
    gsl_linalg_cholesky_decomp1(MVM);
    gsl_linalg_cholesky_invert(MVM);
    gsl_matrix *factor=gsl_matrix_alloc(dim,n);
    gsl_blas_dgemm(CblasNoTrans,CblasTrans,-1.0,MVM,M,0.0,factor);
    gsl_vector *delta_t=gsl_vector_alloc(n);
    gsl_vector *dxi=gsl_vector_alloc(dim);

    double T_fft=(toa[n-1]-toa[0])*2.0;
    int n_fft=(int)pow(2.0,ceil(log2(n)))*4;
    double dt=T_fft/(double)n_fft;
    double fft_time[n_fft];
    gsl_fft_halfcomplex_wavetable *hc=gsl_fft_halfcomplex_wavetable_alloc(n_fft);
    gsl_fft_real_workspace *work=gsl_fft_real_workspace_alloc(n_fft);
    gsl_interp_accel *acc=gsl_interp_accel_alloc();
    gsl_spline *spline=gsl_spline_alloc(gsl_interp_steffen,n_fft);

    for(int i=0;i<n_fft;i++){
        fft_time[i]=i*dt;
    }
    for(int i_A=0;i_A<n_A;i_A++){
        double white[n];

        rp.A=A_range[i_A];
        double red[n];
        double ci[n_fft/2+1];
        for(int i_fft=0;i_fft<n_fft/2+1;i_fft++){
            ci[i_fft]=sqrt(rp.A/pow(1.0+(i_fft/T_fft/rp.fc)*(i_fft/T_fft/rp.fc),rp.alpha/2.0)*n_fft/dt);
        }
        double data[n_fft];

        double delta_xi[dim];
        double sigma_xi[dim];
        double R2=0;
        for(int i_dim=0;i_dim<dim;i_dim++){
            delta_xi[i_dim]=0;
            sigma_xi[i_dim]=0;
        }
        double dxi_real[dim][n_real];

        for(int i_real=0;i_real<n_real;i_real++){
            for(int i_noise=0;i_noise<n;i_noise++){
                white[i_noise]=gsl_ran_gaussian(r,sigma_TOA[i_noise]);
            }

            for(int i=1;i<n_fft-1;i++){
                double rand_phase=2.0*pi*gsl_rng_uniform(r);
                data[i]=ci[(i+1)/2]*cos(rand_phase);
                i++;
                data[i]=ci[(i+1)/2]*sin(rand_phase);
            }
            data[0]=ci[0];
            data[n_fft-1]=ci[n_fft/2];
            gsl_fft_halfcomplex_inverse(data,1,n_fft,hc,work);
            
            gsl_spline_init(spline,fft_time,data,n_fft);
            for(int i_noise=0;i_noise<n;i_noise++){
                red[i_noise]=gsl_spline_eval(spline,toa[i_noise]-toa[0],acc);
            }

            for(int i_noise=0;i_noise<n;i_noise++){
                gsl_vector_set(delta_t,i_noise,white[i_noise]+red[i_noise]);
            }
            gsl_blas_dgemv(CblasNoTrans,1.0,factor,delta_t,0.0,dxi);
            gsl_blas_dgemv(CblasNoTrans,1.0,M,dxi,1.0,delta_t);

            for(int i_dim=0;i_dim<dim;i_dim++){
                dxi_real[i_dim][i_real]=gsl_vector_get(dxi,i_dim);
            }
            double dot_result;
            gsl_blas_ddot(delta_t,delta_t,&dot_result);
            R2+=dot_result/n_real;
        }

        for(int i_dim=0;i_dim<dim;i_dim++){
            for(int i_real=0;i_real<n_real;i_real++){
                delta_xi[i_dim]+=dxi_real[i_dim][i_real]/n_real;
            }
            for(int i_real=0;i_real<n_real;i_real++){
                sigma_xi[i_dim]+=pow(dxi_real[i_dim][i_real]-delta_xi[i_dim],2.0)/(n_real-1.0);
            }
            sigma_xi[i_dim]=sqrt(sigma_xi[i_dim]);
        }

        fprintf(fp,"%.16g ",A_range[i_A]);
        for(int i_dim=0;i_dim<dim;i_dim++){
            fprintf(fp,"%.16g %.16g ",delta_xi[i_dim],sigma_xi[i_dim]);
        }
        fprintf(fp,"%.16g\n",R2);
        printf("%d/%d finished\n",i_A+1,n_A);
    }

    gsl_rng_free(r);
    fclose(fp);
    gsl_matrix_free(M);
    gsl_matrix_free(MVM);
    gsl_matrix_free(factor);
    gsl_vector_free(delta_t);
    gsl_vector_free(dxi);
    gsl_fft_halfcomplex_wavetable_free(hc);
    gsl_fft_real_workspace_free(work);
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
}

void noise_Cfit_statistic_A(unsigned long seed, int n, double toa[], double sigma_TOA[], struct red_noise_params rp, \
struct star_system sys, int n_A, double A_range[], int n_real, char *file_name){
    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup();
    T=gsl_rng_default;
    r=gsl_rng_alloc(T);
    gsl_rng_set(r,seed);

    FILE *fp=fopen(file_name,"wt");

    gsl_matrix *M=design_matrix(n,toa,sys);
    int dim=M->size2;
    gsl_matrix *MVM=gsl_matrix_alloc(dim,dim);
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,M,M,0.0,MVM);
    gsl_linalg_cholesky_decomp1(MVM);
    gsl_linalg_cholesky_invert(MVM);
    gsl_matrix *factor=gsl_matrix_alloc(dim,n);
    gsl_blas_dgemm(CblasNoTrans,CblasTrans,-1.0,MVM,M,0.0,factor);
    gsl_vector *delta_t=gsl_vector_alloc(n);
    gsl_vector *dxi=gsl_vector_alloc(dim);

    gsl_vector *delta_t_r=gsl_vector_alloc(n);
    gsl_vector *dxi_r=gsl_vector_alloc(dim);

    double T_fft=(toa[n-1]-toa[0])*2.0;
    int n_fft=(int)pow(2.0,ceil(log2(n)))*4;
    double dt=T_fft/(double)n_fft;
    double fft_time[n_fft];
    gsl_fft_halfcomplex_wavetable *hc=gsl_fft_halfcomplex_wavetable_alloc(n_fft);
    gsl_fft_real_workspace *work=gsl_fft_real_workspace_alloc(n_fft);
    gsl_interp_accel *acc=gsl_interp_accel_alloc();
    gsl_spline *spline=gsl_spline_alloc(gsl_interp_steffen,n_fft);
    for(int i=0;i<n_fft;i++){
        fft_time[i]=i*dt;
    }

    gsl_vector *CR=gsl_vector_alloc(n);

    for(int i_A=0;i_A<n_A;i_A++){
        //Cov整个得在里面算。。。
        //清除也得在里面


        double white[n];

        rp.A=A_range[i_A];
        double red[n];
        double ci[n_fft/2+1];
        for(int i_fft=0;i_fft<n_fft/2+1;i_fft++){
            ci[i_fft]=sqrt(rp.A/pow(1.0+(i_fft/T_fft/rp.fc)*(i_fft/T_fft/rp.fc),rp.alpha/2.0)*n_fft/dt);
        }
        double data[n_fft];

        gsl_matrix *Cov=residual_covariance(n,toa,sigma_TOA,rp);
        gsl_linalg_cholesky_decomp1(Cov);
        gsl_linalg_cholesky_invert(Cov);
        gsl_matrix *CM=gsl_matrix_alloc(n,dim);
        gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,Cov,M,0.0,CM);
        gsl_matrix *MCM=gsl_matrix_alloc(dim,dim);
        gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,M,CM,0.0,MCM);
        gsl_linalg_cholesky_decomp1(MCM);
        gsl_linalg_cholesky_invert(MCM);
        gsl_matrix *factor_r=gsl_matrix_alloc(dim,n);
        gsl_blas_dgemm(CblasNoTrans,CblasTrans,-1.0,MCM,CM,0.0,factor_r);

        double delta_xi[dim],delta_xi_r[dim];
        double sigma_xi[dim],sigma_xi_r[dim];
        double R2=0,R2_r=0;
        for(int i_dim=0;i_dim<dim;i_dim++){
            delta_xi[i_dim]=0;
            sigma_xi[i_dim]=0;
            delta_xi_r[i_dim]=0;
            sigma_xi_r[i_dim]=0;
        }
        //这一步太占内存？？？感觉是整块的空间分配不出来，我们能拆散了分配吗？
        // double dxi_real[dim][n_real],dxi_real_r[dim][n_real];
        double **dxi_real = (double **)malloc(dim * sizeof(double *));
        double **dxi_real_r = (double **)malloc(dim * sizeof(double *));
        if (dxi_real == NULL || dxi_real_r == NULL) {
            fprintf(stderr, "内存分配失败\n");
            return;
        }
        for (int i = 0; i < dim; i++) {
            dxi_real[i] = (double *)malloc(n_real * sizeof(double));
            dxi_real_r[i] = (double *)malloc(n_real * sizeof(double));
            if (dxi_real[i] == NULL || dxi_real_r[i] == NULL) {
                fprintf(stderr, "内存分配失败\n");
                return;
            }
        }

        for(int i_real=0;i_real<n_real;i_real++){
            for(int i_noise=0;i_noise<n;i_noise++){
                white[i_noise]=gsl_ran_gaussian(r,sigma_TOA[i_noise]);
            }

            for(int i=1;i<n_fft-1;i++){
                double rand_phase=2.0*pi*gsl_rng_uniform(r);
                double rand_ci=gsl_ran_gaussian(r,ci[(i+1)/2]);
                data[i]=rand_ci*cos(rand_phase);
                i++;
                data[i]=rand_ci*sin(rand_phase);
            }
            data[0]=gsl_ran_gaussian(r,ci[0]);
            data[n_fft-1]=gsl_ran_gaussian(r,ci[n_fft/2]);
            gsl_fft_halfcomplex_inverse(data,1,n_fft,hc,work);
            
            gsl_spline_init(spline,fft_time,data,n_fft);
            for(int i_noise=0;i_noise<n;i_noise++){
                red[i_noise]=gsl_spline_eval(spline,toa[i_noise]-toa[0],acc);
            }

            for(int i_noise=0;i_noise<n;i_noise++){
                gsl_vector_set(delta_t,i_noise,white[i_noise]+red[i_noise]);
                gsl_vector_set(delta_t_r,i_noise,white[i_noise]+red[i_noise]);
            }
            gsl_blas_dgemv(CblasNoTrans,1.0,factor,delta_t,0.0,dxi);
            gsl_blas_dgemv(CblasNoTrans,1.0,M,dxi,1.0,delta_t);
            gsl_blas_dgemv(CblasNoTrans,1.0,factor_r,delta_t_r,0.0,dxi_r);
            gsl_blas_dgemv(CblasNoTrans,1.0,M,dxi_r,1.0,delta_t_r);

            for(int i_dim=0;i_dim<dim;i_dim++){
                dxi_real[i_dim][i_real]=gsl_vector_get(dxi,i_dim);
            }
            double dot_result;
            gsl_blas_ddot(delta_t,delta_t,&dot_result);
            R2+=dot_result/n_real;

            for(int i_dim=0;i_dim<dim;i_dim++){
                dxi_real_r[i_dim][i_real]=gsl_vector_get(dxi_r,i_dim);
            }
            gsl_blas_dgemv(CblasNoTrans,1.0,Cov,delta_t_r,0.0,CR);
            gsl_blas_ddot(delta_t_r,CR,&dot_result);
            R2_r+=dot_result/n_real;
        }

        for(int i_dim=0;i_dim<dim;i_dim++){
            for(int i_real=0;i_real<n_real;i_real++){
                delta_xi[i_dim]+=dxi_real[i_dim][i_real]/n_real;
            }
            for(int i_real=0;i_real<n_real;i_real++){
                sigma_xi[i_dim]+=pow(dxi_real[i_dim][i_real]-delta_xi[i_dim],2.0)/(n_real-1.0);
            }
            sigma_xi[i_dim]=sqrt(sigma_xi[i_dim]);
        }

        for(int i_dim=0;i_dim<dim;i_dim++){
            for(int i_real=0;i_real<n_real;i_real++){
                delta_xi_r[i_dim]+=dxi_real_r[i_dim][i_real]/n_real;
            }
            for(int i_real=0;i_real<n_real;i_real++){
                sigma_xi_r[i_dim]+=pow(dxi_real_r[i_dim][i_real]-delta_xi_r[i_dim],2.0)/(n_real-1.0);
            }
            sigma_xi_r[i_dim]=sqrt(sigma_xi_r[i_dim]);
        }

        fprintf(fp,"%.16g ",A_range[i_A]);
        for(int i_dim=0;i_dim<dim;i_dim++){
            fprintf(fp,"%.16g %.16g %.16g %.16g ",sigma_xi[i_dim],sqrt(gsl_matrix_get(MVM,i_dim,i_dim)*R2/(n-dim)),\
            sigma_xi_r[i_dim],sqrt(gsl_matrix_get(MCM,i_dim,i_dim)*R2_r/(n-dim)));
        }
        fprintf(fp,"%.16g\n",R2);
        printf("%d/%d finished\n",i_A+1,n_A);

        for (int i = 0; i < dim; i++) {
            free(dxi_real[i]);
            free(dxi_real_r[i]);
        }
        free(dxi_real);
        free(dxi_real_r);

        gsl_matrix_free(Cov);
        gsl_matrix_free(CM);
        gsl_matrix_free(MCM);
        gsl_matrix_free(factor_r);
    }

    gsl_rng_free(r);
    fclose(fp);
    gsl_matrix_free(M);
    gsl_matrix_free(MVM);
    gsl_matrix_free(factor);
    gsl_vector_free(delta_t);
    gsl_vector_free(dxi);
    gsl_vector_free(delta_t_r);
    gsl_vector_free(dxi_r);
    gsl_fft_halfcomplex_wavetable_free(hc);
    gsl_fft_real_workspace_free(work);
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
    gsl_vector_free(CR);
}

//最小化函数
//v里是三个noise参数，顺序为log_10_A, alpha, EFAC
//params中应当给出不必重复计算的部分包括 delta_t,
double noise_params_fit_func(const gsl_vector *v, void *params){
    struct noise_params_fit_func_params *p=(struct noise_params_fit_func_params *)params;
    gsl_vector *delta_t=p->delta_t;
    double sigma_TOA=p->sigma_TOA;
    int n=p->n;
    int k=p->k;
    gsl_matrix *F=p->F;
    gsl_matrix *FF=p->FF;
    double T_obs=p->T_obs;
    gsl_matrix *U=p->U;
    int n_params=p->n_params;
    gsl_matrix *phi_inv=p->phi_inv;
    gsl_matrix *Sigma=p->Sigma;
    gsl_matrix *Sigma_inv=p->Sigma_inv;
    gsl_matrix *Y=p->Y;
    gsl_matrix *FS=p->FS;
    gsl_matrix *FSF=p->FSF;
    gsl_matrix *YU=p->YU;
    gsl_matrix *UYU=p->UYU;
    gsl_matrix *UYU_inv=p->UYU_inv;
    gsl_matrix *YU_UYU=p->YU_UYU;
    gsl_matrix *YU_UYU_UY=p->YU_UYU_UY;
    gsl_matrix *X=p->X;
    gsl_vector *Xt=p->Xt;

    double A=pow(10.0,gsl_vector_get(v,0));
    double alpha=gsl_vector_get(v,1);
    double EFAC=gsl_vector_get(v,2);

    double ln_det_phi=0;
    for(int i=0;i<k;i++){
        double phi_inv_ii=pow(i+1.0,alpha)/(2*A*T_obs*T_obs);
        gsl_matrix_set(phi_inv,i,i,phi_inv_ii);
        gsl_matrix_set(phi_inv,i+k,i+k,phi_inv_ii);
        ln_det_phi+=-2.0*log(phi_inv_ii);
    }

    double ln_det_Cw=2.0*n*log(EFAC*sigma_TOA);

    for(int i=0;i<2*k;i++){
        for(int j=0;j<2*k;j++){
            gsl_matrix_set(Sigma,i,j,gsl_matrix_get(FF,i,j)/EFAC/EFAC/sigma_TOA/sigma_TOA+gsl_matrix_get(phi_inv,i,j));
        }
    }
    
    gsl_matrix_memcpy(Sigma_inv,Sigma);
    gsl_linalg_cholesky_decomp1(Sigma_inv);
    double ln_det_Sigma=0;
    for(int i=0;i<2*k;i++){
        ln_det_Sigma+=2.0*log(gsl_matrix_get(Sigma_inv,i,i));
    }
    gsl_linalg_cholesky_invert(Sigma_inv);

    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,F,Sigma_inv,0.0,FS);
    gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,FS,F,0.0,FSF);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(i==j){
                gsl_matrix_set(Y,i,j,(1.0-gsl_matrix_get(FSF,i,j)/EFAC/EFAC/sigma_TOA/sigma_TOA)/EFAC/EFAC/sigma_TOA/sigma_TOA);
            }else{
                gsl_matrix_set(Y,i,j,(-gsl_matrix_get(FSF,i,j)/EFAC/EFAC/sigma_TOA/sigma_TOA)/EFAC/EFAC/sigma_TOA/sigma_TOA);
            }
            
        }
    }

    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,Y,U,0.0,YU);
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,U,YU,0.0,UYU);

    gsl_matrix_memcpy(UYU_inv,UYU);
    gsl_linalg_cholesky_decomp1(UYU_inv);
    double ln_det_UYU=0;
    for(int i=0;i<2*k;i++){
        ln_det_UYU+=2.0*log(gsl_matrix_get(UYU_inv,i,i));
    }
    gsl_linalg_cholesky_invert(UYU_inv);

    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,YU,UYU_inv,0.0,YU_UYU);
    gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,YU_UYU,YU,0.0,YU_UYU_UY);

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            gsl_matrix_set(X,i,j,gsl_matrix_get(Y,i,j)-gsl_matrix_get(YU_UYU_UY,i,j));
        }
    }
    
    gsl_blas_dgemv(CblasNoTrans,1.0,X,delta_t,0.0,Xt);
    
    double chi2[1];
    gsl_blas_ddot(delta_t,Xt,chi2);

    return 0.5*(chi2[0]+ln_det_phi+ln_det_Cw+ln_det_Sigma+ln_det_UYU);
}
//N_r是指定生成实例个数
void noise_params_fit_gen(unsigned long seed, int N_r, int n, double toa[], double sigma_TOA[], struct red_noise_params rp, int k, \
struct star_system sys, char *file_name){
    FILE *fp=fopen(file_name,"wt");

    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup();
    T=gsl_rng_default;
    r=gsl_rng_alloc(T);
    gsl_rng_set(r,seed);

    double A=rp.A;
    double fc=rp.fc;
    double alpha=rp.alpha;
    double T_fft=(toa[n-1]-toa[0])*2.0;
    int n_fft=(int)pow(2.0,ceil(log2(n)))*4;
    double dt=T_fft/(double)n_fft;

    double white[n];
    double data[n_fft];
    double fft_time[n_fft];
    gsl_fft_halfcomplex_wavetable *hc=gsl_fft_halfcomplex_wavetable_alloc(n_fft);
    gsl_fft_real_workspace *work=gsl_fft_real_workspace_alloc(n_fft);
    gsl_interp_accel *acc=gsl_interp_accel_alloc();
    gsl_spline *spline=gsl_spline_alloc(gsl_interp_steffen,n_fft);
    double red[n];
    gsl_vector *delta_t=gsl_vector_alloc(n);

    double T_obs=toa[n-1]-toa[0];
    gsl_matrix *F=gsl_matrix_alloc(n,2*k);
    for(int i=0;i<n;i++){
        for(int j=0;j<k;j++){
            gsl_matrix_set(F,i,j,cos(2.0*pi*toa[i]*(j+1.0)/T_obs));
            gsl_matrix_set(F,i,j+k,sin(2.0*pi*toa[i]*(j+1.0)/T_obs));
        }
    }
    gsl_matrix *FF=gsl_matrix_alloc(2*k,2*k);
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,F,F,0.0,FF);

    gsl_matrix *M=design_matrix(n,toa,sys);
    int n_params=M->size2;
    gsl_matrix *U=gsl_matrix_calloc(n,n_params);
    gsl_matrix *V=gsl_matrix_calloc(n_params,n_params);
    gsl_vector *S=gsl_vector_calloc(n_params);
    gsl_matrix *S_inv=gsl_matrix_calloc(n_params,n_params);
    gsl_vector *work_SVD=gsl_vector_calloc(n_params);
    gsl_matrix_memcpy(U,M);
    gsl_linalg_SV_decomp(U,V,S,work_SVD);
    for(int i=0;i<n_params;i++){
        gsl_matrix_set(S_inv,i,i,1.0/gsl_vector_get(S,i));
    }
    gsl_vector *delta_Theta=gsl_vector_alloc(n_params);
    gsl_vector *delta_Theta_mid=gsl_vector_alloc(n_params);

    struct noise_params_fit_func_params p;
    p.n=n;
    p.k=k;
    p.n_params=n_params;
    p.T_obs=T_obs;
    p.sigma_TOA=sigma_TOA[0];
    p.delta_t=delta_t; //这里反正输入的是指针，之后会再改具体值
    p.F=F;
    p.FF=FF;
    p.U=U;
    //分配工作空间
    p.phi_inv=gsl_matrix_calloc(2*k,2*k);
    p.Sigma=gsl_matrix_alloc(2*k,2*k);
    p.Sigma_inv=gsl_matrix_calloc(2*k,2*k);
    p.Y=gsl_matrix_alloc(n,n);
    p.FS=gsl_matrix_alloc(n,2*k);
    p.FSF=gsl_matrix_alloc(n,n);
    p.YU=gsl_matrix_alloc(n,n_params);
    p.UYU=gsl_matrix_alloc(n_params,n_params);
    p.UYU_inv=gsl_matrix_alloc(n_params,n_params);
    p.YU_UYU=gsl_matrix_alloc(n,n_params);
    p.YU_UYU_UY=gsl_matrix_alloc(n,n);
    p.X=gsl_matrix_alloc(n,n);
    p.Xt=gsl_vector_alloc(n);

    gsl_vector *red_para=gsl_vector_alloc(3);

    const gsl_multimin_fminimizer_type *T_min=gsl_multimin_fminimizer_nmsimplex2rand;
    gsl_multimin_fminimizer *s=gsl_multimin_fminimizer_alloc(T_min,3);
    gsl_vector *ss=gsl_vector_alloc(3), *x=red_para;
    gsl_multimin_function min_func;
    min_func.n=3;
    min_func.f=noise_params_fit_func;
    min_func.params=&p;

    clock_t start_t,finish_t;

    start_t=clock();

    for(int i_r=0;i_r<N_r;i_r++){
        //生成噪声
        for(int i=0;i<n;i++){
            white[i]=gsl_ran_gaussian(r,sigma_TOA[i]);
        }

        for(int i=1;i<n_fft-1;i++){
            double rand_phase=2.0*pi*gsl_rng_uniform(r);
            double ci=sqrt(A/pow(1.0+((i+1.0)/2.0/T_fft/fc)*((i+1.0)/2.0/T_fft/fc),alpha/2.0)*n_fft/dt);
            double rand_ci=gsl_ran_gaussian(r,ci);
            data[i]=rand_ci*cos(rand_phase);
            i++;
            data[i]=rand_ci*sin(rand_phase);
        }
        data[0]=sqrt(A*n_fft/dt);
        data[n_fft-1]=sqrt(A/pow(1.0+(1.0/2.0/dt/fc)*(1.0/2.0/dt/fc),alpha/2.0)*n_fft/dt);

        gsl_fft_halfcomplex_inverse(data,1,n_fft,hc,work);
        
        for(int i=0;i<n_fft;i++){
            fft_time[i]=i*dt;
        }
        
        gsl_spline_init(spline,fft_time,data,n_fft);

        for(int i=0;i<n;i++){
            red[i]=gsl_spline_eval(spline,toa[i]-toa[0],acc);
        }

        for(int i=0;i<n;i++){
            gsl_vector_set(delta_t,i,white[i]+red[i]);
        }

        //然后要针对生成的这个噪声最小化红噪声参数red_para
        //考虑到我们的likelihood长得并不复杂，我们先试试只用一种最小化一次的效果
        //但我们实际上只能用不需要导数的，所以也没得选
        size_t iter=0;
        int status;
        double size;
        
        gsl_vector_set(red_para,0,log10(rp.A));
        gsl_vector_set(red_para,1,rp.alpha);
        gsl_vector_set(red_para,2,1.0);
        gsl_vector_set_all(ss,0.1);

        gsl_multimin_fminimizer_set(s,&min_func,x,ss);
        // printf("%.16g ",min_func.f(x,&p));
        do{
            iter++;
            status=gsl_multimin_fminimizer_iterate(s);
            if(status){
                break;
            }
            size=gsl_multimin_fminimizer_size(s);
            status=gsl_multimin_test_size(size,1e-6);
            if(status==GSL_SUCCESS){
                printf ("converged to minimum at %ld\n",iter);
            }
        }while (status == GSL_CONTINUE && iter < 1000);
        // printf("%.16g %.16g %.16g %.16g\n",s->fval,gsl_vector_get(s->x,0),gsl_vector_get(s->x,1),gsl_vector_get(s->x,2));

        //实际上目前这个单纯算红噪声的程序是不经济的，因为不会存储noise,必须要把其他参数也还原回来，但我们姑且如此
        //因此我们下面来计算delta_Theta
        //我们完全可以认为此时p的工作区中存储的就是我们最后一次的矩阵，当然，为防止意外，我们不介意这个时间，顺带也可以作为输出
        double fval=min_func.f(x,&p);
        gsl_blas_dgemv(CblasTrans,1.0,p.YU,delta_t,0.0,delta_Theta_mid);
        gsl_blas_dgemv(CblasNoTrans,1.0,p.UYU_inv,delta_Theta_mid,0.0,delta_Theta);
        gsl_blas_dgemv(CblasNoTrans,1.0,S_inv,delta_Theta,0.0,delta_Theta_mid);
        gsl_blas_dgemv(CblasNoTrans,1.0,V,delta_Theta_mid,0.0,delta_Theta);

        fprintf(fp,"%.16g %.16g %.16g ",gsl_vector_get(s->x,0),gsl_vector_get(s->x,1),gsl_vector_get(s->x,2));
        for(int i=0;i<n_params;i++){
            fprintf(fp,"%.16g ",gsl_vector_get(delta_Theta,i));
        }
        fprintf(fp,"\n");


        finish_t=clock();
        printf("%d/%d  res_time = %.16g\n",i_r+1,N_r,(N_r-i_r-1.0)*(double)(finish_t-start_t)/CLOCKS_PER_SEC/(i_r+1.0));
    }
    gsl_fft_halfcomplex_wavetable_free(hc);
    gsl_fft_real_workspace_free(work);
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
    gsl_vector_free(delta_t);
    gsl_matrix_free(F);
    gsl_matrix_free(FF);
    gsl_matrix_free(M);
    gsl_matrix_free(U);
    gsl_matrix_free(V);
    gsl_vector_free(S);
    gsl_vector_free(work_SVD);
    gsl_vector_free(red_para);
    gsl_matrix_free(p.phi_inv);
    gsl_matrix_free(p.Sigma);
    gsl_matrix_free(p.Sigma_inv);
    gsl_matrix_free(p.Y);
    gsl_matrix_free(p.FS);
    gsl_matrix_free(p.FSF);
    gsl_matrix_free(p.YU);
    gsl_matrix_free(p.UYU);
    gsl_matrix_free(p.UYU_inv);
    gsl_matrix_free(p.YU_UYU);
    gsl_matrix_free(p.YU_UYU_UY);
    gsl_matrix_free(p.X);
    gsl_vector_free(p.Xt);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free(s);
    gsl_matrix_free(S_inv);
    gsl_vector_free(delta_Theta);
    gsl_vector_free(delta_Theta_mid);

    fclose(fp);
}