#include "fitting2.h"

#include <stdio.h>
#include <time.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "const.h"
#include "stars.h"
#include "timing.h"

//这个要直接返回最佳拟合N0，nu,dnu后的TOA,不管sys里给了啥，假设无abb项
static double chi2_no_abb(int n, double toa[],double N[],double sigma_TOA[],struct star_system sys){
    // return chi2_white(n,toa,N,sigma_TOA,sys);
    struct toa_output out[n];
    residual_cal(sys,n,toa,out);
    // if(out[n-1].T==0){//这是参数太逆天没积分出去?
    //     return chi2_white(n,toa,N,sigma_TOA,sys);
    // }

    gsl_matrix *A=gsl_matrix_alloc(n,3);
    gsl_vector *B=gsl_vector_alloc(n);

    for(int i=0;i<n;i++){
        gsl_vector_set(B,i,-out[i].T/sigma_TOA[i]);
        gsl_matrix_set(A,i,0,1.0/sigma_TOA[i]);
        gsl_matrix_set(A,i,1,out[i].T*out[i].T/2.0/sigma_TOA[i]);
        gsl_matrix_set(A,i,2,-N[i]/sigma_TOA[i]);
    }

    gsl_matrix *T=gsl_matrix_alloc(3,3);
    gsl_linalg_QR_decomp_r(A,T);

    gsl_vector *x=gsl_vector_alloc(n);
    gsl_vector *work=gsl_vector_alloc(3);
    gsl_linalg_QR_lssolve_r(A,T,B,x,work);

    double chi2=0;
    for(int i=3;i<n;i++){
        chi2+=gsl_vector_get(x,i)*gsl_vector_get(x,i);
    }

    gsl_matrix_free(A);
    gsl_vector_free(B);
    gsl_matrix_free(T);
    gsl_vector_free(x);
    gsl_vector_free(work);
    
    // printf("%.16g\n",chi2);
    // if(chi2==0){
    //     getchar();
    // }
    return chi2;
}

static struct star_system best_chi2_no_abb_sys(int n, double toa[],double N[],double sigma_TOA[],struct star_system sys){
    struct toa_output out[n];
    residual_cal(sys,n,toa,out);

    gsl_matrix *A=gsl_matrix_alloc(n,3);
    gsl_vector *B=gsl_vector_alloc(n);

    for(int i=0;i<n;i++){
        gsl_vector_set(B,i,-out[i].T/sigma_TOA[i]);
        gsl_matrix_set(A,i,0,1.0/sigma_TOA[i]);
        gsl_matrix_set(A,i,1,out[i].T*out[i].T/2.0/sigma_TOA[i]);
        gsl_matrix_set(A,i,2,-N[i]/sigma_TOA[i]);
    }

    gsl_matrix *T=gsl_matrix_alloc(3,3);
    gsl_linalg_QR_decomp_r(A,T);

    gsl_vector *x=gsl_vector_alloc(n);
    gsl_vector *work=gsl_vector_alloc(3);
    gsl_linalg_QR_lssolve_r(A,T,B,x,work);

    sys.psr.N0=gsl_vector_get(x,0)/gsl_vector_get(x,2);
    sys.psr.nu=1.0/gsl_vector_get(x,2);
    sys.psr.dnu=gsl_vector_get(x,1)/gsl_vector_get(x,2);
    // printf("%.16g\n",sys.psr.nu*sec);
    // printf("%.16g\n",chi2_no_abb(n,toa,N,sigma_TOA,sys));

    gsl_matrix_free(A);
    gsl_vector_free(B);
    gsl_matrix_free(T);
    gsl_vector_free(x);
    gsl_vector_free(work);
    
    return sys;
}

struct fitting_no_abb_GR_para{
    int n;
    double *toa;
    double *N;
    double *sigma_TOA;
    struct star_system sys;
};

static double fitting_no_abb_GR_func(const gsl_vector *v,void *params){
    struct fitting_no_abb_GR_para *p=(struct fitting_no_abb_GR_para *)params;
    struct star_system sys=p->sys;
    
    int dim=0;
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu
    if(p->sys.fit_op[0]==1){
        // printf("%.16g\n",sys.BH.M);
        sys.BH.M=gsl_vector_get(v,dim);
        dim+=1;
    }
    if(p->sys.fit_op[1]==1){
        sys.BH.chi=gsl_vector_get(v,dim);
        dim+=1;
    }
    if(p->sys.fit_op[2]==1){
        sys.BH.q=gsl_vector_get(v,dim);
        dim+=1;
    }
    if(p->sys.fit_op[3]==1){
        sys.BH.lambda=gsl_vector_get(v,dim);
        dim+=1;
    }
    if(p->sys.fit_op[4]==1){
        sys.BH.eta=gsl_vector_get(v,dim);
        dim+=1;
    }
    if(p->sys.fit_op[5]==1){
        sys.psr.oe[0]=gsl_vector_get(v,dim);
        dim+=1;
    }
    if(p->sys.fit_op[6]==1){
        sys.psr.oe[1]=gsl_vector_get(v,dim);
        if(sys.psr.oe[1]>1-5e-2){
            sys.psr.oe[1]=1-5e-2;
        }else if(sys.psr.oe[1]<5e-2){
            sys.psr.oe[1]=5e-2;
        }
        dim+=1;
    }
    if(p->sys.fit_op[7]==1){
        sys.psr.oe[2]=gsl_vector_get(v,dim);
        dim+=1;
    }
    if(p->sys.fit_op[8]==1){
        sys.psr.oe[3]=gsl_vector_get(v,dim);
        dim+=1;
    }
    if(p->sys.fit_op[9]==1){
        sys.psr.oe[4]=gsl_vector_get(v,dim);
        dim+=1;
    }
    if(p->sys.fit_op[10]==1){
        sys.psr.oe[5]=gsl_vector_get(v,dim);
        dim+=1;
    }
    
    star_PSR_rv_ini(&sys.psr,sys.BH,0);

    //接下来要计算对应的N0,nu,dnu?
    //既然不返回，不如直接给出最佳拟合chi2
    double chi2=chi2_no_abb(p->n,p->toa,p->N,p->sigma_TOA,sys);
    return chi2;
}

struct star_system fitting_no_abb_GR(int n,double toa[],double N[],double sigma_TOA[],struct star_system sys){
    double chi2_ini=chi2_white(n,toa,N,sigma_TOA,sys);
    // double chi2_ini=chi2_no_abb(n,toa,N,sigma_TOA,sys);
    printf("chi2_ini = %g\n",chi2_ini);

    int high_iter=0,not_increase=0;
    double chi_last=chi2_ini;

    do{
    //我们从单参数开始，首先对不包括Omega的所有参数进行一轮，也不包括自转
    for(int i=0;i<11;i++){
        if(i==8||i==2||i==1||i==3||i==4){ //考虑GR
            continue;
        }
        //首先我们就把sys的op调到正确位置
        for(int j=0;j<28;j++){
            sys.fit_op[j]=0;
        }
        sys.fit_op[i]=1;

        int iter=0;
        double size;
        int status;
        const gsl_multimin_fminimizer_type *T=gsl_multimin_fminimizer_nmsimplex2;
        gsl_multimin_fminimizer *s=gsl_multimin_fminimizer_alloc(T,1);
        //初始值
        gsl_vector *x=gsl_vector_alloc(1);
        int dim=0;
        if(1){
            if(sys.fit_op[0]==1){
                gsl_vector_set(x,dim,sys.BH.M);
                dim+=1;
            }
            if(sys.fit_op[1]==1){
                gsl_vector_set(x,dim,sys.BH.chi);
                dim+=1;
            }
            if(sys.fit_op[2]==1){
                gsl_vector_set(x,dim,sys.BH.q);
                dim+=1;
            }
            if(sys.fit_op[3]==1){
                gsl_vector_set(x,dim,sys.BH.lambda);
                dim+=1;
            }
            if(sys.fit_op[4]==1){
                gsl_vector_set(x,dim,sys.BH.eta);
                dim+=1;
            }
            if(sys.fit_op[5]==1){
                gsl_vector_set(x,dim,sys.psr.oe[0]);
                dim+=1;
            }
            if(sys.fit_op[6]==1){
                gsl_vector_set(x,dim,sys.psr.oe[1]);
                dim+=1;
            }
            if(sys.fit_op[7]==1){
                gsl_vector_set(x,dim,sys.psr.oe[2]);
                dim+=1;
            }
            if(sys.fit_op[8]==1){
                gsl_vector_set(x,dim,sys.psr.oe[3]);
                dim+=1;
            }
            if(sys.fit_op[9]==1){
                gsl_vector_set(x,dim,sys.psr.oe[4]);
                dim+=1;
            }
            if(sys.fit_op[10]==1){
                gsl_vector_set(x,dim,sys.psr.oe[5]);
                dim+=1;
            }
        }

        //步长
        gsl_vector *ss=gsl_vector_alloc(1);
        dim=0;
        if(1){
            if(sys.fit_op[0]==1){
                gsl_vector_set(ss,dim,sys.BH.M*1e-5);
                dim+=1;
            }
            if(sys.fit_op[1]==1){
                gsl_vector_set(ss,dim,sys.BH.chi*1e-2);
                dim+=1;
            }
            if(sys.fit_op[2]==1){
                gsl_vector_set(ss,dim,sys.BH.q*1e-2);
                dim+=1;
            }
            if(sys.fit_op[3]==1){
                gsl_vector_set(ss,dim,1e-2);
                dim+=1;
            }
            if(sys.fit_op[4]==1){
                gsl_vector_set(ss,dim,1e-2);
                dim+=1;
            }
            if(sys.fit_op[5]==1){
                gsl_vector_set(ss,dim,sys.psr.oe[0]*1e-5);
                dim+=1;
            }
            if(sys.fit_op[6]==1){
                gsl_vector_set(ss,dim,1e-5);
                dim+=1;
            }
            if(sys.fit_op[7]==1){
                gsl_vector_set(ss,dim,1e-4);
                dim+=1;
            }
            if(sys.fit_op[8]==1){
                gsl_vector_set(ss,dim,1e-4);
                dim+=1;
            }
            if(sys.fit_op[9]==1){
                gsl_vector_set(ss,dim,1e-3);
                dim+=1;
            }
            if(sys.fit_op[10]==1){
                gsl_vector_set(ss,dim,1e-4);
                dim+=1;
            }
        }

        struct fitting_no_abb_GR_para params={n,toa,N,sigma_TOA,sys};
        gsl_multimin_function func;
        func.n=1;
        func.f=fitting_no_abb_GR_func;
        func.params=&params;

        gsl_multimin_fminimizer_set(s,&func,x,ss);
        do
        {
            iter++;
            status = gsl_multimin_fminimizer_iterate(s);
            if (status)
                break;
            size = gsl_multimin_fminimizer_size (s);
            status = gsl_multimin_test_size (size, 1e-13);
        }
        while (status == GSL_CONTINUE && iter < 10000);

        // printf ("%2d f() = %.16g size = %.3f\n",
        //             i,
        //             s->fval, size);

        //做完之后要赋值
        if(1){
            dim=0;
            if(sys.fit_op[0]==1){
                sys.BH.M=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[1]==1){
                sys.BH.chi=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[2]==1){
                sys.BH.q=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[3]==1){
                sys.BH.lambda=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[4]==1){
                sys.BH.eta=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[5]==1){
                sys.psr.oe[0]=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[6]==1){
                sys.psr.oe[1]=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[7]==1){
                sys.psr.oe[2]=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[8]==1){
                sys.psr.oe[3]=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[9]==1){
                sys.psr.oe[4]=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[10]==1){
                sys.psr.oe[5]=gsl_vector_get(s->x,dim);
                dim+=1;
            }
        }

        gsl_vector_free(x);
        gsl_vector_free(ss);
        gsl_multimin_fminimizer_free (s);
    }
    //然后进行一轮完整的

    if(1){
        high_iter++;
        for(int j=0;j<11;j++){
            sys.fit_op[j]=1;
        }
        sys.fit_op[8]=0;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        sys.fit_op[2]=0;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        sys.fit_op[1]=0;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        sys.fit_op[3]=0;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        sys.fit_op[4]=0;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        int iter=0;
        double size;
        int status;
        const gsl_multimin_fminimizer_type *T=gsl_multimin_fminimizer_nmsimplex2;
        // gsl_multimin_fminimizer *s=gsl_multimin_fminimizer_alloc(T,13);
        gsl_multimin_fminimizer *s=gsl_multimin_fminimizer_alloc(T,6);//！！！！！！！！！！！！！！！！！！！！！！
        //初始值
        // gsl_vector *x=gsl_vector_alloc(13);
        gsl_vector *x=gsl_vector_alloc(6);//！！！！！！！！！！！！！！！！！！！！！！！！！！
        int dim=0;
        if(1){
            if(sys.fit_op[0]==1){
                gsl_vector_set(x,dim,sys.BH.M);
                dim+=1;
            }
            if(sys.fit_op[1]==1){
                gsl_vector_set(x,dim,sys.BH.chi);
                dim+=1;
            }
            if(sys.fit_op[2]==1){
                gsl_vector_set(x,dim,sys.BH.q);
                dim+=1;
            }
            if(sys.fit_op[3]==1){
                gsl_vector_set(x,dim,sys.BH.lambda);
                dim+=1;
            }
            if(sys.fit_op[4]==1){
                gsl_vector_set(x,dim,sys.BH.eta);
                dim+=1;
            }
            if(sys.fit_op[5]==1){
                gsl_vector_set(x,dim,sys.psr.oe[0]);
                dim+=1;
            }
            if(sys.fit_op[6]==1){
                gsl_vector_set(x,dim,sys.psr.oe[1]);
                dim+=1;
            }
            if(sys.fit_op[7]==1){
                gsl_vector_set(x,dim,sys.psr.oe[2]);
                dim+=1;
            }
            if(sys.fit_op[8]==1){
                gsl_vector_set(x,dim,sys.psr.oe[3]);
                dim+=1;
            }
            if(sys.fit_op[9]==1){
                gsl_vector_set(x,dim,sys.psr.oe[4]);
                dim+=1;
            }
            if(sys.fit_op[10]==1){
                gsl_vector_set(x,dim,sys.psr.oe[5]);
                dim+=1;
            }
        }

        //步长
        // gsl_vector *ss=gsl_vector_alloc(13);
        gsl_vector *ss=gsl_vector_alloc(6);//！！！！！！！！！！！！！！！！！！！！！！！！！！
        dim=0;
        if(1){
            if(sys.fit_op[0]==1){
                gsl_vector_set(ss,dim,sys.BH.M*1e-5);
                dim+=1;
            }
            if(sys.fit_op[1]==1){
                gsl_vector_set(ss,dim,sys.BH.chi*1e-2);
                dim+=1;
            }
            if(sys.fit_op[2]==1){
                gsl_vector_set(ss,dim,sys.BH.q*1e-2);
                dim+=1;
            }
            if(sys.fit_op[3]==1){
                gsl_vector_set(ss,dim,1e-2);
                dim+=1;
            }
            if(sys.fit_op[4]==1){
                gsl_vector_set(ss,dim,1e-2);
                dim+=1;
            }
            if(sys.fit_op[5]==1){
                gsl_vector_set(ss,dim,sys.psr.oe[0]*1e-5);
                dim+=1;
            }
            if(sys.fit_op[6]==1){
                gsl_vector_set(ss,dim,1e-5);
                dim+=1;
            }
            if(sys.fit_op[7]==1){
                gsl_vector_set(ss,dim,1e-4);
                dim+=1;
            }
            if(sys.fit_op[8]==1){
                gsl_vector_set(ss,dim,1e-4);
                dim+=1;
            }
            if(sys.fit_op[9]==1){
                gsl_vector_set(ss,dim,1e-3);
                dim+=1;
            }
            if(sys.fit_op[10]==1){
                gsl_vector_set(ss,dim,1e-4);
                dim+=1;
            }
        }

        struct fitting_no_abb_GR_para params={n,toa,N,sigma_TOA,sys};
        gsl_multimin_function func;
        // func.n=13;
        func.n=6;//！！！！！！！！！！！！！！！！！！！！！！！！！！
        func.f=fitting_no_abb_GR_func;
        func.params=&params;

        gsl_multimin_fminimizer_set(s,&func,x,ss);
        do
        {
            iter++;
            status = gsl_multimin_fminimizer_iterate(s);
            if (status)
                break;
            size = gsl_multimin_fminimizer_size (s);
            status = gsl_multimin_test_size (size, 1e-13);
        }
        while (status == GSL_CONTINUE && iter < 1000);

        printf ("%5d f() = %7.3f size = %.3f\n",
                    high_iter,
                    s->fval, size);

        if(s->fval<chi_last){
            not_increase=0;
            chi_last=s->fval;
        }else{
            not_increase++;
        }
        //做完之后要赋值
        if(1){
            dim=0;
            if(sys.fit_op[0]==1){
                sys.BH.M=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[1]==1){
                sys.BH.chi=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[2]==1){
                sys.BH.q=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[3]==1){
                sys.BH.lambda=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[4]==1){
                sys.BH.eta=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[5]==1){
                sys.psr.oe[0]=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[6]==1){
                sys.psr.oe[1]=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[7]==1){
                sys.psr.oe[2]=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[8]==1){
                sys.psr.oe[3]=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[9]==1){
                sys.psr.oe[4]=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[10]==1){
                sys.psr.oe[5]=gsl_vector_get(s->x,dim);
                dim+=1;
            }
        }

        gsl_vector_free(x);
        gsl_vector_free(ss);
        gsl_multimin_fminimizer_free (s);
    }
    }while(high_iter<100 && not_increase<5);

    star_PSR_rv_ini(&sys.psr,sys.BH,0);
    sys=best_chi2_no_abb_sys(n,toa,N,sigma_TOA,sys);

    printf("chi2_final = %.16g\n",chi2_white(n,toa,N,sigma_TOA,sys));
    fflush(stdout);
    return sys;
}