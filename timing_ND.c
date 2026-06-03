#include "timing.h"
#include "timing_ND.h"

#include <math.h>

#include <gsl/gsl_deriv.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include "const.h"
#include "stars.h"

static double f_full(double x, void *params){
    struct ND_params *p=(struct ND_params *)params;
    struct star_system sys=p->sys;
    double toa=p->toa;
    int op=p->op;

    switch(op){
        case 0:
            sys.BH.M=x;
            star_PSR_rv_ini(&sys.psr,sys.BH,0);
            break;
        case 1:
            sys.BH.chi=x;
            break;
        case 2:
            sys.BH.q=x;
            break;
        case 3:
            sys.BH.lambda=x;
            break;
        case 4:
            sys.BH.eta=x;
            break;
        case 5:
            sys.psr.oe[0]=x;
            star_PSR_rv_ini(&sys.psr,sys.BH,0);
            break;
        case 6:
            sys.psr.oe[1]=x;
            star_PSR_rv_ini(&sys.psr,sys.BH,0);
            break;
        case 7:
            sys.psr.oe[2]=x;
            star_PSR_rv_ini(&sys.psr,sys.BH,0);
            break;
        case 8:
            sys.psr.oe[3]=x;
            star_PSR_rv_ini(&sys.psr,sys.BH,0);
            break;
        case 9:
            sys.psr.oe[4]=x;
            star_PSR_rv_ini(&sys.psr,sys.BH,0);
            break;
        case 10:
            sys.psr.oe[5]=x;
            star_PSR_rv_ini(&sys.psr,sys.BH,0);
            break;
        case 11:
            sys.psr.N0=x;
            break;
        case 12:
            sys.psr.nu=x;
            break;
        case 13:
            sys.psr.dnu=x;
            break;
        case 14:
            sys.psr.d2nu=x;
            break;
        case 15:
            sys.BH.mu_alpha=x;
            break;
        case 16:
            sys.BH.mu_delta=x;
            break;
        case 17:
            sys.psr.lambda_p=x;
            break;
        case 18:
            sys.psr.eta_p=x;
            break;
        case 19:
            sys.psr.d3nu=x;
            break;
        case 20:
            sys.psr.d4nu=x;
            break;
        default:
            break;
    }

    struct toa_output out;
    residual_cal(sys,1,&toa,&out);

    return out.N;
}

gsl_matrix *design_matrix_ND(int n, double toa[], struct star_system sys){
    int dim=0;
    for(int i=0;i<21;i++){
        dim+=(int)(sys.fit_op[i]);
    }
    gsl_matrix *M=gsl_matrix_alloc(n,dim);

    dim=0;
    struct ND_params p;
    p.sys=sys;
    gsl_function F;
    F.function=&f_full;
    F.params=&p;
    double x;
    double h;
    for(int i=0;i<21;i++){
        if(sys.fit_op[i]==1){
            // printf("%d/%d\n",i,dim);
            p.op=i;
            switch(i){
                case 0:
                    x=sys.BH.M;
                    h=sys.BH.M*1e-3;
                    break;
                case 1:
                    x=sys.BH.chi;
                    h=10.0;
                    break;
                case 2:
                    x=sys.BH.q;
                    h=1e6;
                    break;
                case 3:
                    x=sys.BH.lambda;
                    h=1e-2;
                    break;
                case 4:
                    x=sys.BH.eta;
                    h=1e-2;
                    break;
                case 5:
                    x=sys.psr.oe[0];
                    h=sys.psr.oe[0]*1e-3;
                    break;
                case 6:
                    x=sys.psr.oe[1];
                    h=1e-3;
                    break;
                case 7:
                    x=sys.psr.oe[2];
                    h=1e-3;
                    break;
                case 8:
                    x=sys.psr.oe[3];
                    h=1e-1;
                    break;
                case 9:
                    x=sys.psr.oe[4];
                    h=1e-3;
                    break;
                case 10:
                    x=sys.psr.oe[5];
                    h=sys.psr.oe[0]*1e-4;
                    break;
                case 11:
                    x=sys.psr.N0;
                    h=1.0;
                    break;
                case 12:
                    x=sys.psr.nu;
                    h=1e-2/sec;
                    break;
                case 13:
                    x=sys.psr.dnu;
                    h=1e-5/sec/sec;
                    break;
                case 14:
                    x=sys.psr.d2nu;
                    h=1e-8/sec/sec/sec;
                    break;
                case 15:
                    x=sys.BH.mu_alpha;
                    h=1e4*mas/yr;
                    break;
                case 16:
                    x=sys.BH.mu_delta;
                    h=1e4*mas/yr;
                    break;
                case 17:
                    x=sys.psr.lambda_p;
                    h=1e-2;
                    break;
                case 18:
                    x=sys.psr.eta_p;
                    h=1e-2;
                    break;
                case 19:
                    x=sys.psr.d3nu;
                    h=1e-8/sec/sec/sec;
                    break;
                case 20:
                    x=sys.psr.d4nu;
                    h=1e-10/sec/sec/sec/sec;
                    break;
                default:
                    break;
            }
            for(int j=0;j<n;j++){
                p.toa=toa[j];
                int iter_max=30;
                int success=0,iter=0;
                double h_back=h;
                double D1,err1,D2,err2;
                while(success==0 && iter<iter_max){
                    iter++;
                    gsl_deriv_central(&F,x,h*2,&D1,&err1);
                    gsl_deriv_central(&F,x,h,&D2,&err2);
                    // printf("%.3g %.3g %.3g %.3g\n",h,fabs(D2-D1),err1,err2);
                    if(fabs(D2-D1)<1e-13*fabs(D2)){
                        // printf("warning: in design_matrix_ND: are you sure this will happen?\n");
                        success=1;
                    }else if(err2 > 2*err1){
                        //小步长更差，但略宽容
                        h*=2.0;
                    }else if(fabs(D2-D1)>2.0*err2){
                        //误差估计有问题，但不排除是估计的太小  
                        h*=0.5;
                    }else{
                        success=1;
                    }

                }
                if(iter==iter_max){//这是迭代爆了，取消变化并警告
                    // printf("warning: in design_matrix_ND: step can't find\n");
                    h=h_back;
                }

                gsl_deriv_central(&F,x,h,&D1,&err1);
                gsl_matrix_set(M,j,dim,D1/sys.psr.nu);
                // printf("%d/%d %.16g %.16g %.3g\n",j+1,n,D1,err1,fabs(err1/D1));
        
            }
            dim++;
        
        }
    }

    return M;
}

gsl_matrix *Fisher_white_from_design_ND(int n, double toa[], double sigma_TOA[], struct star_system sys){
    int dim=0;
    for(int i=0;i<21;i++){
        dim+=(int)(sys.fit_op[i]);
    }
    gsl_matrix *M=design_matrix_ND(n,toa,sys);
    gsl_matrix *fisher=gsl_matrix_alloc(dim,dim);
    //先处理一下M按j把sigma_j给除了
    for(int j=0;j<n;j++){
        for(int i=0;i<dim;i++){
            gsl_matrix_set(M,j,i,gsl_matrix_get(M,j,i)/sigma_TOA[j]);
        }
    }

    gsl_blas_dsyrk(CblasUpper,CblasTrans,1.0,M,0.0,fisher);
    for(int i=0;i<dim;i++){
        for(int j=0;j<i;j++){
            gsl_matrix_set(fisher,i,j,gsl_matrix_get(fisher,j,i));
        }
    }

    gsl_matrix_free(M);
    return fisher;
}