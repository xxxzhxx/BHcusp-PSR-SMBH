#include "fitting.h"

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

/*sys作为猜测值,并且包含哪些参数需要fit的信息*/
/*由于计时模型的复杂性,我们希望使用不需要提供导数的最小化方法*/
/*params的结构需要重新写*/
double fitting_white_func(const gsl_vector *v,void *params){
    struct fitting_white_params *p=(struct fitting_white_params *)params;

    struct SMBH SgrA;
    struct PSR psr;

    /*vector v中应该按顺序保存了需要拟合的参数的值*/
    /*不需要拟合的参数则应该使用params中给的参考值*/
    int dim=0;
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu
    if(p->sys.fit_op[0]==1){
        SgrA.M=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        SgrA.M=p->sys.BH.M;
    }
    if(p->sys.fit_op[1]==1){
        SgrA.chi=gsl_vector_get(v,dim);
        // if(SgrA.chi>1.0){
        //     SgrA.chi=1.0;
        // }else if(SgrA.chi<-1.0){
        //     SgrA.chi=-1.0;
        // }
        dim+=1;
    }else{
        SgrA.chi=p->sys.BH.chi;
    }
    if(p->sys.fit_op[2]==1){
        SgrA.q=gsl_vector_get(v,dim);
        // if(SgrA.q>1.0){
        //     SgrA.q=1.0;
        // }else if(SgrA.q<-1.0){
        //     SgrA.q=-1.0;
        // }
        dim+=1;
    }else{
        SgrA.q=p->sys.BH.q;
    }
    if(p->sys.fit_op[3]==1){
        SgrA.lambda=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        SgrA.lambda=p->sys.BH.lambda;
    }
    if(p->sys.fit_op[4]==1){
        SgrA.eta=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        SgrA.eta=p->sys.BH.eta;
    }
    if(p->sys.fit_op[5]==1){
        psr.oe[0]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[0]=p->sys.psr.oe[0];
    }
    if(p->sys.fit_op[6]==1){
        psr.oe[1]=gsl_vector_get(v,dim);
        if(psr.oe[1]>1-5e-2){
            psr.oe[1]=1-5e-2;
        }else if(psr.oe[1]<5e-2){
            psr.oe[1]=5e-2;
        }
        dim+=1;
    }else{
        psr.oe[1]=p->sys.psr.oe[1];
    }
    if(p->sys.fit_op[7]==1){
        psr.oe[2]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[2]=p->sys.psr.oe[2];
    }
    if(p->sys.fit_op[8]==1){
        psr.oe[3]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[3]=p->sys.psr.oe[3];
    }
    if(p->sys.fit_op[9]==1){
        psr.oe[4]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[4]=p->sys.psr.oe[4];
    }
    if(p->sys.fit_op[10]==1){
        psr.oe[5]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[5]=p->sys.psr.oe[5];
    }
    if(p->sys.fit_op[11]==1){
        psr.N0=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.N0=p->sys.psr.N0;
    }
    if(p->sys.fit_op[12]==1){
        psr.nu=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.nu=p->sys.psr.nu;
    }
    if(p->sys.fit_op[13]==1){
        psr.dnu=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.dnu=p->sys.psr.dnu;
    }
    if(p->sys.fit_op[14]==1){
        psr.d2nu=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.d2nu=p->sys.psr.d2nu;
    }

    //需要根据oe生成rv
    star_PSR_rv_ini(&psr,SgrA,1);
    psr.op.Newton=1;
    psr.op.Test_1PN=1;
    psr.op.Test_2PN=1;
    psr.op.Test_S=1;
    psr.op.Test_Q=1;
    psr.op.No_hair=1;//这个会忽略自旋输入
    psr.op.Delta_S_2PN=1;
    psr.op.Delta_FD=1;
    psr.op.Delta_E_2PN=1;
    psr.op.proper_motion=0;
    psr.op.Delta_A1=0;
    psr.op.Delta_A2=0;
    psr.op.Delta_L=0;
    psr.op.IMBH=0;
    psr.op.star_cluster=0;

    //制作好system
    struct star_system sys;
    sys.BH=SgrA;
    sys.psr=psr;
    for(int i=0;i<28;i++){
        sys.fit_op[i]=p->sys.fit_op[i];
    }
    sys.BH.mu_alpha=0.0;
    sys.BH.mu_delta=0.0;
    sys.psr.d3nu=0.0;
    sys.psr.d4nu=0.0;
    sys.psr.eta_p=0.0;
    sys.psr.lambda_p=0.0;
    sys.IM.m=0;
    sys.IM.oe[0]=0.3*yr;
    sys.IM.oe[1]=0.3;
    sys.IM.oe[2]=1.0;
    sys.IM.oe[3]=2.0;
    sys.IM.oe[4]=0.5;
    sys.IM.oe[5]=0.0;

    double chi2=chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys);
    // if(1){
    //     printf("chi2 = %g\n",chi2);
    //     printf("M = %g M_sun, chi = %g, q = %g, lambda = %g, eta = %g\n"\
    //     "Pb = %g yr, e = %g, omega = %g pi, Omega = %g pi, inc = %g pi, T0 = %g Pb\n"\
    //     "N0 = %g, nu = %g /sec, dnu = %g /sec/sec, d2nu = %g /sec/sec/sec\n",\
    //     sys.BH.M/(double)M_sun,sys.BH.chi,sys.BH.q,sys.BH.lambda,sys.BH.eta,\
    //     sys.psr.oe[0]/(double)yr,sys.psr.oe[1],sys.psr.oe[2]/(double)pi,sys.psr.oe[3]/(double)pi,sys.psr.oe[4]/(double)pi,\
    //     sys.psr.oe[5]/sys.psr.oe[0],sys.psr.N0,sys.psr.nu*(double)sec,sys.psr.dnu*(double)sec*(double)sec,\
    //     sys.psr.d2nu*(double)sec*(double)sec*(double)sec);
    // }
    // printf("?: %.16g\n",chi2);
    return chi2;
}
/*这个搜索实在有些太低效了,特别是我们在远处已经把它强行拉起来(使用真实N而非整数N)的情况下*/
/*这个不真实的构造我认为在噪声远小于脉冲星自转周期的情况下还是可以接受的?即不产生N的混淆*/
/*我们可能仍然应该采用数值计算雅可比矩阵的方法来作为加速*/
/*使用雅可比矩阵就甚至可能使用trust_region方案了*/
/*由于我们在搜寻最小值,因此即使雅可比矩阵的计算是不精确的也并无关系,这很好*/
struct star_system fitting_white_nmsimplex2rand(int n, double toa[], double N[], double sigma_TOA[], struct star_system sys){
    int dim=0;
    for(int i=0;i<15;i++){
        dim+=(int)(sys.fit_op[i]);
    }

    //初始值
    //还是需要写一个超级判断....
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu
    gsl_vector *x=gsl_vector_alloc(dim);
    dim=0;
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
    if(sys.fit_op[11]==1){
        gsl_vector_set(x,dim,sys.psr.N0);
        dim+=1;
    }
    if(sys.fit_op[12]==1){
        gsl_vector_set(x,dim,sys.psr.nu);
        dim+=1;
    }
    if(sys.fit_op[13]==1){
        gsl_vector_set(x,dim,sys.psr.dnu);
        dim+=1;
    }
    if(sys.fit_op[14]==1){
        gsl_vector_set(x,dim,sys.psr.d2nu);
        dim+=1;
    }

    //需要设定step_size,这个有点麻烦...
    //同样需要超级判断,同时对每个量应该给合理的估计
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu
    gsl_vector *ss=gsl_vector_alloc(dim);
    dim=0;
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
    if(sys.fit_op[11]==1){
        gsl_vector_set(ss,dim,1e-8);
        dim+=1;
    }
    if(sys.fit_op[12]==1){
        gsl_vector_set(ss,dim,sys.psr.nu*1e-8);
        dim+=1;
    }
    if(sys.fit_op[13]==1){
        gsl_vector_set(ss,dim,1e-15/sec/sec*1e-4);
        dim+=1;
    }
    if(sys.fit_op[14]==1){
        gsl_vector_set(ss,dim,1e-24/sec/sec/sec*1e-2);
        dim+=1;
    }

    //让我们给初始值一个随机扰动...
    //这个事情还是暂时先不要在这里做了
    // gsl_rng *r=gsl_rng_alloc(gsl_rng_default);
    // gsl_rng_set(r,(unsigned long)time(NULL));
    // for(int i=0;i<dim;i++){
    //     gsl_vector_set(x,i,gsl_vector_get(x,i)+gsl_ran_gaussian(r,1.0)*gsl_vector_get(ss,i));
    // }
    double chi2_ini=chi2_white(n,toa,N,sigma_TOA,sys);
printf("chi2_ini = %g\n",chi2_ini);

    struct fitting_white_params params={n,toa,N,sigma_TOA,sys};

    gsl_multimin_function func;
    func.n=dim;
    func.f=fitting_white_func;
    func.params=&params;

    gsl_multimin_fminimizer *s=gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2rand,dim);
    gsl_multimin_fminimizer_set(s,&func,x,ss);

    int iter=0;
    int status;
    double size;
    int higher_iter=0;
    double f1=0,f2=0;
    double higher_f1,higher_f2;
    do{
        // printf("iter %d\n",iter);
        iter++;
        status=gsl_multimin_fminimizer_iterate(s);

        if(status){
            break;
        }

        size=gsl_multimin_fminimizer_size(s);
        status=gsl_multimin_test_size(size,1e-2);

        // if(status==GSL_SUCCESS){
        //     printf("converged to minimum at\n");
        // }

        // printf("%5d f()=%.16g size=%g\n",iter,s->fval,size);
        f1=f2;
        f2=s->fval;
    }while(status == GSL_CONTINUE && (iter < 1000 || (f1-f2)/f2>1e-3));
    higher_iter++;
    higher_f2=s->fval;
//     printf("%5d f()=%.16g size=%g\n",higher_iter,s->fval,size);
// printf("dM/M = %g, dchi/chi = %g, dq/q = %g, dlambda/lambda = %g, deta/eta = %g,\ndPb/Pb = %g, "\
// "de/e = %g, domega/omega = %g, dinc/inc = %g, dT0/T0 = %g,\n"\
// "dN0/N0 = %g, dnu/nu = %g, ddnu/dnu = %g\n",\
// (gsl_vector_get(s->x,0)-sys.BH.M)/sys.BH.M,\
// (gsl_vector_get(s->x,1)-sys.BH.chi)/sys.BH.chi,\
// (gsl_vector_get(s->x,2)-sys.BH.q)/sys.BH.q,\
// (gsl_vector_get(s->x,3)-sys.BH.lambda)/sys.BH.lambda,\
// (gsl_vector_get(s->x,4)-sys.BH.eta)/sys.BH.eta,\
// (gsl_vector_get(s->x,5)-sys.psr.oe[0])/sys.psr.oe[0],\
// (gsl_vector_get(s->x,6)-sys.psr.oe[1])/sys.psr.oe[1],\
// (gsl_vector_get(s->x,7)-sys.psr.oe[2])/sys.psr.oe[2],\
// (gsl_vector_get(s->x,8)-sys.psr.oe[4])/sys.psr.oe[4],\
// (gsl_vector_get(s->x,9)-sys.psr.oe[5])/sys.psr.oe[5],\
// (gsl_vector_get(s->x,10)-sys.psr.N0)/sys.psr.N0,\
// (gsl_vector_get(s->x,11)-sys.psr.nu)/sys.psr.nu,\
// (gsl_vector_get(s->x,12)-sys.psr.dnu)/sys.psr.dnu);
    int n_ext=0;
    while(higher_iter <100){
    // printf("?\n");
        higher_iter++;
        higher_f1=higher_f2;
        gsl_multimin_fminimizer_set(s,&func,s->x,ss);
        iter=0;
        do{
            iter++;
            status=gsl_multimin_fminimizer_iterate(s);

            if(status){
                break;
            }

            size=gsl_multimin_fminimizer_size(s);
            status=gsl_multimin_test_size(size,1e-2);

            // if(status==GSL_SUCCESS){
            //     printf("converged to minimum at\n");
            // }

            // printf("%5d f()=%.16g size=%g\n",iter,s->fval,size);
            f1=f2;
            f2=s->fval;
        }while(status == GSL_CONTINUE && (iter < 1000 || (f1-f2)/f2>1e-3));
        printf("%5d f()=%.16g size=%g\n",higher_iter,s->fval,size);
// printf("dM/M = %g, dchi/chi = %g, dq/q = %g, dlambda/lambda = %g, deta/eta = %g,\ndPb/Pb = %g, "\
// "de/e = %g, domega/omega = %g, dinc/inc = %g, dT0/T0 = %g,\n"\
// "dN0/N0 = %g, dnu/nu = %g, ddnu/dnu = %g\n",\
// (gsl_vector_get(s->x,0)-sys.BH.M)/sys.BH.M,\
// (gsl_vector_get(s->x,1)-sys.BH.chi)/sys.BH.chi,\
// (gsl_vector_get(s->x,2)-sys.BH.q)/sys.BH.q,\
// (gsl_vector_get(s->x,3)-sys.BH.lambda)/sys.BH.lambda,\
// (gsl_vector_get(s->x,4)-sys.BH.eta)/sys.BH.eta,\
// (gsl_vector_get(s->x,5)-sys.psr.oe[0])/sys.psr.oe[0],\
// (gsl_vector_get(s->x,6)-sys.psr.oe[1])/sys.psr.oe[1],\
// (gsl_vector_get(s->x,7)-sys.psr.oe[2])/sys.psr.oe[2],\
// (gsl_vector_get(s->x,8)-sys.psr.oe[4])/sys.psr.oe[4],\
// (gsl_vector_get(s->x,9)-sys.psr.oe[5])/sys.psr.oe[5],\
// (gsl_vector_get(s->x,10)-sys.psr.N0)/sys.psr.N0,\
// (gsl_vector_get(s->x,11)-sys.psr.nu)/sys.psr.nu,\
// (gsl_vector_get(s->x,12)-sys.psr.dnu)/sys.psr.dnu);
        higher_f2=s->fval;
        if((higher_f1-higher_f2)/higher_f2<1e-3 && n_ext<5){
            for(int i=0;i<dim;i++){
                gsl_vector_set(ss,i,gsl_vector_get(ss,i)*1.2);
            }
            n_ext++;
        }else if(n_ext>0){
            for(int i=0;i<dim;i++){
                gsl_vector_set(ss,i,gsl_vector_get(ss,i)*0.9);
            }
            n_ext--;
        }
        gsl_multimin_fminimizer_set(s,&func,s->x,ss);
    }

    struct star_system sys_fit;
    struct SMBH SgrA;
    struct PSR psr;

    dim=0;
    if(sys.fit_op[0]==1){
        SgrA.M=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        SgrA.M=sys.BH.M;
    }
    if(sys.fit_op[1]==1){
        SgrA.chi=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        SgrA.chi=sys.BH.chi;
    }
    if(sys.fit_op[2]==1){
        SgrA.q=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        SgrA.q=sys.BH.q;
    }
    if(sys.fit_op[3]==1){
        SgrA.lambda=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        SgrA.lambda=sys.BH.lambda;
    }
    if(sys.fit_op[4]==1){
        SgrA.eta=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        SgrA.eta=sys.BH.eta;
    }
    if(sys.fit_op[5]==1){
        psr.oe[0]=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.oe[0]=sys.psr.oe[0];
    }
    if(sys.fit_op[6]==1){
        psr.oe[1]=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.oe[1]=sys.psr.oe[1];
    }
    if(sys.fit_op[7]==1){
        psr.oe[2]=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.oe[2]=sys.psr.oe[2];
    }
    if(sys.fit_op[8]==1){
        psr.oe[3]=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.oe[3]=sys.psr.oe[3];
    }
    if(sys.fit_op[9]==1){
        psr.oe[4]=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.oe[4]=sys.psr.oe[4];
    }
    if(sys.fit_op[10]==1){
        psr.oe[5]=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.oe[5]=sys.psr.oe[5];
    }
    if(sys.fit_op[11]==1){
        psr.N0=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.N0=sys.psr.N0;
    }
    if(sys.fit_op[12]==1){
        psr.nu=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.nu=sys.psr.nu;
    }
    if(sys.fit_op[13]==1){
        psr.dnu=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.dnu=sys.psr.dnu;
    }
    if(sys.fit_op[14]==1){
        psr.d2nu=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.d2nu=sys.psr.d2nu;
    }

    star_PSR_rv_ini(&psr,SgrA,1);
    psr.op.Newton=1;
    psr.op.Test_1PN=1;
    psr.op.Test_2PN=1;
    psr.op.Test_S=1;
    psr.op.Test_Q=1;

    //制作好system
    sys_fit.BH=SgrA;
    sys_fit.psr=psr;
    for(int i=0;i<15;i++){
        sys_fit.fit_op[i]=sys.fit_op[i];
    }

    gsl_vector_free(x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free(s);

    return sys_fit;
}

//大失败，这一坨
//完全没用的感觉。。。。
//按说使用导数的方法不应该这么失败？？问题出在哪里呢？
//是否是我应该按步长rescale系统？让它尽可能导数比较适中？
void fitting_white_dfunc(const gsl_vector *v, void *params, gsl_vector *df){
    struct fitting_white_params *p=(struct fitting_white_params *)params;

    struct SMBH SgrA;
    struct PSR psr;

    /*vector v中应该按顺序保存了需要拟合的参数的值*/
    /*不需要拟合的参数则应该使用params中给的参考值*/
    int dim=0;
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu
    if(p->sys.fit_op[0]==1){
        SgrA.M=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        SgrA.M=p->sys.BH.M;
    }
    if(p->sys.fit_op[1]==1){
        SgrA.chi=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        SgrA.chi=p->sys.BH.chi;
    }
    if(p->sys.fit_op[2]==1){
        SgrA.q=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        SgrA.q=p->sys.BH.q;
    }
    if(p->sys.fit_op[3]==1){
        SgrA.lambda=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        SgrA.lambda=p->sys.BH.lambda;
    }
    if(p->sys.fit_op[4]==1){
        SgrA.eta=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        SgrA.eta=p->sys.BH.eta;
    }
    if(p->sys.fit_op[5]==1){
        psr.oe[0]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[0]=p->sys.psr.oe[0];
    }
    if(p->sys.fit_op[6]==1){
        psr.oe[1]=gsl_vector_get(v,dim);
        if(psr.oe[1]>1-1e-6){
            psr.oe[1]=1-1e-6;
        }else if(psr.oe[1]<1e-6){
            psr.oe[1]=1e-6;
        }
        dim+=1;
    }else{
        psr.oe[1]=p->sys.psr.oe[1];
    }
    if(p->sys.fit_op[7]==1){
        psr.oe[2]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[2]=p->sys.psr.oe[2];
    }
    if(p->sys.fit_op[8]==1){
        psr.oe[3]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[3]=p->sys.psr.oe[3];
    }
    if(p->sys.fit_op[9]==1){
        psr.oe[4]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[4]=p->sys.psr.oe[4];
    }
    if(p->sys.fit_op[10]==1){
        psr.oe[5]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[5]=p->sys.psr.oe[5];
    }
    if(p->sys.fit_op[11]==1){
        psr.N0=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.N0=p->sys.psr.N0;
    }
    if(p->sys.fit_op[12]==1){
        psr.nu=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.nu=p->sys.psr.nu;
    }
    if(p->sys.fit_op[13]==1){
        psr.dnu=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.dnu=p->sys.psr.dnu;
    }
    if(p->sys.fit_op[14]==1){
        psr.d2nu=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.d2nu=p->sys.psr.d2nu;
    }

    //需要根据oe生成rv
    star_PSR_rv_ini(&psr,SgrA,1);
    psr.op.Newton=1;
    psr.op.Test_1PN=1;
    psr.op.Test_2PN=1;
    psr.op.Test_S=1;
    psr.op.Test_Q=1;

    //制作好system
    struct star_system sys;
    sys.BH=SgrA;
    sys.psr=psr;
    for(int i=0;i<15;i++){
        sys.fit_op[i]=p->sys.fit_op[i];
    }

    dim=0;
    if(sys.fit_op[0]==1){
        struct star_system sys1,sys2;
        double index=12;
        double dM=sys.BH.M*pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.BH.M+=dM;
        sys2.BH.M-=dM;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dM;
        //计算index+1和-1的情况
        dM=sys.BH.M*pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.M+=dM;
        sys2.BH.M-=dM;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dM;
        dM=sys.BH.M*pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.M+=dM;
        sys2.BH.M-=dM;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dM;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dM=sys.BH.M*pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.M+=dM;
            sys2.BH.M-=dM;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dM;
            dM=sys.BH.M*pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.M+=dM;
            sys2.BH.M-=dM;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dM;
        }
        // printf("index for M is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[1]==1){
        struct star_system sys1,sys2;
        double index=9.0;
        double dchi=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.BH.chi+=dchi;
        sys2.BH.chi-=dchi;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dchi;
        //计算index+1和-1的情况
        dchi=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.chi+=dchi;
        sys2.BH.chi-=dchi;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dchi;
        dchi=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.chi+=dchi;
        sys2.BH.chi-=dchi;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dchi;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dchi=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.chi+=dchi;
            sys2.BH.chi-=dchi;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dchi;
            dchi=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.chi+=dchi;
            sys2.BH.chi-=dchi;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dchi;
        }
        // printf("index for chi is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[2]==1){
        struct star_system sys1,sys2;
        double index=8.0;
        double dq=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.BH.q+=dq;
        sys2.BH.q-=dq;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        //计算index+1和-1的情况
        dq=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.q+=dq;
        sys2.BH.q-=dq;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        dq=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.q+=dq;
        sys2.BH.q-=dq;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dq=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.q+=dq;
            sys2.BH.q-=dq;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
            dq=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.q+=dq;
            sys2.BH.q-=dq;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        }
        // printf("index for q is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[3]==1){
        struct star_system sys1,sys2;
        double index=5.0;
        double dlambda=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.BH.lambda+=dlambda;
        sys2.BH.lambda-=dlambda;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dlambda;
        //计算index+1和-1的情况
        dlambda=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.lambda+=dlambda;
        sys2.BH.lambda-=dlambda;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dlambda;
        dlambda=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.lambda+=dlambda;
        sys2.BH.lambda-=dlambda;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dlambda;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dlambda=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.lambda+=dlambda;
            sys2.BH.lambda-=dlambda;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dlambda;
            dlambda=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.lambda+=dlambda;
            sys2.BH.lambda-=dlambda;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dlambda;
        }
        // printf("index for lambda is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[4]==1){
        struct star_system sys1,sys2;
        double index=8.0;
        double deta=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.BH.eta+=deta;
        sys2.BH.eta-=deta;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/deta;
        //计算index+1和-1的情况
        deta=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.eta+=deta;
        sys2.BH.eta-=deta;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/deta;
        deta=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.eta+=deta;
        sys2.BH.eta-=deta;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/deta;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            deta=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.eta+=deta;
            sys2.BH.eta-=deta;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/deta;
            deta=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.eta+=deta;
            sys2.BH.eta-=deta;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/deta;
        }
        // printf("index for eta is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[5]==1){
        struct star_system sys1,sys2;
        double index=9;
        double dPb=sys.psr.oe[0]*pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[0]+=dPb;
        sys2.psr.oe[0]-=dPb;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        //计算index+1和-1的情况
        dPb=sys.psr.oe[0]*pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[0]+=dPb;
        sys2.psr.oe[0]-=dPb;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        dPb=sys.psr.oe[0]*pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[0]+=dPb;
        sys2.psr.oe[0]-=dPb;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dPb=sys.psr.oe[0]*pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[0]+=dPb;
            sys2.psr.oe[0]-=dPb;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
            dPb=sys.psr.oe[0]*pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[0]+=dPb;
            sys2.psr.oe[0]-=dPb;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        }
        // printf("index for Pb is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[6]==1){
        struct star_system sys1,sys2;
        double index=8;
        double de=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[1]+=de;
        sys2.psr.oe[1]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        //计算index+1和-1的情况
        de=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[1]+=de;
        sys2.psr.oe[1]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        de=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[1]+=de;
        sys2.psr.oe[1]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            de=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[1]+=de;
            sys2.psr.oe[1]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
            de=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[1]+=de;
            sys2.psr.oe[1]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        }
        // printf("index for e is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[7]==1){
        struct star_system sys1,sys2;
        double index=11;
        double de=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[2]+=de;
        sys2.psr.oe[2]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        //计算index+1和-1的情况
        de=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[2]+=de;
        sys2.psr.oe[2]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        de=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[2]+=de;
        sys2.psr.oe[2]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            de=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[2]+=de;
            sys2.psr.oe[2]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
            de=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[2]+=de;
            sys2.psr.oe[2]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        }
        // printf("index for omega is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[8]==1){
        struct star_system sys1,sys2;
        double index=8;
        double de=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[3]+=de;
        sys2.psr.oe[3]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        //计算index+1和-1的情况
        de=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[3]+=de;
        sys2.psr.oe[3]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        de=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[3]+=de;
        sys2.psr.oe[3]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            de=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[3]+=de;
            sys2.psr.oe[3]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
            de=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[3]+=de;
            sys2.psr.oe[3]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        }
        // printf("index for Omega is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[9]==1){
        struct star_system sys1,sys2;
        double index=6;
        double de=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[4]+=de;
        sys2.psr.oe[4]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        //计算index+1和-1的情况
        de=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[4]+=de;
        sys2.psr.oe[4]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        de=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[4]+=de;
        sys2.psr.oe[4]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            de=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[4]+=de;
            sys2.psr.oe[4]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
            de=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[4]+=de;
            sys2.psr.oe[4]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        }
        // printf("index for inc is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[10]==1){
        struct star_system sys1,sys2;
        double index=11;
        double dPb=sys.psr.oe[0]*pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[5]+=dPb;
        sys2.psr.oe[5]-=dPb;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        //计算index+1和-1的情况
        dPb=sys.psr.oe[0]*pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[5]+=dPb;
        sys2.psr.oe[5]-=dPb;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        dPb=sys.psr.oe[0]*pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[5]+=dPb;
        sys2.psr.oe[5]-=dPb;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dPb=sys.psr.oe[0]*pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[5]+=dPb;
            sys2.psr.oe[5]-=dPb;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
            dPb=sys.psr.oe[0]*pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[5]+=dPb;
            sys2.psr.oe[5]-=dPb;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        }
        // printf("index for T0 is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[11]==1){
        struct star_system sys1,sys2;
        double index=4;
        double dq=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.N0+=dq;
        sys2.psr.N0-=dq;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        //计算index+1和-1的情况
        dq=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.N0+=dq;
        sys2.psr.N0-=dq;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        dq=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.N0+=dq;
        sys2.psr.N0-=dq;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dq=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.N0+=dq;
            sys2.psr.N0-=dq;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
            dq=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.N0+=dq;
            sys2.psr.N0-=dq;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        }
        // printf("index for N0 is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[12]==1){
        struct star_system sys1,sys2;
        double index=4;
        double dq=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.nu+=dq;
        sys2.psr.nu-=dq;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        //计算index+1和-1的情况
        dq=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.nu+=dq;
        sys2.psr.nu-=dq;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        dq=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.nu+=dq;
        sys2.psr.nu-=dq;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dq=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.nu+=dq;
            sys2.psr.nu-=dq;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
            dq=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.nu+=dq;
            sys2.psr.nu-=dq;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        }
        // printf("index for nu is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[13]==1){
        struct star_system sys1,sys2;
        double index=5;
        double dq=1e-15/sec/sec*pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.dnu+=dq;
        sys2.psr.dnu-=dq;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        //计算index+1和-1的情况
        dq=1e-15/sec/sec*pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.dnu+=dq;
        sys2.psr.dnu-=dq;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        dq=1e-15/sec/sec*pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.dnu+=dq;
        sys2.psr.dnu-=dq;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dq=1e-15/sec/sec*pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.dnu+=dq;
            sys2.psr.dnu-=dq;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
            dq=1e-15/sec/sec*pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.dnu+=dq;
            sys2.psr.dnu-=dq;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        }
        // printf("index for dnu is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[14]==1){
        struct star_system sys1=sys,sys2=sys;
        double dd2nu=1e-24/sec/sec/sec*1e-2;
        sys1.psr.d2nu+=dd2nu;
        sys2.psr.d2nu-=dd2nu;
        gsl_vector_set(df,dim,(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dd2nu);
        dim+=1;
    }

    for(int i=0;i<dim;i++){
        printf("%g ",gsl_vector_get(df,i));
    }
    printf("\n");
}
void fitting_white_fdfunc(const gsl_vector *v, void *params, double *f,gsl_vector *df){
    struct fitting_white_params *p=(struct fitting_white_params *)params;

    struct SMBH SgrA;
    struct PSR psr;

    /*vector v中应该按顺序保存了需要拟合的参数的值*/
    /*不需要拟合的参数则应该使用params中给的参考值*/
    int dim=0;
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu
    if(p->sys.fit_op[0]==1){
        SgrA.M=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        SgrA.M=p->sys.BH.M;
    }
    if(p->sys.fit_op[1]==1){
        SgrA.chi=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        SgrA.chi=p->sys.BH.chi;
    }
    if(p->sys.fit_op[2]==1){
        SgrA.q=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        SgrA.q=p->sys.BH.q;
    }
    if(p->sys.fit_op[3]==1){
        SgrA.lambda=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        SgrA.lambda=p->sys.BH.lambda;
    }
    if(p->sys.fit_op[4]==1){
        SgrA.eta=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        SgrA.eta=p->sys.BH.eta;
    }
    if(p->sys.fit_op[5]==1){
        psr.oe[0]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[0]=p->sys.psr.oe[0];
    }
    if(p->sys.fit_op[6]==1){
        psr.oe[1]=gsl_vector_get(v,dim);
        if(psr.oe[1]>1-1e-6){
            psr.oe[1]=1-1e-6;
        }else if(psr.oe[1]<1e-6){
            psr.oe[1]=1e-6;
        }
        dim+=1;
    }else{
        psr.oe[1]=p->sys.psr.oe[1];
    }
    if(p->sys.fit_op[7]==1){
        psr.oe[2]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[2]=p->sys.psr.oe[2];
    }
    if(p->sys.fit_op[8]==1){
        psr.oe[3]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[3]=p->sys.psr.oe[3];
    }
    if(p->sys.fit_op[9]==1){
        psr.oe[4]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[4]=p->sys.psr.oe[4];
    }
    if(p->sys.fit_op[10]==1){
        psr.oe[5]=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.oe[5]=p->sys.psr.oe[5];
    }
    if(p->sys.fit_op[11]==1){
        psr.N0=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.N0=p->sys.psr.N0;
    }
    if(p->sys.fit_op[12]==1){
        psr.nu=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.nu=p->sys.psr.nu;
    }
    if(p->sys.fit_op[13]==1){
        psr.dnu=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.dnu=p->sys.psr.dnu;
    }
    if(p->sys.fit_op[14]==1){
        psr.d2nu=gsl_vector_get(v,dim);
        dim+=1;
    }else{
        psr.d2nu=p->sys.psr.d2nu;
    }

    //需要根据oe生成rv
    star_PSR_rv_ini(&psr,SgrA,1);
    psr.op.Newton=1;
    psr.op.Test_1PN=1;
    psr.op.Test_2PN=1;
    psr.op.Test_S=1;
    psr.op.Test_Q=1;

    //制作好system
    struct star_system sys;
    sys.BH=SgrA;
    sys.psr=psr;
    for(int i=0;i<15;i++){
        sys.fit_op[i]=p->sys.fit_op[i];
    }

    *f=chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys);

    dim=0;
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu
    /*oe[6]={Pb,e,omega,Omega,inc,T0}*/
    if(sys.fit_op[0]==1){
        struct star_system sys1,sys2;
        double index=12;
        double dM=sys.BH.M*pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.BH.M+=dM;
        sys2.BH.M-=dM;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dM;
        //计算index+1和-1的情况
        dM=sys.BH.M*pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.M+=dM;
        sys2.BH.M-=dM;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dM;
        dM=sys.BH.M*pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.M+=dM;
        sys2.BH.M-=dM;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dM;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dM=sys.BH.M*pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.M+=dM;
            sys2.BH.M-=dM;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dM;
            dM=sys.BH.M*pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.M+=dM;
            sys2.BH.M-=dM;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dM;
        }
        // printf("index for M is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[1]==1){
        struct star_system sys1,sys2;
        double index=9.0;
        double dchi=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.BH.chi+=dchi;
        sys2.BH.chi-=dchi;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dchi;
        //计算index+1和-1的情况
        dchi=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.chi+=dchi;
        sys2.BH.chi-=dchi;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dchi;
        dchi=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.chi+=dchi;
        sys2.BH.chi-=dchi;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dchi;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dchi=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.chi+=dchi;
            sys2.BH.chi-=dchi;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dchi;
            dchi=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.chi+=dchi;
            sys2.BH.chi-=dchi;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dchi;
        }
        // printf("index for chi is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[2]==1){
        struct star_system sys1,sys2;
        double index=8.0;
        double dq=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.BH.q+=dq;
        sys2.BH.q-=dq;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        //计算index+1和-1的情况
        dq=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.q+=dq;
        sys2.BH.q-=dq;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        dq=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.q+=dq;
        sys2.BH.q-=dq;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dq=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.q+=dq;
            sys2.BH.q-=dq;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
            dq=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.q+=dq;
            sys2.BH.q-=dq;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        }
        // printf("index for q is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[3]==1){
        struct star_system sys1,sys2;
        double index=5.0;
        double dlambda=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.BH.lambda+=dlambda;
        sys2.BH.lambda-=dlambda;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dlambda;
        //计算index+1和-1的情况
        dlambda=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.lambda+=dlambda;
        sys2.BH.lambda-=dlambda;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dlambda;
        dlambda=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.lambda+=dlambda;
        sys2.BH.lambda-=dlambda;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dlambda;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dlambda=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.lambda+=dlambda;
            sys2.BH.lambda-=dlambda;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dlambda;
            dlambda=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.lambda+=dlambda;
            sys2.BH.lambda-=dlambda;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dlambda;
        }
        // printf("index for lambda is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[4]==1){
        struct star_system sys1,sys2;
        double index=8.0;
        double deta=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.BH.eta+=deta;
        sys2.BH.eta-=deta;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/deta;
        //计算index+1和-1的情况
        deta=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.eta+=deta;
        sys2.BH.eta-=deta;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/deta;
        deta=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.BH.eta+=deta;
        sys2.BH.eta-=deta;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/deta;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            deta=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.eta+=deta;
            sys2.BH.eta-=deta;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/deta;
            deta=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.BH.eta+=deta;
            sys2.BH.eta-=deta;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/deta;
        }
        // printf("index for eta is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[5]==1){
        struct star_system sys1,sys2;
        double index=9;
        double dPb=sys.psr.oe[0]*pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[0]+=dPb;
        sys2.psr.oe[0]-=dPb;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        //计算index+1和-1的情况
        dPb=sys.psr.oe[0]*pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[0]+=dPb;
        sys2.psr.oe[0]-=dPb;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        dPb=sys.psr.oe[0]*pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[0]+=dPb;
        sys2.psr.oe[0]-=dPb;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dPb=sys.psr.oe[0]*pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[0]+=dPb;
            sys2.psr.oe[0]-=dPb;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
            dPb=sys.psr.oe[0]*pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[0]+=dPb;
            sys2.psr.oe[0]-=dPb;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        }
        // printf("index for Pb is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[6]==1){
        struct star_system sys1,sys2;
        double index=8;
        double de=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[1]+=de;
        sys2.psr.oe[1]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        //计算index+1和-1的情况
        de=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[1]+=de;
        sys2.psr.oe[1]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        de=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[1]+=de;
        sys2.psr.oe[1]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            de=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[1]+=de;
            sys2.psr.oe[1]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
            de=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[1]+=de;
            sys2.psr.oe[1]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        }
        // printf("index for e is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[7]==1){
        struct star_system sys1,sys2;
        double index=11;
        double de=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[2]+=de;
        sys2.psr.oe[2]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        //计算index+1和-1的情况
        de=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[2]+=de;
        sys2.psr.oe[2]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        de=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[2]+=de;
        sys2.psr.oe[2]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            de=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[2]+=de;
            sys2.psr.oe[2]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
            de=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[2]+=de;
            sys2.psr.oe[2]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        }
        // printf("index for omega is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[8]==1){
        struct star_system sys1,sys2;
        double index=8;
        double de=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[3]+=de;
        sys2.psr.oe[3]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        //计算index+1和-1的情况
        de=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[3]+=de;
        sys2.psr.oe[3]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        de=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[3]+=de;
        sys2.psr.oe[3]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            de=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[3]+=de;
            sys2.psr.oe[3]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
            de=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[3]+=de;
            sys2.psr.oe[3]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        }
        // printf("index for Omega is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[9]==1){
        struct star_system sys1,sys2;
        double index=6;
        double de=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[4]+=de;
        sys2.psr.oe[4]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        //计算index+1和-1的情况
        de=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[4]+=de;
        sys2.psr.oe[4]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        de=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[4]+=de;
        sys2.psr.oe[4]-=de;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            de=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[4]+=de;
            sys2.psr.oe[4]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
            de=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[4]+=de;
            sys2.psr.oe[4]-=de;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/de;
        }
        // printf("index for inc is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[10]==1){
        struct star_system sys1,sys2;
        double index=11;
        double dPb=sys.psr.oe[0]*pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[5]+=dPb;
        sys2.psr.oe[5]-=dPb;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        //计算index+1和-1的情况
        dPb=sys.psr.oe[0]*pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[5]+=dPb;
        sys2.psr.oe[5]-=dPb;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        dPb=sys.psr.oe[0]*pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.oe[5]+=dPb;
        sys2.psr.oe[5]-=dPb;
        star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
        star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dPb=sys.psr.oe[0]*pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[5]+=dPb;
            sys2.psr.oe[5]-=dPb;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
            dPb=sys.psr.oe[0]*pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.oe[5]+=dPb;
            sys2.psr.oe[5]-=dPb;
            star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
            star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dPb;
        }
        // printf("index for T0 is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[11]==1){
        struct star_system sys1,sys2;
        double index=4;
        double dq=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.N0+=dq;
        sys2.psr.N0-=dq;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        //计算index+1和-1的情况
        dq=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.N0+=dq;
        sys2.psr.N0-=dq;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        dq=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.N0+=dq;
        sys2.psr.N0-=dq;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dq=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.N0+=dq;
            sys2.psr.N0-=dq;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
            dq=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.N0+=dq;
            sys2.psr.N0-=dq;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        }
        // printf("index for N0 is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[12]==1){
        struct star_system sys1,sys2;
        double index=4;
        double dq=pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.nu+=dq;
        sys2.psr.nu-=dq;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        //计算index+1和-1的情况
        dq=pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.nu+=dq;
        sys2.psr.nu-=dq;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        dq=pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.nu+=dq;
        sys2.psr.nu-=dq;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dq=pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.nu+=dq;
            sys2.psr.nu-=dq;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
            dq=pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.nu+=dq;
            sys2.psr.nu-=dq;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        }
        // printf("index for nu is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[13]==1){
        struct star_system sys1,sys2;
        double index=5;
        double dq=1e-15/sec/sec*pow(10.0,-index);
        sys1=sys;
        sys2=sys;
        sys1.psr.dnu+=dq;
        sys2.psr.dnu-=dq;
        double der=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        //计算index+1和-1的情况
        dq=1e-15/sec/sec*pow(10.0,-(index+1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.dnu+=dq;
        sys2.psr.dnu-=dq;
        double der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        dq=1e-15/sec/sec*pow(10.0,-(index-1.0));
        sys1=sys;
        sys2=sys;
        sys1.psr.dnu+=dq;
        sys2.psr.dnu-=dq;
        double der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        while(fabs(der1)<fabs(der) || fabs(der2)<fabs(der)){
            if(fabs(der1)<fabs(der)){
                index+=1.0;
                der=der1;
            }else{
                index-=1.0;
                der=der2;
            }
            dq=1e-15/sec/sec*pow(10.0,-(index+1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.dnu+=dq;
            sys2.psr.dnu-=dq;
            der1=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
            dq=1e-15/sec/sec*pow(10.0,-(index-1.0));
            sys1=sys;
            sys2=sys;
            sys1.psr.dnu+=dq;
            sys2.psr.dnu-=dq;
            der2=(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dq;
        }
        // printf("index for dnu is %g, der = %g\n",index,der);
        gsl_vector_set(df,dim,der);
        dim+=1;
    }
    if(sys.fit_op[14]==1){
        struct star_system sys1=sys,sys2=sys;
        double dd2nu=1e-24/sec/sec/sec*1e-2;
        sys1.psr.d2nu+=dd2nu;
        sys2.psr.d2nu-=dd2nu;
        gsl_vector_set(df,dim,(chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys1)-chi2_white(p->n,p->toa,p->N,p->sigma_TOA,sys2))/2.0/dd2nu);
        dim+=1;
    }
for(int i=0;i<dim;i++){
        printf("%g ",gsl_vector_get(df,i));
    }
    printf("\n");
}
struct star_system fitting_white_conjugate_fr(int n, double toa[], double N[], double sigma_TOA[], struct star_system sys){
    int dim=0;
    for(int i=0;i<15;i++){
        dim+=(int)(sys.fit_op[i]);
    }

    //初始值
    //还是需要写一个超级判断....
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu
    gsl_vector *x=gsl_vector_alloc(dim);
    dim=0;
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
    if(sys.fit_op[11]==1){
        gsl_vector_set(x,dim,sys.psr.N0);
        dim+=1;
    }
    if(sys.fit_op[12]==1){
        gsl_vector_set(x,dim,sys.psr.nu);
        dim+=1;
    }
    if(sys.fit_op[13]==1){
        gsl_vector_set(x,dim,sys.psr.dnu);
        dim+=1;
    }
    if(sys.fit_op[14]==1){
        gsl_vector_set(x,dim,sys.psr.d2nu);
        dim+=1;
    }

    //需要设定step_size,这个有点麻烦...
    //同样需要超级判断,同时对每个量应该给合理的估计
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu
    gsl_vector *ss=gsl_vector_alloc(dim);
    dim=0;
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
        gsl_vector_set(ss,dim,sys.psr.oe[1]*1e-5);
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
        gsl_vector_set(ss,dim,sys.psr.oe[0]*1e-4);
        dim+=1;
    }
    if(sys.fit_op[11]==1){
        gsl_vector_set(ss,dim,1e-8);
        dim+=1;
    }
    if(sys.fit_op[12]==1){
        gsl_vector_set(ss,dim,sys.psr.nu*1e-8);
        dim+=1;
    }
    if(sys.fit_op[13]==1){
        gsl_vector_set(ss,dim,1e-15/sec/sec*1e-4);
        dim+=1;
    }
    if(sys.fit_op[14]==1){
        gsl_vector_set(ss,dim,1e-24/sec/sec/sec*1e-2);
        dim+=1;
    }

    //让我们给初始值一个随机扰动...
    //这个事情还是暂时先不要在这里做了
    gsl_rng *r=gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(r,(unsigned long)time(NULL));
    for(int i=0;i<dim;i++){
        gsl_vector_set(x,i,gsl_vector_get(x,i)+gsl_ran_gaussian(r,1.0)*gsl_vector_get(ss,i));
    }


    struct fitting_white_params params={n,toa,N,sigma_TOA,sys};

    gsl_multimin_fdfminimizer *s=gsl_multimin_fdfminimizer_alloc(gsl_multimin_fdfminimizer_conjugate_fr,dim);

    gsl_multimin_function_fdf fdf_func;
    fdf_func.n=dim;
    fdf_func.f=fitting_white_func;
    fdf_func.df=fitting_white_dfunc;
    fdf_func.fdf=fitting_white_fdfunc;
    fdf_func.params=&params;

    gsl_multimin_fdfminimizer_set(s,&fdf_func,x,1e-2,0.1);
    int iter=0;
    int status;
    do{
        iter++;
        status=gsl_multimin_fdfminimizer_iterate(s);
        if(status){
            printf("what's wrong?\n");
            break;
        }
        status=gsl_multimin_test_gradient(s->gradient,1e-3);
        if(status==GSL_SUCCESS){
            printf("Minimum found at:\n");
        }
        printf ("iter = %5d, f = %g\n", iter,s->f);
    }while(status == GSL_CONTINUE && iter < 100);

    struct star_system sys_fit;
    struct SMBH SgrA;
    struct PSR psr;

    dim=0;
    if(sys.fit_op[0]==1){
        SgrA.M=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        SgrA.M=sys.BH.M;
    }
    if(sys.fit_op[1]==1){
        SgrA.chi=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        SgrA.chi=sys.BH.chi;
    }
    if(sys.fit_op[2]==1){
        SgrA.q=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        SgrA.q=sys.BH.q;
    }
    if(sys.fit_op[3]==1){
        SgrA.lambda=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        SgrA.lambda=sys.BH.lambda;
    }
    if(sys.fit_op[4]==1){
        SgrA.eta=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        SgrA.eta=sys.BH.eta;
    }
    if(sys.fit_op[5]==1){
        psr.oe[0]=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.oe[0]=sys.psr.oe[0];
    }
    if(sys.fit_op[6]==1){
        psr.oe[1]=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.oe[1]=sys.psr.oe[1];
    }
    if(sys.fit_op[7]==1){
        psr.oe[2]=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.oe[2]=sys.psr.oe[2];
    }
    if(sys.fit_op[8]==1){
        psr.oe[3]=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.oe[3]=sys.psr.oe[3];
    }
    if(sys.fit_op[9]==1){
        psr.oe[4]=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.oe[4]=sys.psr.oe[4];
    }
    if(sys.fit_op[10]==1){
        psr.oe[5]=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.oe[5]=sys.psr.oe[5];
    }
    if(sys.fit_op[11]==1){
        psr.N0=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.N0=sys.psr.N0;
    }
    if(sys.fit_op[12]==1){
        psr.nu=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.nu=sys.psr.nu;
    }
    if(sys.fit_op[13]==1){
        psr.dnu=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.dnu=sys.psr.dnu;
    }
    if(sys.fit_op[14]==1){
        psr.d2nu=gsl_vector_get(s->x,dim);
        dim+=1;
    }else{
        psr.d2nu=sys.psr.d2nu;
    }

    star_PSR_rv_ini(&psr,SgrA,1);
    psr.op.Newton=1;
    psr.op.Test_1PN=1;
    psr.op.Test_2PN=1;
    psr.op.Test_S=1;
    psr.op.Test_Q=1;

    //制作好system
    sys_fit.BH=SgrA;
    sys_fit.psr=psr;
    for(int i=0;i<15;i++){
        sys_fit.fit_op[i]=sys.fit_op[i];
    }

    gsl_multimin_fdfminimizer_free (s);
    gsl_vector_free (x);

    return sys_fit;
}

/*考虑使用design matrix简化来进行参数拟合*/
gsl_matrix *fitting_white_design_matrix(int n, double toa[], int n_real, double noise[n][n_real], double sigma_TOA[], struct star_system sys){
    //首先根据sigma_toa进行一个缩放
    gsl_matrix *V=gsl_matrix_calloc(n,n);
    for(int i=0;i<n;i++){
        gsl_matrix_set(V,i,i,1.0/sigma_TOA[i]/sigma_TOA[i]);
    }
    //因为在给一个相同sigma_TOA的情况下V是对角的，这就完全和没有是一样的了

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

    //计算一个乘法，感觉问题可能出在这里
    gsl_matrix *factor=gsl_matrix_alloc(dim,n);
    gsl_blas_dgemm(CblasNoTrans,CblasTrans,-1.0,MVM,VM,0.0,factor);
// for(int i=0;i<n;i++){
//     for(int j=0;j<dim;j++){
//         printf("%g ",gsl_matrix_get(factor,j,i));
//     }
//     printf("\n");
// }

    //由最小二乘，可以直接写出参数的结果：dxi=-(M^T V M)^-1 M^T V dt
    gsl_matrix *dt=gsl_matrix_alloc(n,n_real);
    for(int i=0;i<n;i++){
        for(int j=0;j<n_real;j++){
            gsl_matrix_set(dt,i,j,noise[i][j]);
        }
    }

    gsl_matrix *dxi=gsl_matrix_alloc(dim,n_real);
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,factor,dt,0.0,dxi);

    gsl_matrix_free(V);
    gsl_matrix_free(M);
    gsl_matrix_free(VM);
    gsl_matrix_free(dt);
    gsl_matrix_free(factor);

    return dxi;
}



//以下是我们特化写的一个拟合函数，
//注意！！！！！！！！我们改了下，现在不fit M chi q lambda eta,记得改回来！！！！！
struct star_system fitting_MSQ(int n, double toa[], double N[], double sigma_TOA[], struct star_system sys){
    //默认拟合参数为M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu
    //编号为。     0,1. ,2,3.    ,4. ,5 ,6,7    ,8.   ,9  ,10,11,12,13
    //会采用：单参数最小化+全参数最小化循环
    double chi2_ini=chi2_white(n,toa,N,sigma_TOA,sys);
    printf("chi2_ini = %g\n",chi2_ini);

    int high_iter=0,not_increase=0;
    double chi_last=chi2_ini;

    do{
    //我们从单参数开始，首先对不包括Omega的所有参数进行一轮
    for(int i=0;i<14;i++){
    // for(int i=5;i<14;i++){//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if(i==8 || i==2){ //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
            if(sys.fit_op[11]==1){
                gsl_vector_set(x,dim,sys.psr.N0);
                dim+=1;
            }
            if(sys.fit_op[12]==1){
                gsl_vector_set(x,dim,sys.psr.nu);
                dim+=1;
            }
            if(sys.fit_op[13]==1){
                gsl_vector_set(x,dim,sys.psr.dnu);
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
            if(sys.fit_op[11]==1){
                gsl_vector_set(ss,dim,1e-8);
                dim+=1;
            }
            if(sys.fit_op[12]==1){
                gsl_vector_set(ss,dim,sys.psr.nu*1e-8);
                dim+=1;
            }
            if(sys.fit_op[13]==1){
                gsl_vector_set(ss,dim,1e-15/sec/sec*1e-4);
                dim+=1;
            }
        }

        struct fitting_white_params params={n,toa,N,sigma_TOA,sys};
        gsl_multimin_function func;
        func.n=1;
        func.f=fitting_white_func;
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
            // if (status == GSL_SUCCESS)
            //     {
            //     printf ("converged to minimum at\n");
            //     }
            // printf ("%5d f() = %7.3f size = %.3f\n",
            //         iter,
            //         s->fval, size);
        }
        while (status == GSL_CONTINUE && iter < 10000);

        printf ("%2d f() = %7.3f size = %.3f\n",
                    i,
                    s->fval, size);

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
            if(sys.fit_op[11]==1){
                sys.psr.N0=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[12]==1){
                sys.psr.nu=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[13]==1){
                sys.psr.dnu=gsl_vector_get(s->x,dim);
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
        for(int j=0;j<14;j++){
        // for(int j=5;j<14;j++){//！！！！！！！！！！！！！！！！！！！！
            sys.fit_op[j]=1;
        }
        sys.fit_op[8]=0;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        sys.fit_op[2]=0;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        int iter=0;
        double size;
        int status;
        const gsl_multimin_fminimizer_type *T=gsl_multimin_fminimizer_nmsimplex2;
        // gsl_multimin_fminimizer *s=gsl_multimin_fminimizer_alloc(T,13);
        gsl_multimin_fminimizer *s=gsl_multimin_fminimizer_alloc(T,12);//！！！！！！！！！！！！！！！！！！！！！！
        //初始值
        // gsl_vector *x=gsl_vector_alloc(13);
        gsl_vector *x=gsl_vector_alloc(12);//！！！！！！！！！！！！！！！！！！！！！！！！！！
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
            if(sys.fit_op[11]==1){
                gsl_vector_set(x,dim,sys.psr.N0);
                dim+=1;
            }
            if(sys.fit_op[12]==1){
                gsl_vector_set(x,dim,sys.psr.nu);
                dim+=1;
            }
            if(sys.fit_op[13]==1){
                gsl_vector_set(x,dim,sys.psr.dnu);
                dim+=1;
            }
        }

        //步长
        // gsl_vector *ss=gsl_vector_alloc(13);
        gsl_vector *ss=gsl_vector_alloc(12);//！！！！！！！！！！！！！！！！！！！！！！！！！！
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
            if(sys.fit_op[11]==1){
                gsl_vector_set(ss,dim,1e-8);
                dim+=1;
            }
            if(sys.fit_op[12]==1){
                gsl_vector_set(ss,dim,sys.psr.nu*1e-8);
                dim+=1;
            }
            if(sys.fit_op[13]==1){
                gsl_vector_set(ss,dim,1e-15/sec/sec*1e-4);
                dim+=1;
            }
        }

        struct fitting_white_params params={n,toa,N,sigma_TOA,sys};
        gsl_multimin_function func;
        // func.n=13;
        func.n=12;//！！！！！！！！！！！！！！！！！！！！！！！！！！
        func.f=fitting_white_func;
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
            // if (status == GSL_SUCCESS)
            //     {
            //     printf ("converged to minimum at\n");
            //     }
            // printf ("%5d f() = %7.3f size = %.3f\n",
            //         iter,
            //         s->fval, size);
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
            if(sys.fit_op[11]==1){
                sys.psr.N0=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[12]==1){
                sys.psr.nu=gsl_vector_get(s->x,dim);
                dim+=1;
            }
            if(sys.fit_op[13]==1){
                sys.psr.dnu=gsl_vector_get(s->x,dim);
                dim+=1;
            }
        }

        gsl_vector_free(x);
        gsl_vector_free(ss);
        gsl_multimin_fminimizer_free (s);
    }
    }while(high_iter<100 && not_increase<5);

    printf("chi2_final = %.16g\n",chi2_white(n,toa,N,sigma_TOA,sys));
    return sys;
}

static double chi2_disconnect(int n_peri, int n, double toa[n_peri][n], double N[n_peri][n], double sigma_TOA[n_peri][n], struct star_system sys_list[n_peri]){
    double chi2=0;
    for(int i=0;i<n_peri;i++){
        struct toa_output out[n];
        struct star_system sys=sys_list[i];
        star_PSR_rv_ini(&sys.psr,sys.BH,0);
        residual_cal(sys,n,toa[i],out);
        for(int j=0;j<n;j++){
            chi2+=((out[j].N-N[i][j])/sys.psr.nu/sigma_TOA[i][j])*((out[j].N-N[i][j])/sys.psr.nu/sigma_TOA[i][j]);
        }
    }
    return chi2;
}

struct fitting_disconnect_para{
    int n_peri;
    int n;
    //这两个还得拉直。。。
    double *toa;
    double *N;
    double *sigma_TOA;
    struct star_system *sys_list;
    int i_peri;
    int i_para;
};

static double fitting_disconnect_func_psr_single(const gsl_vector *v,void *params){
    //v里面保存了一个变量，所以其他的交给para
    struct fitting_disconnect_para *p=(struct fitting_disconnect_para *)params;
    int n_peri=p->n_peri,n=p->n;
    double toa[n_peri][n],N[n_peri][n],sigma_TOA[n_peri][n];
    struct star_system sys_list[n_peri];
    for(int i=0;i<n_peri;i++){
        for(int j=0;j<n;j++){
            toa[i][j]=p->toa[n*i+j];
            N[i][j]=p->N[n*i+j];
            sigma_TOA[i][j]=p->sigma_TOA[n*i+j];
        }
        sys_list[i]=p->sys_list[i];
    }

    //然后是改动v
    switch(p->i_para){
        case 5:
            sys_list[p->i_peri].psr.oe[0]=gsl_vector_get(v,0);
            break;
        case 6:
            sys_list[p->i_peri].psr.oe[1]=gsl_vector_get(v,0);
            break;
        case 7:
            sys_list[p->i_peri].psr.oe[2]=gsl_vector_get(v,0);
            break;
        case 8:
            sys_list[p->i_peri].psr.oe[3]=gsl_vector_get(v,0);
            break;
        case 9:
            sys_list[p->i_peri].psr.oe[4]=gsl_vector_get(v,0);
            break;
        case 10:
            sys_list[p->i_peri].psr.oe[5]=gsl_vector_get(v,0);
            break;
        case 11:
            sys_list[p->i_peri].psr.N0=gsl_vector_get(v,0);
            break;
        case 12:
            sys_list[p->i_peri].psr.nu=gsl_vector_get(v,0);
            break;
        case 13:
            sys_list[p->i_peri].psr.dnu=gsl_vector_get(v,0);
            break;
        default: break;
    }

    double chi2=chi2_disconnect(n_peri,n,toa,N,sigma_TOA,sys_list);
    return chi2;
}

static double fitting_disconnect_func_psr_full(const gsl_vector *v, void *params){
    struct fitting_disconnect_para *p=(struct fitting_disconnect_para *)params;
    int n_peri=p->n_peri,n=p->n;
    double toa[n_peri][n],N[n_peri][n],sigma_TOA[n_peri][n];
    struct star_system sys_list[n_peri];
    for(int i=0;i<n_peri;i++){
        for(int j=0;j<n;j++){
            toa[i][j]=p->toa[n*i+j];
            N[i][j]=p->N[n*i+j];
            sigma_TOA[i][j]=p->sigma_TOA[n*i+j];
        }
        sys_list[i]=p->sys_list[i];
    }

    int dim=0,i_peri=p->i_peri;
    if(sys_list[i_peri].fit_op[5]==1){
        sys_list[i_peri].psr.oe[0]=gsl_vector_get(v,dim);
        dim++;
    }
    if(sys_list[i_peri].fit_op[6]==1){
        sys_list[i_peri].psr.oe[1]=gsl_vector_get(v,dim);
        dim++;
    }
    if(sys_list[i_peri].fit_op[7]==1){
        sys_list[i_peri].psr.oe[2]=gsl_vector_get(v,dim);
        dim++;
    }
    if(sys_list[i_peri].fit_op[8]==1){
        sys_list[i_peri].psr.oe[3]=gsl_vector_get(v,dim);
        dim++;
    }
    if(sys_list[i_peri].fit_op[9]==1){
        sys_list[i_peri].psr.oe[4]=gsl_vector_get(v,dim);
        dim++;
    }
    if(sys_list[i_peri].fit_op[10]==1){
        sys_list[i_peri].psr.oe[5]=gsl_vector_get(v,dim);
        dim++;
    }
    if(sys_list[i_peri].fit_op[11]==1){
        sys_list[i_peri].psr.N0=gsl_vector_get(v,dim);
        dim++;
    }
    if(sys_list[i_peri].fit_op[12]==1){
        sys_list[i_peri].psr.nu=gsl_vector_get(v,dim);
        dim++;
    }
    if(sys_list[i_peri].fit_op[13]==1){
        sys_list[i_peri].psr.dnu=gsl_vector_get(v,dim);
        dim++;
    }

    double chi2=chi2_disconnect(n_peri,n,toa,N,sigma_TOA,sys_list);
    return chi2;
}

static double fitting_disconnect_func_BH_single(const gsl_vector *v, void *params){
    struct fitting_disconnect_para *p=(struct fitting_disconnect_para *)params;
    int n_peri=p->n_peri,n=p->n;
    double toa[n_peri][n],N[n_peri][n],sigma_TOA[n_peri][n];
    struct star_system sys_list[n_peri];
    for(int i=0;i<n_peri;i++){
        for(int j=0;j<n;j++){
            toa[i][j]=p->toa[n*i+j];
            N[i][j]=p->N[n*i+j];
            sigma_TOA[i][j]=p->sigma_TOA[n*i+j];
        }
        sys_list[i]=p->sys_list[i];
    }

    switch(p->i_peri){
        case 0:
            for(int i=0;i<n_peri;i++){
                sys_list[i].BH.M=gsl_vector_get(v,0);
            }
            break;
        case 1:
            for(int i=0;i<n_peri;i++){
                sys_list[i].BH.chi=gsl_vector_get(v,0);
            }
            break;
        case 2:
            for(int i=0;i<n_peri;i++){
                sys_list[i].BH.q=gsl_vector_get(v,0);
            }
            break;
        case 3:
            for(int i=0;i<n_peri;i++){
                sys_list[i].BH.lambda=gsl_vector_get(v,0);
            }
            break;
        case 4:
            for(int i=0;i<n_peri;i++){
                sys_list[i].BH.eta=gsl_vector_get(v,0);
            }
            break;
        default: break;
    }

    double chi2=chi2_disconnect(n_peri,n,toa,N,sigma_TOA,sys_list);
    return chi2;
}

static double fitting_disconnect_func_BH_full(const gsl_vector *v, void *params){
    struct fitting_disconnect_para *p=(struct fitting_disconnect_para *)params;
    int n_peri=p->n_peri,n=p->n;
    double toa[n_peri][n],N[n_peri][n],sigma_TOA[n_peri][n];
    struct star_system sys_list[n_peri];
    for(int i=0;i<n_peri;i++){
        for(int j=0;j<n;j++){
            toa[i][j]=p->toa[n*i+j];
            N[i][j]=p->N[n*i+j];
            sigma_TOA[i][j]=p->sigma_TOA[n*i+j];
        }
        sys_list[i]=p->sys_list[i];
    }

    for(int i=0;i<n_peri;i++){
        sys_list[i].BH.M=gsl_vector_get(v,0);
        sys_list[i].BH.chi=gsl_vector_get(v,1);
        sys_list[i].BH.q=gsl_vector_get(v,2);
        sys_list[i].BH.lambda=gsl_vector_get(v,3);
        sys_list[i].BH.eta=gsl_vector_get(v,4);
    }

    double chi2=chi2_disconnect(n_peri,n,toa,N,sigma_TOA,sys_list);
    return chi2;
}

struct fitting_disconnect_out fitting_disconnect(int n_peri,int n,double toa[n_peri][n], double N[n_peri][n],double sigma_TOA[n_peri][n], struct star_system sys, double N_fit[n_peri][n]){
    struct star_system sys_list[n_peri];
    for(int i=0;i<n_peri;i++){
        sys_list[i]=sys;
    }
    //这个是单纯connect的残差/可以理解为一开始全一致的残差
    double chi2_ini=chi2_disconnect(n_peri,n,toa,N,sigma_TOA,sys_list);
    printf("chi2_ini = %.16g\n",chi2_ini);

    double toa_list[n_peri*n],N_list[n_peri*n],sigma_TOA_list[n_peri*n];
    for(int i=0;i<n_peri;i++){
        for(int j=0;j<n;j++){
            toa_list[n*i+j]=toa[i][j];
            N_list[n*i+j]=N[i][j];
            sigma_TOA_list[n*i+j]=sigma_TOA[i][j];
        }
    }

    double chi2_last=chi2_ini;
    //我们就不进行全参数拟合了，但是进行一下循环
    //首先对每个部分进行拟合，这些拟合是互相独立的，直接写出顺序的即可
    //这些拟合只改变脉冲星参数
    int con=0;
    do{
    for(int i_peri=0;i_peri<n_peri;i_peri++){
        //这个拟合可以看作do_while小循环，循环到找到极值为止，但不用特别严格
        int high_iter=0;
        int not_increase=0;
        do{
            //我们从单参数开始，
            //不包括黑洞参数
            //除了第一个阶段，其余也拟合Omega?
            for(int i=5;i<14;i++){
                if(i==8 && i_peri==0){
                    continue;
                }
                //首先我们就把sys_list[i_peri]的op调到正确位置
                for(int j=0;j<28;j++){
                    sys_list[i_peri].fit_op[j]=0;
                }
                sys_list[i_peri].fit_op[i]=1;

                int iter=0;
                double size;
                int status;
                const gsl_multimin_fminimizer_type *T=gsl_multimin_fminimizer_nmsimplex2;
                gsl_multimin_fminimizer *s=gsl_multimin_fminimizer_alloc(T,1);
                //初始值，注意这个要改
                gsl_vector *x=gsl_vector_alloc(1);
                int dim=0;
                if(1){
                    if(sys_list[i_peri].fit_op[0]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].BH.M);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[1]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].BH.chi);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[2]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].BH.q);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[3]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].BH.lambda);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[4]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].BH.eta);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[5]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.oe[0]);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[6]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.oe[1]);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[7]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.oe[2]);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[8]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.oe[3]);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[9]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.oe[4]);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[10]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.oe[5]);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[11]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.N0);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[12]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.nu);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[13]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.dnu);
                        dim+=1;
                    }
                }

                //步长
                gsl_vector *ss=gsl_vector_alloc(1);
                dim=0;
                if(1){
                    if(sys_list[i_peri].fit_op[0]==1){
                        gsl_vector_set(ss,dim,sys_list[i_peri].BH.M*1e-5);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[1]==1){
                        gsl_vector_set(ss,dim,sys_list[i_peri].BH.chi*1e-2);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[2]==1){
                        gsl_vector_set(ss,dim,sys_list[i_peri].BH.q*1e-2);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[3]==1){
                        gsl_vector_set(ss,dim,1e-2);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[4]==1){
                        gsl_vector_set(ss,dim,1e-2);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[5]==1){
                        gsl_vector_set(ss,dim,sys_list[i_peri].psr.oe[0]*1e-5);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[6]==1){
                        gsl_vector_set(ss,dim,1e-5);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[7]==1){
                        gsl_vector_set(ss,dim,1e-4);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[8]==1){
                        gsl_vector_set(ss,dim,1e-4);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[9]==1){
                        gsl_vector_set(ss,dim,1e-3);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[10]==1){
                        gsl_vector_set(ss,dim,1e-4);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[11]==1){
                        gsl_vector_set(ss,dim,1e-8);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[12]==1){
                        gsl_vector_set(ss,dim,sys_list[i_peri].psr.nu*1e-8);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[13]==1){
                        gsl_vector_set(ss,dim,1e-15/sec/sec*1e-4);
                        dim+=1;
                    }
                }

                struct fitting_disconnect_para params={n_peri,n,toa_list,N_list,sigma_TOA_list,sys_list,i_peri,i};
                gsl_multimin_function func;
                func.n=1;
                func.f=fitting_disconnect_func_psr_single;
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
                    // printf("%d %.16g\n",iter,s->fval);
                }
                while (status == GSL_CONTINUE && iter < 100);

                // printf ("%2d %2d f() = %7.3f size = %.3f\n",i_peri,i,s->fval, size);

                if(1){
                    dim=0;
                    if(sys_list[i_peri].fit_op[0]==1){
                        sys_list[i_peri].BH.M=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[1]==1){
                        sys_list[i_peri].BH.chi=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[2]==1){
                        sys_list[i_peri].BH.q=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[3]==1){
                        sys_list[i_peri].BH.lambda=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[4]==1){
                        sys_list[i_peri].BH.eta=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[5]==1){
                        sys_list[i_peri].psr.oe[0]=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[6]==1){
                        sys_list[i_peri].psr.oe[1]=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[7]==1){
                        sys_list[i_peri].psr.oe[2]=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[8]==1){
                        sys_list[i_peri].psr.oe[3]=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[9]==1){
                        sys_list[i_peri].psr.oe[4]=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[10]==1){
                        sys_list[i_peri].psr.oe[5]=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[11]==1){
                        sys_list[i_peri].psr.N0=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[12]==1){
                        sys_list[i_peri].psr.nu=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[13]==1){
                        sys_list[i_peri].psr.dnu=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                }

                gsl_vector_free(x);
                gsl_vector_free(ss);
                gsl_multimin_fminimizer_free (s);

            }

            //然后是一个多参数拟合
            if(1){
                for(int j=5;j<14;j++){
                    sys_list[i_peri].fit_op[j]=1;
                }
                if(i_peri==0){
                    sys_list[i_peri].fit_op[8]=0;
                }

                int iter=0;
                double size;
                int status;
                const gsl_multimin_fminimizer_type *T=gsl_multimin_fminimizer_nmsimplex2;

                int n_para;
                if(i_peri==0){
                    n_para=8;
                }else{
                    n_para=9;
                }
                gsl_multimin_fminimizer *s=gsl_multimin_fminimizer_alloc(T,n_para);
                gsl_vector *x=gsl_vector_alloc(n_para);
                int dim=0;
                if(1){
                    if(sys_list[i_peri].fit_op[0]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].BH.M);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[1]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].BH.chi);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[2]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].BH.q);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[3]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].BH.lambda);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[4]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].BH.eta);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[5]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.oe[0]);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[6]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.oe[1]);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[7]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.oe[2]);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[8]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.oe[3]);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[9]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.oe[4]);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[10]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.oe[5]);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[11]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.N0);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[12]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.nu);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[13]==1){
                        gsl_vector_set(x,dim,sys_list[i_peri].psr.dnu);
                        dim+=1;
                    }
                }

                //步长
                gsl_vector *ss=gsl_vector_alloc(n_para);
                dim=0;
                if(1){
                    if(sys_list[i_peri].fit_op[0]==1){
                        gsl_vector_set(ss,dim,sys_list[i_peri].BH.M*1e-5);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[1]==1){
                        gsl_vector_set(ss,dim,sys_list[i_peri].BH.chi*1e-2);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[2]==1){
                        gsl_vector_set(ss,dim,sys_list[i_peri].BH.q*1e-2);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[3]==1){
                        gsl_vector_set(ss,dim,1e-2);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[4]==1){
                        gsl_vector_set(ss,dim,1e-2);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[5]==1){
                        gsl_vector_set(ss,dim,sys_list[i_peri].psr.oe[0]*1e-5);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[6]==1){
                        gsl_vector_set(ss,dim,1e-5);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[7]==1){
                        gsl_vector_set(ss,dim,1e-4);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[8]==1){
                        gsl_vector_set(ss,dim,1e-4);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[9]==1){
                        gsl_vector_set(ss,dim,1e-3);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[10]==1){
                        gsl_vector_set(ss,dim,1e-4);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[11]==1){
                        gsl_vector_set(ss,dim,1e-8);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[12]==1){
                        gsl_vector_set(ss,dim,sys_list[i_peri].psr.nu*1e-8);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[13]==1){
                        gsl_vector_set(ss,dim,1e-15/sec/sec*1e-4);
                        dim+=1;
                    }
                }

                struct fitting_disconnect_para params={n_peri,n,toa_list,N_list,sigma_TOA_list,sys_list,i_peri,n_para};
                gsl_multimin_function func;
                func.n=n_para;
                func.f=fitting_disconnect_func_psr_full;
                func.params=&params;

                gsl_multimin_fminimizer_set(s,&func,x,ss);
                do
                {
                    iter++;
                    status = gsl_multimin_fminimizer_iterate(s);
                    if (status)
                        break;
                    size = gsl_multimin_fminimizer_size (s);
                    status = gsl_multimin_test_size (size, 1e-5);
                    // printf("%d %.16g\n",iter,s->fval);
                }
                while (status == GSL_CONTINUE && iter < 1000);
                printf ("%5d f() = %7.3f size = %.3f\n",high_iter,s->fval, size);

                //在free掉之前
                if(s->fval<chi2_last){
                    not_increase=0;
                    chi2_last=s->fval;
                }else{
                    not_increase++;
                }

                //做完之后要赋值
                if(1){
                    dim=0;
                    if(sys_list[i_peri].fit_op[0]==1){
                        sys_list[i_peri].BH.M=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[1]==1){
                        sys_list[i_peri].BH.chi=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[2]==1){
                        sys_list[i_peri].BH.q=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[3]==1){
                        sys_list[i_peri].BH.lambda=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[4]==1){
                        sys_list[i_peri].BH.eta=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[5]==1){
                        sys_list[i_peri].psr.oe[0]=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[6]==1){
                        sys_list[i_peri].psr.oe[1]=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[7]==1){
                        sys_list[i_peri].psr.oe[2]=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[8]==1){
                        sys_list[i_peri].psr.oe[3]=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[9]==1){
                        sys_list[i_peri].psr.oe[4]=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[10]==1){
                        sys_list[i_peri].psr.oe[5]=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[11]==1){
                        sys_list[i_peri].psr.N0=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[12]==1){
                        sys_list[i_peri].psr.nu=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                    if(sys_list[i_peri].fit_op[13]==1){
                        sys_list[i_peri].psr.dnu=gsl_vector_get(s->x,dim);
                        dim+=1;
                    }
                }

                gsl_vector_free(x);
                gsl_vector_free(ss);
                gsl_multimin_fminimizer_free (s);
            }

            high_iter++;
        }while(high_iter<20 && not_increase<2);
    }

    //然后是我们对M,chi,q,lambda,eta这5个参数进行拟合
    //这里记录下拟合前chi2
    double chi2_before_MSQ=chi2_last;
    //同样先单参数，然后全参数
    int high_iter=0;
    int not_decrease=0;
    do{
        for(int i=0;i<5;i++){
            int iter=0;
            double size;
            int status;
            const gsl_multimin_fminimizer_type *T=gsl_multimin_fminimizer_nmsimplex2;
            gsl_multimin_fminimizer *s=gsl_multimin_fminimizer_alloc(T,1);

            gsl_vector *x=gsl_vector_alloc(1);
            switch(i){
                case 0:
                    gsl_vector_set(x,0,sys_list[0].BH.M);
                    break;
                case 1:
                    gsl_vector_set(x,0,sys_list[0].BH.chi);
                    break;
                case 2:
                    gsl_vector_set(x,0,sys_list[0].BH.q);
                    break;
                case 3:
                    gsl_vector_set(x,0,sys_list[0].BH.lambda);
                    break;
                case 4:
                    gsl_vector_set(x,0,sys_list[0].BH.eta);
                    break;
                default: break;
            }

            gsl_vector *ss=gsl_vector_alloc(1);
            switch(i){
                case 0:
                    gsl_vector_set(ss,0,sys.BH.M*1e-5);
                    break;
                case 1:
                    gsl_vector_set(ss,0,sys.BH.chi*1e-2);
                    break;
                case 2:
                    gsl_vector_set(ss,0,sys.BH.q*1e-2);
                    break;
                case 3:
                    gsl_vector_set(ss,0,1e-2);
                    break;
                case 4:
                    gsl_vector_set(ss,0,1e-2);
                    break;  
                default: break;
            }

            struct fitting_disconnect_para params={n_peri,n,toa_list,N_list,sigma_TOA_list,sys_list,i,i};
            gsl_multimin_function func;
            func.n=1;
            func.f=fitting_disconnect_func_BH_single;
            func.params=&params;

            gsl_multimin_fminimizer_set(s,&func,x,ss);
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
            printf ("%2d f() = %7.3f size = %.3f\n",i,s->fval, size);

            //赋值
            switch(i){
                case 0:
                    for(int j=0;j<n_peri;j++){
                        sys_list[j].BH.M=gsl_vector_get(s->x,0);
                    }
                    break;
                case 1:
                    for(int j=0;j<n_peri;j++){
                        sys_list[j].BH.chi=gsl_vector_get(s->x,0);
                    }
                    break;
                case 2:
                    for(int j=0;j<n_peri;j++){
                        sys_list[j].BH.q=gsl_vector_get(s->x,0);
                    }
                    break;
                case 3:
                    for(int j=0;j<n_peri;j++){
                        sys_list[j].BH.lambda=gsl_vector_get(s->x,0);
                    }
                    break;
                case 4:
                    for(int j=0;j<n_peri;j++){
                        sys_list[j].BH.eta=gsl_vector_get(s->x,0);
                    }
                    break;
                default: break;
            }

            gsl_vector_free(x);
            gsl_vector_free(ss);
            gsl_multimin_fminimizer_free (s);
        }

        //然后是完整的
        high_iter++;
        int iter=0;
        double size;
        int status;
        const gsl_multimin_fminimizer_type *T=gsl_multimin_fminimizer_nmsimplex2;
        gsl_multimin_fminimizer *s=gsl_multimin_fminimizer_alloc(T,5);

        //初值
        gsl_vector *x=gsl_vector_alloc(5);
        gsl_vector_set(x,0,sys_list[0].BH.M);
        gsl_vector_set(x,1,sys_list[0].BH.chi);
        gsl_vector_set(x,2,sys_list[0].BH.q);
        gsl_vector_set(x,3,sys_list[0].BH.lambda);
        gsl_vector_set(x,4,sys_list[0].BH.eta);

        //步长
        gsl_vector *ss=gsl_vector_alloc(5);
        gsl_vector_set(ss,0,sys.BH.M*1e-5);
        gsl_vector_set(ss,1,sys.BH.chi*1e-2);
        gsl_vector_set(ss,2,sys.BH.q*1e-2);
        gsl_vector_set(ss,3,1e-2);
        gsl_vector_set(ss,4,1e-2);

        struct fitting_disconnect_para params={n_peri,n,toa_list,N_list,sigma_TOA_list,sys_list,0,0};
        gsl_multimin_function func;
        func.n=5;
        func.f=fitting_disconnect_func_BH_full;
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
        while (status == GSL_CONTINUE && iter < 100);

        printf ("%5d f() = %7.3f size = %.3f\n",high_iter,s->fval, size);

        if(s->fval<chi2_last){
            not_decrease=0;
            chi2_last=s->fval;
        }else{
            not_decrease++;
        }

        //赋值
        for(int i=0;i<n_peri;i++){
            sys_list[i].BH.M=gsl_vector_get(s->x,0);
            sys_list[i].BH.chi=gsl_vector_get(s->x,1);
            sys_list[i].BH.q=gsl_vector_get(s->x,2);
            sys_list[i].BH.lambda=gsl_vector_get(s->x,3);
            sys_list[i].BH.eta=gsl_vector_get(s->x,4);
        }

        gsl_vector_free(x);
        gsl_vector_free(ss);
        gsl_multimin_fminimizer_free (s);
    }while(high_iter<20&&not_decrease<2);

    if(chi2_last<chi2_before_MSQ){
        con==1;
    }else{
        con==0;
    }
    //虽然说残差很小很好了，但没拟合到chi又没那么好（
    //但今天不想干了，套个循环就这样吧
    }while(con==1);


    struct fitting_disconnect_out out;
    out.chi2=chi2_last;
    out.M=sys_list[0].BH.M;
    out.chi=sys_list[0].BH.chi;
    out.q=sys_list[0].BH.q;
    out.lambda=sys_list[0].BH.lambda;
    out.eta=sys_list[0].BH.eta;

    return out;
}