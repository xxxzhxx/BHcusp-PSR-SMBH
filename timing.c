#include "timing.h"

#include <math.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_blas.h>

#include "const.h"
#include "stars.h"
#include "motion.h"
#include "noise.h"

void toa_generator(struct star_system star_sys,int n,double t[],struct toa_output out[]){
    struct mot_output mot_out[n];
    mot_evolve(star_sys,n,t,mot_out);
    
    struct PSR psr=star_sys.psr;
    double N0=psr.N0,nu=psr.nu,dnu=psr.dnu,d2nu=psr.d2nu,d3nu=psr.d3nu,d4nu=psr.d4nu;
    // double lambda_p=psr.lambda_p, eta_p=psr.eta_p+psr.oe[3];
    double lambda_p=psr.lambda_p, eta_p=psr.eta_p;
    struct SMBH BH=star_sys.BH;
    double M=BH.M,chi=BH.chi;
    // double lambda=BH.lambda,eta=BH.eta+psr.oe[3];
    double lambda=BH.lambda,eta=BH.eta;
    double mu_alpha=BH.mu_alpha,mu_delta=BH.mu_delta;

    for(int i=0;i<n;i++){
        double x=mot_out[i].rv[0],y=mot_out[i].rv[1],z=mot_out[i].rv[2];
        double vx=mot_out[i].rv[3],vy=mot_out[i].rv[4],vz=mot_out[i].rv[5];
        double r=sqrt(x*x+y*y+z*z);
        double a=3.0*G*M/2.0/c/c/(1.0-psr.Delta_E_norm);
        double Pb=psr.oe[0];
        // double a=pow(G*M*Pb*Pb/4.0/pi/pi,1.0/3.0);
        double tM=G*M/c/c/c;
        double C=G*M/r/c/c;
        double vke=vy*sin(lambda_p)*cos(eta_p)-vx*sin(lambda_p)*sin(eta_p);
        double ke2=sin(lambda_p)*sin(lambda_p);
        double ve=vx*sin(lambda_p)*cos(eta_p)+vy*sin(lambda_p)*sin(eta_p)+vz*cos(lambda_p);
        double nke=(y*sin(lambda_p)*cos(eta_p)-x*sin(lambda_p)*sin(eta_p))/r;
        double skn=sin(lambda)*sin(eta)*x/r-sin(lambda)*cos(eta)*y/r;

        double Delta_A1=0;
        double Delta_A2=0;
        double Delta_L=0;

        out[i].t=mot_out[i].t;

        if(star_sys.psr.op.Delta_A1==1){
            Delta_A1=-vke/c/ke2/2.0/pi/nu;
        }
        if(star_sys.psr.op.Delta_A2==1){
            Delta_A2=-vke/c/ke2/2.0/pi/nu*(vz/2.0/c+(vz-cos(lambda_p)*ve)/c/ke2);
        }
        if(star_sys.psr.op.Delta_L==1){
            Delta_L=2.0*C/(1.0-z/r)/2.0/pi/nu*nke/ke2;
        }


        out[i].T=mot_out[i].t-mot_out[i].Delta_E-Delta_A1-Delta_A2-Delta_L;
        out[i].N=N0+nu*out[i].T+dnu/2.0*out[i].T*out[i].T+d2nu/6.0*out[i].T*out[i].T*out[i].T+d3nu/24.0*pow(out[i].T,4.0)+d4nu/120.0*pow(out[i].T,5.0);


        out[i].Delta_E=mot_out[i].Delta_E;
        out[i].Delta_A1=Delta_A1;
        out[i].Delta_A2=Delta_A2;
        out[i].Delta_L=Delta_L;

        double pro_t=sqrt(mu_alpha*mu_alpha+mu_delta*mu_delta)*(out[i].t-mot_out[i].t0);
        double Delta_R=z/c;
        double Delta_S_1PN=0;
        double Delta_S_DPN=0;
        double Delta_S_PPN=0;
        double Delta_FD=0;

        if(star_sys.psr.op.proper_motion==1){
            Delta_R=(z*cos(pro_t)+sin(pro_t)*(x*mu_alpha+y*mu_delta)/sqrt(mu_alpha*mu_alpha+mu_delta*mu_delta))/c;
        }
        Delta_S_1PN=-2.0*tM*log((r-z)/a);
        if(star_sys.psr.op.Delta_S_2PN==1){
            Delta_S_DPN=-4.0*tM*C/(1.0-z/r);
            Delta_S_PPN=C*tM*(-z/r/4.0+15.0/4.0*acos(-z/r)/sqrt(1.0-z/r*z/r));
        }
        if(star_sys.psr.op.Delta_FD==1){
            Delta_FD=-chi*2.0*tM*C/(1.0-z/r)*skn;
        }

        out[i].Delta_R=Delta_R;
        out[i].Delta_S_1PN=Delta_S_1PN;
        out[i].Delta_S_DPN=Delta_S_DPN;
        out[i].Delta_S_PPN=Delta_S_PPN;
        out[i].Delta_FD=Delta_FD;

        out[i].toa=mot_out[i].t+out[i].Delta_R+out[i].Delta_S_1PN+out[i].Delta_S_DPN+out[i].Delta_S_PPN+out[i].Delta_FD;
    }
}

int func_inverse_timing_model(double toa, const double rvEt[], double f[], void *params){
    struct star_system *sys=(struct star_system *)params;
    double M=sys->BH.M;

    double t=rvEt[7];
    double rvE[7]={rvEt[0],rvEt[1],rvEt[2],rvEt[3],rvEt[4],rvEt[5],rvEt[6]};
    double drvE_dt[7]; //最后一个t并不加入
    func_mot_evolve(t,rvE,drvE_dt,params);

    double x=rvEt[0],y=rvEt[1],z=rvEt[2],vx=rvEt[3],vy=rvEt[4],vz=rvEt[5];
    double r=sqrt(x*x+y*y+z*z);
    double vr=(x*vx+y*vy+z*vz)/r;
    double tM=G*M/c/c/c;
    double mu_alpha=sys->BH.mu_alpha;
    double mu_delta=sys->BH.mu_delta;
    double mu=sqrt(mu_alpha*mu_alpha+mu_delta*mu_delta);
    double chi=sys->BH.chi;
    // double lambda=sys->BH.lambda,eta=sys->BH.eta+sys->psr.oe[3];
    double lambda=sys->BH.lambda,eta=sys->BH.eta;

    double sKv=-vy*sin(lambda)*cos(eta)+sin(lambda)*sin(eta)*vx;
    double sKn=-y/r*sin(lambda)*cos(eta)+sin(lambda)*sin(eta)*x/r;

    double dDelta_R_dt=vz/c;
    double dDelta_S1_dt=-2.0*tM*(vr-vz)/(r-z);
    double dDelta_DPN_dt=0;
    double dDelta_PPN_dt=0;
    double dDelta_FD_dt=0;

    if(sys->psr.op.proper_motion==1){
        dDelta_R_dt=vz/c*cos(mu*t)+(mu_alpha/mu*vx/c+mu_delta/mu*vy/c)*sin(mu*t)+\
                    (-mu*z/c*sin(mu*t)+(mu_alpha*x/c+mu_delta*y/c)*cos(mu*t));
    }
    if(sys->psr.op.Delta_S_2PN==1){
        dDelta_DPN_dt=4.0*tM*tM*c*(vr-vz)/(r-z)/(r-z);
        dDelta_PPN_dt=tM*tM*c/4.0*(-vz/r/r+2.0*z*vr/r/r/r+15.0*(vz-vr*z/r)/(r*r-z*z)\
                      -15.0*acos(-z/r)*(r*vr-z*vz)/pow(r*r-z*z,1.5));
    }
    if(sys->psr.op.Delta_FD==1){
        dDelta_FD_dt=-chi*2.0*tM*tM*c*((sKv/r-sKn*vr/r)/(r-z)-sKn/(r-z)*(vr-vz)/(r-z));
    }

    double dDelta_S_dt=dDelta_S1_dt+dDelta_DPN_dt+dDelta_PPN_dt;
    double dt_dtoa=1.0/(1.0+dDelta_R_dt+dDelta_S_dt+dDelta_FD_dt);

    f[0]=drvE_dt[0]*dt_dtoa;
    f[1]=drvE_dt[1]*dt_dtoa;
    f[2]=drvE_dt[2]*dt_dtoa;
    f[3]=drvE_dt[3]*dt_dtoa;
    f[4]=drvE_dt[4]*dt_dtoa;
    f[5]=drvE_dt[5]*dt_dtoa;
    f[6]=drvE_dt[6]*dt_dtoa;
    f[7]=dt_dtoa;

    return GSL_SUCCESS;
}
void residual_cal(struct star_system star_sys,int n, double toa[], struct toa_output out[]){
    gsl_odeiv2_system sys={func_inverse_timing_model,NULL,8,&star_sys};

    /*首先要初始化到第一步上,由于此时的方向不一定是正是负,所以需要单独拿出来做*/
    gsl_odeiv2_driver *d0;
    //计算t=0时刻对应的toa
    struct PSR psr=star_sys.psr;

    double N0=psr.N0,nu=psr.nu,dnu=psr.dnu,d2nu=psr.d2nu,d3nu=psr.d3nu,d4nu=psr.d4nu;
    double M=star_sys.BH.M;
    double x=psr.rv[0],y=psr.rv[1],z=psr.rv[2],vx=psr.rv[3],vy=psr.rv[4],vz=psr.rv[5];
    double r=sqrt(x*x+y*y+z*z);
    double vr=(x*vx+y*vy+z*vz)/r;
    double a=3.0*G*M/2.0/c/c/(1.0-psr.Delta_E_norm);
    double tM=G*M/c/c/c;
    double C=G*M/r/c/c;
    double chi=star_sys.BH.chi;
    // double lambda=star_sys.BH.lambda,eta=star_sys.BH.eta+star_sys.psr.oe[3];
    double lambda=star_sys.BH.lambda,eta=star_sys.BH.eta;
    double mu_alpha=star_sys.BH.mu_alpha,mu_delta=star_sys.BH.mu_delta;
    // double lambda_p=star_sys.psr.lambda_p,eta_p=star_sys.psr.eta_p+star_sys.psr.oe[3];
    double lambda_p=star_sys.psr.lambda_p,eta_p=star_sys.psr.eta_p;

    double skn=sin(lambda)*sin(eta)*x/r-sin(lambda)*cos(eta)*y/r;

    double Delta_R=z/c;
    double Delta_S_1PN=-2.0*tM*log((r-z)/a);
    double Delta_S_DPN=-4.0*tM*C/(1.0-z/r);
    double Delta_S_PPN=C*tM*(-z/r/4.0+15.0/4.0*acos(-z/r)/sqrt(1.0-z/r*z/r));
    double Delta_FD=-chi*2.0*tM*C/(1.0-z/r)*skn;

    if(psr.op.Delta_S_2PN==0){
        Delta_S_DPN=0;
        Delta_S_PPN=0;
    }
    if(psr.op.Delta_FD==0){
        Delta_FD=0;
    }

    double toa0=Delta_R+Delta_S_1PN+Delta_S_DPN+Delta_S_PPN+Delta_FD;
    if(toa0<=toa[0]){
        d0=gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd,1e-8,1e-12,1e-12);
    }else{
        d0=gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd,-1e-8,1e-12,1e-12);
    }

    double rvEt[8]={x,y,z,vx,vy,vz,0.0,0.0};
    gsl_odeiv2_driver_apply(d0,&toa0,toa[0],rvEt);

    gsl_odeiv2_driver_free(d0);

    /*按顺序计算每一个toa*/
    gsl_odeiv2_driver *d=gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd,1e-8,1e-12,1e-12);

    for(int i=0;i<n;i++){
        gsl_odeiv2_driver_apply(d,&toa0,toa[i],rvEt);
        out[i].toa=toa0;

        x=rvEt[0];y=rvEt[1];z=rvEt[2];vx=rvEt[3];vy=rvEt[4];vz=rvEt[5];
        r=sqrt(x*x+y*y+z*z);
        vr=(x*vx+y*vy+z*vz)/r;
        C=G*M/r/c/c;
        
        double t=rvEt[7];
        double Delta_E=rvEt[6];

        double pro_t=sqrt(mu_alpha*mu_alpha+mu_delta*mu_delta)*t;

        Delta_R=(z*cos(pro_t)+sin(pro_t)*(x*mu_alpha+y*mu_delta)/sqrt(mu_alpha*mu_alpha+mu_delta*mu_delta))/c;
        Delta_S_1PN=-2.0*tM*log((r-z)/a);
        Delta_S_DPN=-4.0*tM*C/(1.0-z/r);
        Delta_S_PPN=C*tM*(-z/r/4.0+15.0/4.0*acos(-z/r)/sqrt(1.0-z/r*z/r));
        Delta_FD=-chi*2.0*tM*C/(1.0-z/r)*skn;

        double vke=vy*sin(lambda_p)*cos(eta_p)-vx*sin(lambda_p)*sin(eta_p);
        double ke2=sin(lambda_p)*sin(lambda_p);
        double ve=vx*sin(lambda_p)*cos(eta_p)+vy*sin(lambda_p)*sin(eta_p)+vz*cos(lambda_p);
        double nke=(y*sin(lambda_p)*cos(eta_p)-x*sin(lambda_p)*sin(eta_p))/r;

        double Delta_A1=-vke/c/ke2/2.0/pi/nu;
        double Delta_A2=Delta_A1*(vz/2.0/c+(vz-cos(lambda_p)*ve)/c/ke2);
        double Delta_L=2.0*C/(1.0-z/r)/2.0/pi/nu*nke/ke2;

        if(psr.op.proper_motion==0){
            Delta_R=z/c;
        }
        if(psr.op.Delta_S_2PN==0){
            Delta_S_DPN=0;
            Delta_S_PPN=0;
        }
        if(psr.op.Delta_A1==0){
            Delta_A1=0;
        }
        if(psr.op.Delta_A2==0){
            Delta_A2=0;
        }
        if(psr.op.Delta_FD==0){
            Delta_FD=0;
        }
        if(psr.op.Delta_L==0){
            Delta_L=0;
        }

        out[i].t=t; //这能拿来检查自洽性...
        out[i].T=t-Delta_E-Delta_A1-Delta_A2-Delta_L;
        out[i].N=N0+nu*out[i].T+dnu/2.0*out[i].T*out[i].T+d2nu/6.0*out[i].T*out[i].T*out[i].T+d3nu/24.0*pow(out[i].T,4.0)+d4nu/120.0*pow(out[i].T,5.0);
        out[i].Delta_E=Delta_E;
        out[i].Delta_R=Delta_R;
        out[i].Delta_S_1PN=Delta_S_1PN;
        out[i].Delta_S_DPN=Delta_S_DPN;
        out[i].Delta_S_PPN=Delta_S_PPN;
        out[i].Delta_A1=Delta_A1;
        out[i].Delta_A2=Delta_A2;
        out[i].Delta_FD=Delta_FD;
        out[i].Delta_L=Delta_L;
    }

    gsl_odeiv2_driver_free(d);

}

double chi2_white(int n, double toa[], double N[], double sigma_TOA[], struct star_system sys){
    struct toa_output out[n];
    //为了减少python中的调用，我们把oe到rv的初始化放在这个地方
    star_PSR_rv_ini(&sys.psr,sys.BH,0);
    residual_cal(sys,n,toa,out);

    double chi2=0;
    for(int i=0;i<n;i++){
        chi2+=((out[i].N-N[i])/sys.psr.nu/sigma_TOA[i])*((out[i].N-N[i])/sys.psr.nu/sigma_TOA[i]);
    }
    return chi2;
}

gsl_matrix *design_matrix(int n, double toa[], struct star_system sys){
    int dim=0;
    for(int i=0;i<28;i++){
        dim+=(int)(sys.fit_op[i]);
    }
    gsl_matrix *M=gsl_matrix_alloc(n,dim);

    dim=0;
    struct star_system sys1,sys2,sys3,sys4;
    struct toa_output out1[n],out2[n],out3[n],out4[n];
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu,mu_alpha,mu_delta,lambda_p,eta_p,d3nu,d4nu,m_I
    for(int i=0;i<28;i++){
        if(sys.fit_op[i]==1){
            sys1=sys;
            sys2=sys;
            sys3=sys;
            sys4=sys;

            double del;
            switch(i){
                case 0:
                    del=sys.BH.M*1e-6;
                    sys1.BH.M+=2.0*del;
                    sys2.BH.M+=del;
                    sys3.BH.M-=del;
                    sys4.BH.M-=2.0*del;
                    break;
                case 1:
                    del=5.0*1e-6;
                    sys1.BH.chi+=2.0*del;
                    sys2.BH.chi+=del;
                    sys3.BH.chi-=del;
                    sys4.BH.chi-=2.0*del;
                    break;
                case 2:
                    del=0.2*1e-1;
                    sys1.BH.q+=2.0*del;
                    sys2.BH.q+=del;
                    sys3.BH.q-=del;
                    sys4.BH.q-=2.0*del;
                    break;
                case 3:
                    del=1e-5;
                    sys1.BH.lambda+=2.0*del;
                    sys2.BH.lambda+=del;
                    sys3.BH.lambda-=del;
                    sys4.BH.lambda-=2.0*del;
                    break;
                case 4:
                    del=2.0*1e-5;
                    sys1.BH.eta+=2.0*del;
                    sys2.BH.eta+=del;
                    sys3.BH.eta-=del;
                    sys4.BH.eta-=2.0*del;
                    break;
                case 5:
                    del=sys.psr.oe[0]*0.2*1e-8;
                    sys1.psr.oe[0]+=2.0*del;
                    sys2.psr.oe[0]+=del;
                    sys3.psr.oe[0]-=del;
                    sys4.psr.oe[0]-=2.0*del;
                    break;
                case 6:
                    del=0.3*1e-6;
                    sys1.psr.oe[1]+=2.0*del;
                    sys2.psr.oe[1]+=del;
                    sys3.psr.oe[1]-=del;
                    sys4.psr.oe[1]-=2.0*del;
                    break;
                case 7:
                    del=3.0*1e-7;
                    sys1.psr.oe[2]+=2.0*del;
                    sys2.psr.oe[2]+=del;
                    sys3.psr.oe[2]-=del;
                    sys4.psr.oe[2]-=2.0*del;
                    break;
                case 8:
                    del=1.0*1e-2;
                    sys1.psr.oe[3]+=2.0*del;
                    sys2.psr.oe[3]+=del;
                    sys3.psr.oe[3]-=del;
                    sys4.psr.oe[3]-=2.0*del;
                    break;
                case 9:
                    del=2.0*1e-7;
                    sys1.psr.oe[4]+=2.0*del;
                    sys2.psr.oe[4]+=del;
                    sys3.psr.oe[4]-=del;
                    sys4.psr.oe[4]-=2.0*del;
                    break;
                case 10:
                    del=sys.psr.oe[0]*0.6*1e-7;
                    // del=1e-6;
                    sys1.psr.oe[5]+=2.0*del;
                    sys2.psr.oe[5]+=del;
                    sys3.psr.oe[5]-=del;
                    sys4.psr.oe[5]-=2.0*del;
                    break;
                case 11:
                    del=0.7*1e-1;
                    sys1.psr.N0+=2.0*del;
                    sys2.psr.N0+=del;
                    sys3.psr.N0-=del;
                    sys4.psr.N0-=2.0*del;
                    break;
                case 12:
                    del=1.0/sec*1e-10*0.8;
                    sys1.psr.nu+=2.0*del;
                    sys2.psr.nu+=del;
                    sys3.psr.nu-=del;
                    sys4.psr.nu-=2.0*del;
                    break;
                case 13:
                    del=1e-15/sec/sec*1e-3*2.0;
                    sys1.psr.dnu+=2.0*del;
                    sys2.psr.dnu+=del;
                    sys3.psr.dnu-=del;
                    sys4.psr.dnu-=2.0*del;
                    break;
                case 14:
                    del=1e-25/sec/sec/sec*0.3;
                    sys1.psr.d2nu+=2.0*del;
                    sys2.psr.d2nu+=del;
                    sys3.psr.d2nu-=del;
                    sys4.psr.d2nu-=2.0*del;
                    break;
                case 15:
                    del=0.1*mas/yr*10.0;
                    sys1.BH.mu_alpha+=2.0*del;
                    sys2.BH.mu_alpha+=del;
                    sys3.BH.mu_alpha-=del;
                    sys4.BH.mu_alpha-=2.0*del;
                    break;
                case 16:
                    del=0.2*mas/yr*10;
                    sys1.BH.mu_delta+=2.0*del;
                    sys2.BH.mu_delta+=del;
                    sys3.BH.mu_delta-=del;
                    sys4.BH.mu_delta-=2.0*del;
                    break;
                case 17:
                    del=2.0*1e-2;
                    sys1.psr.lambda_p+=2.0*del;
                    sys2.psr.lambda_p+=del;
                    sys3.psr.lambda_p-=del;
                    sys4.psr.lambda_p-=2.0*del;
                    break;
                case 18:
                    del=2.0*1e-2;
                    sys1.psr.eta_p+=2.0*del;
                    sys2.psr.eta_p+=del;
                    sys3.psr.eta_p-=del;
                    sys4.psr.eta_p-=2.0*del;
                    break;
                case 19:
                    del=1e-33/sec/sec/sec/sec;
                    sys1.psr.d3nu+=2.0*del;
                    sys2.psr.d3nu+=del;
                    sys3.psr.d3nu-=del;
                    sys4.psr.d3nu-=2.0*del;
                    break;
                case 20:
                    del=1e-40/sec/sec/sec/sec/sec;
                    sys1.psr.d4nu+=2.0*del;
                    sys2.psr.d4nu+=del;
                    sys3.psr.d4nu-=del;
                    sys4.psr.d4nu-=2.0*del;
                    break;
                case 21:
                    del=1.0*M_sun;
                    sys1.IM.m+=2.0*del;
                    sys2.IM.m+=1.0*del;
                    sys3.IM.m-=1.0*del;
                    sys4.IM.m-=2.0*del;
                    break;
                case 22:
                    del=sys.IM.oe[0]*1e-6;
                    sys1.IM.oe[0]+=2.0*del;
                    sys2.IM.oe[0]+=1.0*del;
                    sys3.IM.oe[0]-=1.0*del;
                    sys4.IM.oe[0]-=2.0*del;
                    break;
                case 23:
                    del=1e-4;
                    sys1.IM.oe[1]+=2.0*del;
                    sys2.IM.oe[1]+=1.0*del;
                    sys3.IM.oe[1]-=1.0*del;
                    sys4.IM.oe[1]-=2.0*del;
                    break;
                case 24:
                    del=1e-4;
                    sys1.IM.oe[2]+=2.0*del;
                    sys2.IM.oe[2]+=1.0*del;
                    sys3.IM.oe[2]-=1.0*del;
                    sys4.IM.oe[2]-=2.0*del;
                    break;
                case 25:
                    del=1e-4;
                    sys1.IM.oe[3]+=2.0*del;
                    sys2.IM.oe[3]+=1.0*del;
                    sys3.IM.oe[3]-=1.0*del;
                    sys4.IM.oe[3]-=2.0*del;
                    break;
                case 26:
                    del=1e-4;
                    sys1.IM.oe[4]+=2.0*del;
                    sys2.IM.oe[4]+=1.0*del;
                    sys3.IM.oe[4]-=1.0*del;
                    sys4.IM.oe[4]-=2.0*del;
                    break;
                case 27:
                    del=sys.IM.oe[0]*1e-5;
                    sys1.IM.oe[5]+=2.0*del;
                    sys2.IM.oe[5]+=1.0*del;
                    sys3.IM.oe[5]-=1.0*del;
                    sys4.IM.oe[5]-=2.0*del;
                    break;
                default: break;
            }

            if(i==0 || i==5 || i==6 || i==7 || i==8 || i==9 || i==10){
                star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
                star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
                star_PSR_rv_ini(&sys3.psr,sys3.BH,0);
                star_PSR_rv_ini(&sys4.psr,sys4.BH,0);
            }

            residual_cal(sys1,n,toa,out1);
            residual_cal(sys2,n,toa,out2);
            residual_cal(sys3,n,toa,out3);
            residual_cal(sys4,n,toa,out4);

            for(int j=0;j<n;j++){
                gsl_matrix_set(M,j,dim,(-out1[j].N+8.0*out2[j].N-8.0*out3[j].N+out4[j].N)/sys.psr.nu/12.0/del);
            }

            dim++;
        }
    }

    return M;
}

gsl_matrix *Fisher_white_from_design(int n, double toa[], double sigma_TOA[], struct star_system sys){
    int dim=0;
    for(int i=0;i<28;i++){
        dim+=(int)(sys.fit_op[i]);
    }
    gsl_matrix *M=design_matrix(n,toa,sys);
    gsl_matrix *fisher=gsl_matrix_alloc(dim,dim);
    //先处理一下M按j把sigma_j给除了
    for(int j=0;j<n;j++){
        for(int i=0;i<dim;i++){
            gsl_matrix_set(M,j,i,gsl_matrix_get(M,j,i)/sigma_TOA[j]);
        }
    }
    // printf("$d ?\n",dim);
    gsl_blas_dsyrk(CblasUpper,CblasTrans,1.0,M,0.0,fisher);
    // printf("?\n");
    for(int i=0;i<dim;i++){
        for(int j=0;j<i;j++){
            gsl_matrix_set(fisher,i,j,gsl_matrix_get(fisher,j,i));
        }
    }

    gsl_matrix_free(M);
    return fisher;
}

// 也许我们这里应该写一个Fisher from design的code,这个老的Fisher white应该被弃用
gsl_matrix *Fisher_white(int n, double toa[], double sigma_TOA[], struct star_system sys){
    int dim=0;
    for(int i=0;i<19;i++){
        dim+=(int)(sys.fit_op[i]);
    }

    gsl_matrix *fisher=gsl_matrix_calloc(dim,dim);
    gsl_matrix *V=gsl_matrix_alloc(dim,n);

    dim=0;
    struct star_system sys1,sys2,sys3,sys4;
    struct toa_output out1[n],out2[n],out3[n],out4[n];
    //M,chi,q,lambda,eta,Pb,e,omega,Omega,inc,T0,N0,nu,dnu,d2nu
    for(int i=0;i<19;i++){
        if(sys.fit_op[i]==1){
            sys1=sys;
            sys2=sys;
            sys3=sys;
            sys4=sys;

            double del;
            switch(i){
                case 0:
                    del=sys.BH.M*1e-8;
                    sys1.BH.M+=2.0*del;
                    sys2.BH.M+=del;
                    sys3.BH.M-=del;
                    sys4.BH.M-=2.0*del;
                    break;
                case 1:
                    del=0.5*1e-5;
                    sys1.BH.chi+=2.0*del;
                    sys2.BH.chi+=del;
                    sys3.BH.chi-=del;
                    sys4.BH.chi-=2.0*del;
                    break;
                case 2:
                    del=0.3*1e-4;
                    sys1.BH.q+=2.0*del;
                    sys2.BH.q+=del;
                    sys3.BH.q-=del;
                    sys4.BH.q-=2.0*del;
                    break;
                case 3:
                    del=1.0*1e-5;
                    sys1.BH.lambda+=2.0*del;
                    sys2.BH.lambda+=del;
                    sys3.BH.lambda-=del;
                    sys4.BH.lambda-=2.0*del;
                    break;
                case 4:
                    del=3.0*1e-5;
                    sys1.BH.eta+=2.0*del;
                    sys2.BH.eta+=del;
                    sys3.BH.eta-=del;
                    sys4.BH.eta-=2.0*del;
                    break;
                case 5:
                    del=sys.psr.oe[0]*0.5*1e-9;
                    sys1.psr.oe[0]+=2.0*del;
                    sys2.psr.oe[0]+=del;
                    sys3.psr.oe[0]-=del;
                    sys4.psr.oe[0]-=2.0*del;
                    break;
                case 6:
                    del=1e-9;
                    sys1.psr.oe[1]+=2.0*del;
                    sys2.psr.oe[1]+=del;
                    sys3.psr.oe[1]-=del;
                    sys4.psr.oe[1]-=2.0*del;
                    break;
                case 7:
                    del=5.0*1e-8;
                    sys1.psr.oe[2]+=2.0*del;
                    sys2.psr.oe[2]+=del;
                    sys3.psr.oe[2]-=del;
                    sys4.psr.oe[2]-=2.0*del;
                    break;
                case 8:
                    del=3.0*1e-5;
                    sys1.psr.oe[3]+=2.0*del;
                    sys2.psr.oe[3]+=del;
                    sys3.psr.oe[3]-=del;
                    sys4.psr.oe[3]-=2.0*del;
                    break;
                case 9:
                    del=3.0*1e-8;
                    sys1.psr.oe[4]+=2.0*del;
                    sys2.psr.oe[4]+=del;
                    sys3.psr.oe[4]-=del;
                    sys4.psr.oe[4]-=2.0*del;
                    break;
                case 10:
                    del=sys.psr.oe[0]*0.5*1e-9;
                    sys1.psr.oe[5]+=2.0*del;
                    sys2.psr.oe[5]+=del;
                    sys3.psr.oe[5]-=del;
                    sys4.psr.oe[5]-=2.0*del;
                    break;
                case 11:
                    del=1e-3;
                    sys1.psr.N0+=2.0*del;
                    sys2.psr.N0+=del;
                    sys3.psr.N0-=del;
                    sys4.psr.N0-=2.0*del;
                    break;
                case 12:
                    del=1.0/sec*1e-11;
                    sys1.psr.nu+=2.0*del;
                    sys2.psr.nu+=del;
                    sys3.psr.nu-=del;
                    sys4.psr.nu-=2.0*del;
                    break;
                case 13:
                    del=1e-15/sec/sec*1e-4*3;
                    sys1.psr.dnu+=2.0*del;
                    sys2.psr.dnu+=del;
                    sys3.psr.dnu-=del;
                    sys4.psr.dnu-=2.0*del;
                    break;
                case 14:
                    del=1e-26/sec/sec/sec*0.5;
                    sys1.psr.d2nu+=2.0*del;
                    sys2.psr.d2nu+=del;
                    sys3.psr.d2nu-=del;
                    sys4.psr.d2nu-=2.0*del;
                    break;
                case 15:
                    del=0.1*mas/yr;
                    sys1.BH.mu_alpha+=2.0*del;
                    sys2.BH.mu_alpha+=del;
                    sys3.BH.mu_alpha-=del;
                    sys4.BH.mu_alpha-=2.0*del;
                    break;
                case 16:
                    del=0.1*mas/yr;
                    sys1.BH.mu_delta+=2.0*del;
                    sys2.BH.mu_delta+=del;
                    sys3.BH.mu_delta-=del;
                    sys4.BH.mu_delta-=2.0*del;
                    break;
                case 17:
                    del=3.0*1e-3;
                    sys1.psr.lambda_p+=2.0*del;
                    sys2.psr.lambda_p+=del;
                    sys3.psr.lambda_p-=del;
                    sys4.psr.lambda_p-=2.0*del;
                    break;
                case 18:
                    del=3.0*1e-3;
                    sys1.psr.eta_p+=2.0*del;
                    sys2.psr.eta_p+=del;
                    sys3.psr.eta_p-=del;
                    sys4.psr.eta_p-=2.0*del;
                    break;
                default: break;
            }

            if(i==0 || i==5 || i==6 || i==7 || i==8 || i==9 || i==10){
                star_PSR_rv_ini(&sys1.psr,sys1.BH,0);
                star_PSR_rv_ini(&sys2.psr,sys2.BH,0);
                star_PSR_rv_ini(&sys3.psr,sys3.BH,0);
                star_PSR_rv_ini(&sys4.psr,sys4.BH,0);
            }

            residual_cal(sys1,n,toa,out1);
            residual_cal(sys2,n,toa,out2);
            residual_cal(sys3,n,toa,out3);
            residual_cal(sys4,n,toa,out4);

            for(int j=0;j<n;j++){
                gsl_matrix_set(V,dim,j,(-out1[j].N+8.0*out2[j].N-8.0*out3[j].N+out4[j].N)/sys.psr.nu/sigma_TOA[j]/12.0/del);
            }

            dim++;
        }
    }

    gsl_blas_dsyrk(CblasUpper,CblasNoTrans,1.0,V,0.0,fisher);
    for(int i=0;i<dim;i++){
        for(int j=0;j<i;j++){
            gsl_matrix_set(fisher,i,j,gsl_matrix_get(fisher,j,i));
        }
    }

    gsl_matrix_free(V);
    return fisher;
}

gsl_matrix *Fisher_to_Cov(gsl_matrix *fisher){
    //对一个对称正定的矩阵求逆
    gsl_matrix *cov=gsl_matrix_alloc(fisher->size1,fisher->size2);
    gsl_matrix_memcpy(cov,fisher);

    //使用Cholesky分解的方法
    gsl_linalg_cholesky_decomp1(cov);
    gsl_linalg_cholesky_invert(cov);

    return cov;
}

gsl_matrix *residual_covariance(int n, double toa[], double sigma_TOA[], struct red_noise_params rp){
    gsl_matrix *Cov=gsl_matrix_calloc(n,n);

    for(int i=0;i<n;i++){
        gsl_matrix_set(Cov,i,i,sigma_TOA[i]*sigma_TOA[i]+rp.A*sqrt(pi)*rp.fc*\
        gsl_sf_gamma((rp.alpha-1.0)/2.0)/gsl_sf_gamma(rp.alpha/2.0));
    }

    for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            double tau=toa[j]-toa[i];
            gsl_matrix_set(Cov,i,j,2.0*rp.A*pow(rp.fc,(rp.alpha+1.0)/2.0)*pow(pi,rp.alpha/2.0)/gsl_sf_gamma(rp.alpha/2.0)*\
            pow(tau,(rp.alpha-1.0)/2.0)*gsl_sf_bessel_Knu((rp.alpha-1.0)/2.0,2.0*pi*rp.fc*tau));
        }
    }

    for(int i=0;i<n;i++){
        for(int j=0;j<i;j++){
            gsl_matrix_set(Cov,i,j,gsl_matrix_get(Cov,j,i));
        }
    }

    return Cov;
}

