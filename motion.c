#include "motion.h"

#include <stdio.h>
#include <math.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

#include "const.h"
#include "stars.h"
#include "star_cluster.h"

// long double c=vc;

int func_mot_evolve(double t, const double rv[], double f[], void *params){
    //r={x,y,z,vx,vy,vz,Delta_E}
    double x=rv[0],y=rv[1],z=rv[2],vx=rv[3],vy=rv[4],vz=rv[5];
    double r=sqrt(x*x+y*y+z*z);
    double v2=vx*vx+vy*vy+vz*vz;
    double vr=(vx*x+vy*y+vz*z)/r;
    
    struct star_system *sys=(struct star_system *)params;
    double M=sys->BH.M;
    double chi=sys->BH.chi;
    double q=sys->BH.q;
    if(sys->psr.op.No_hair==1){
        q=-chi*chi;
    }
    double lambda=sys->BH.lambda;
    // double eta=sys->BH.eta+sys->psr.oe[3];
    double eta=sys->BH.eta;
    double Delta_E_norm=sys->psr.Delta_E_norm;

    double nx=x/r;
    double ny=y/r;
    double nz=z/r;
    double FN=G*M/r/r;
    double C=G*M/r/c/c;
    double b2=v2/c/c;
    double bx=vx/c;
    double by=vy/c;
    double bz=vz/c;
    double br=vr/c;
    double sx=sin(lambda)*cos(eta);
    double sy=sin(lambda)*sin(eta);
    double sz=cos(lambda);

    double aN[3]={0},aT1PN[3]={0},aT2PN[3]={0},aTS[3]={0},aTQ[3]={0},dE1=0,dE2=0;

    if(sys->psr.op.Newton==1){
        aN[0]=-FN*nx;
        aN[1]=-FN*ny;
        aN[2]=-FN*nz;
    }
    if(sys->psr.op.Test_1PN==1){
        aT1PN[0]=-FN*((-4.0*C+b2)*nx-4.0*br*bx);
        aT1PN[1]=-FN*((-4.0*C+b2)*ny-4.0*br*by);
        aT1PN[2]=-FN*((-4.0*C+b2)*nz-4.0*br*bz);
    }
    if(sys->psr.op.Test_2PN==1){
        aT2PN[0]=FN*C*(nx*(2.0*br*br-9.0*C)-2.0*br*bx);
        aT2PN[1]=FN*C*(ny*(2.0*br*br-9.0*C)-2.0*br*by);
        aT2PN[2]=FN*C*(nz*(2.0*br*br-9.0*C)-2.0*br*bz);
    }
    if(sys->psr.op.Test_S==1){
        double snb=sx*(ny*bz-nz*by)+sy*(nz*bx-nx*bz)+sz*(nx*by-ny*bx);
        aTS[0]=chi*6.0*FN*C*(nx*snb+br*(ny*sz-nz*sy)-2.0/3.0*(by*sz-bz*sy));
        aTS[1]=chi*6.0*FN*C*(ny*snb+br*(nz*sx-nx*sz)-2.0/3.0*(bz*sx-bx*sz));
        aTS[2]=chi*6.0*FN*C*(nz*snb+br*(nx*sy-ny*sx)-2.0/3.0*(bx*sy-by*sx));
    }
    if(sys->psr.op.Test_Q==1){
        double ns=nx*sx+ny*sy+nz*sz;
        aTQ[0]=-q*3.0/2.0*FN*C*C*(nx*(5.0*ns*ns-1.0)-2.0*sx*ns);
        aTQ[1]=-q*3.0/2.0*FN*C*C*(ny*(5.0*ns*ns-1.0)-2.0*sy*ns);
        aTQ[2]=-q*3.0/2.0*FN*C*C*(nz*(5.0*ns*ns-1.0)-2.0*sz*ns);
    }

    double aIM[3]={0};
    if(sys->psr.op.IMBH==1){
        double IM_rv[6];
        oe_to_rv(M,t,sys->IM.oe,IM_rv);
        double xIP=x-IM_rv[0],yIP=y-IM_rv[1],zIP=z-IM_rv[2];
        double rIP=sqrt(xIP*xIP+yIP*yIP+zIP*zIP);
        aIM[0]=-G*sys->IM.m/rIP/rIP*xIP/rIP;
        aIM[1]=-G*sys->IM.m/rIP/rIP*yIP/rIP;
        aIM[2]=-G*sys->IM.m/rIP/rIP*zIP/rIP;
        // printf("%.16g\n",aIM[2]);
    }

    double aCluster[3]={0};
    if(sys->psr.op.star_cluster==1){
        for(int i=0;i<Sstars.n;i++){
            double IM_rv[6];
            oe_to_rv(M,t,Sstars.stars[i].oe,IM_rv);
            double xIP=x-IM_rv[0],yIP=y-IM_rv[1],zIP=z-IM_rv[2];
            double rIP=sqrt(xIP*xIP+yIP*yIP+zIP*zIP);
// printf("%.3g ",rIP);
            aCluster[0]+=-G*Sstars.stars[i].m/rIP/rIP*xIP/rIP;
            aCluster[1]+=-G*Sstars.stars[i].m/rIP/rIP*yIP/rIP;
            aCluster[2]+=-G*Sstars.stars[i].m/rIP/rIP*zIP/rIP;
        }
    }

    double aMyCluster[3]={0};
    if(sys->psr.op.My_cluster==1){
        for(int i=0;i<sys->My_cluster.n;i++){
            double IM_rv[6];
            oe_to_rv(M,t,sys->My_cluster.stars[i].oe,IM_rv);
            double xIP=x-IM_rv[0],yIP=y-IM_rv[1],zIP=z-IM_rv[2];
            double rIP=sqrt(xIP*xIP+yIP*yIP+zIP*zIP);
            aMyCluster[0]+=-G*sys->My_cluster.stars[i].m/rIP/rIP*xIP/rIP;
            aMyCluster[1]+=-G*sys->My_cluster.stars[i].m/rIP/rIP*yIP/rIP;
            aMyCluster[2]+=-G*sys->My_cluster.stars[i].m/rIP/rIP*zIP/rIP;
        }
    }

    dE1=1.0-C-b2/2.0;
    if(sys->psr.op.Delta_E_2PN==1){
        dE2=C*C/2.0-3.0/2.0*C*b2-b2*b2/8.0;
    }

    f[0]=vx;
    f[1]=vy;
    f[2]=vz;
    f[3]=aN[0]+aT1PN[0]+aT2PN[0]+aTS[0]+aTQ[0]+aIM[0]+aCluster[0]+aMyCluster[0];
    f[4]=aN[1]+aT1PN[1]+aT2PN[1]+aTS[1]+aTQ[1]+aIM[1]+aCluster[1]+aMyCluster[1];
    f[5]=aN[2]+aT1PN[2]+aT2PN[2]+aTS[2]+aTQ[2]+aIM[2]+aCluster[2]+aMyCluster[2];
    f[6]=1.0-(dE1+dE2)/Delta_E_norm;
// printf("%.16g %.16g %.16g %.16g %.16g %.16g %.16g\n",t,x,y,z,aMyCluster[0]/FN,aMyCluster[1]/FN,aMyCluster[2]/FN);
    return GSL_SUCCESS;
}
void mot_evolve(struct star_system star_sys, int n, double t[], struct mot_output out[]){
    gsl_odeiv2_system sys={func_mot_evolve,NULL,7,&star_sys};

    double t_now=0.0;
    //在t=0时刻，t=T=0，Delta_E=t-T=0
    //(我们要把initialcondition改成toa是0的时候是。。。。
    //所以要先算出toa=0时对应的t)改回来此处

    struct PSR psr=star_sys.psr;
    // double N0=psr.N0,nu=psr.nu,dnu=psr.dnu,d2nu=psr.d2nu,d3nu=psr.d3nu,d4nu=psr.d4nu;
    // double lambda_p=psr.lambda_p, eta_p=psr.eta_p+psr.oe[3];
    // struct SMBH BH=star_sys.BH;
    // double M=BH.M,chi=BH.chi;
    // double lambda=BH.lambda,eta=BH.eta+psr.oe[3];
    // double mu_alpha=BH.mu_alpha,mu_delta=BH.mu_delta;
    // double pro_t=0;

    double rv[7]={psr.rv[0],psr.rv[1],psr.rv[2],psr.rv[3],psr.rv[4],psr.rv[5],0.0};
    // double x=rv[0],y=rv[1],z=rv[2];
    // double vx=rv[3],vy=rv[4],vz=rv[5];
    // double r=sqrt(x*x+y*y+z*z);
    // // double a=3.0*G*M/2.0/c/c/(1.0-psr.Delta_E_norm);
    // double Pb=psr.oe[0];
    // double a=pow(G*M*Pb*Pb/4.0/pi/pi,1.0/3.0);
    // double tM=G*M/c/c/c;
    // double C=G*M/r/c/c;
    // double vke=vy*sin(lambda_p)*cos(eta_p)-vx*sin(lambda_p)*sin(eta_p);
    // double ke2=sin(lambda_p)*sin(lambda_p);
    // double ve=vx*sin(lambda_p)*cos(eta_p)+vy*sin(lambda_p)*sin(eta_p)+vz*cos(lambda_p);
    // double nke=(y*sin(lambda_p)*cos(eta_p)-x*sin(lambda_p)*sin(eta_p))/r;
    // double skn=sin(lambda)*sin(eta)*x/r-sin(lambda)*cos(eta)*y/r;

    // double Delta_R=z/c;
    // double Delta_S_1PN=0;
    // double Delta_S_DPN=0;
    // double Delta_S_PPN=0;
    // double Delta_FD=0;
    // if(star_sys.psr.op.proper_motion==1){
    //     Delta_R=(z*cos(pro_t)+sin(pro_t)*(x*mu_alpha+y*mu_delta)/sqrt(mu_alpha*mu_alpha+mu_delta*mu_delta))/c;
    // }
    // Delta_S_1PN=-2.0*tM*log((r-z)/a);
    // if(star_sys.psr.op.Delta_S_2PN==1){
    //     Delta_S_DPN=-4.0*tM*C/(1.0-z/r);
    //     Delta_S_PPN=C*tM*(-z/r/4.0+15.0/4.0*acos(-z/r)/sqrt(1.0-z/r*z/r));
    // }
    // if(star_sys.psr.op.Delta_FD==1){
    //     Delta_FD=-chi*2.0*tM*C/(1.0-z/r)*skn;
    // }
    
    // double Delta_S1=Delta_S_1PN;
    // double Delta_S2=Delta_S_DPN+Delta_S_PPN;

    // double t0=0-Delta_R-Delta_S1-Delta_S2-Delta_FD;
    // double t0_out=t0;
    // printf("%.16g\n",t0_out);

    gsl_odeiv2_driver *d0;
    if(t_now<=t[0]){
        d0=gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd,1e-8,0,1e-13);
    }else{
        // printf("?\n");
        d0=gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd,-1e-8,0,1e-13);
    }
    gsl_odeiv2_driver_apply(d0,&t_now,t[0],rv);
    gsl_odeiv2_driver_free(d0);

    gsl_odeiv2_driver *d=gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd,1e-8,0,1e-13);
    for(int i=0;i<n;i++){
        gsl_odeiv2_driver_apply(d,&t_now,t[i],rv);
// printf("%.16g %.16g\n",t_now,rv[0]);
        out[i].t=t_now;
        out[i].rv[0]=rv[0];
        out[i].rv[1]=rv[1];
        out[i].rv[2]=rv[2];
        out[i].rv[3]=rv[3];
        out[i].rv[4]=rv[4];
        out[i].rv[5]=rv[5];
        out[i].Delta_E=rv[6];
        // out[i].t0=t0_out;
    }

    gsl_odeiv2_driver_free(d);
    return;
}

/*由于解析模型总共也没几个，所以分开来写，这样也能省去类型判断的代码*/
void mot_evolve_Newton(struct star_system star_sys, int n, double t[], struct mot_output out[]){
    struct PSR psr=star_sys.psr;
    struct SMBH BH=star_sys.BH;

    /*使用开普勒方程演化的话，首先要把pv转化成oe*/
    /*oe[6]={Pb,e,omega,Omega,inc,T0}*/
    double x=psr.rv[0],y=psr.rv[1],z=psr.rv[2],vx=psr.rv[3],vy=psr.rv[4],vz=psr.rv[5];
    double M=BH.M;

    double r=sqrt(x*x+y*y+z*z);
    double v2=vx*vx+vy*vy+vz*vz;
    double vrr=vx*x+vy*y+vz*z;

    double a=1.0/(2.0/r-v2/G/M);
    double Pb=2.0*pi*sqrt(a*a*a/G/M);

    double e_cos_u=1.0-r/a;
    double e_sin_u=vrr/sqrt(G*M*a);

    double e=sqrt(e_cos_u*e_cos_u+e_sin_u*e_sin_u);

    double u0=atan2(e_sin_u,e_cos_u);
    double T0=-Pb/2.0/pi*(u0-e_sin_u);

    double f0=2.0*atan2(sqrt((1.0+e)/(1.0-e))*sin(u0/2.0),cos(u0/2.0));

    double Lx=y*vz-z*vy;
    double Ly=z*vx-x*vz;
    double Lz=x*vy-y*vx;
    double L=sqrt(Lx*Lx+Ly*Ly+Lz*Lz);
    double cos_i=Lz/L;
    double inc=acos(cos_i);

    double cos_Omega=-Ly/(L*sin(inc));
    double sin_Omega=Lx/(L*sin(inc));
    double Omega=atan2(sin_Omega,cos_Omega);

    double sin_omega_f=z/r/sin(inc);
    double cos_omega_f=(x*cos_Omega+y*sin_Omega)/r;
    double vv=sqrt(G*M/a/(1.0-e*e));
    double cos_omega=(vz/vv/sin(inc)-cos_omega_f)/e;
    double sin_omega=(-(vx*cos_Omega+vy*sin_Omega)/vv-sin_omega_f)/e;
    double omega=atan2(sin_omega,cos_omega);

    /*从t_now=0开始演化*/
    /*对于解析模型，逐步的演化也许是没有必要的，并不会带来计算量的优势，反而会积累误差*/
    /*每个t的结果应由初值而非上一步的值直接计算得到*/
    for(int i=0;i<n;i++){
        out[i].t=t[i];

        /*需要计算t[i]时刻的f*/
        double rhs=2.0*pi*(t[i]-T0)/Pb;
        double u0=rhs;
        double u1=u0+(rhs-(u0-e*sin(u0)))/(1.0-e*cos(u0));
        int n_iter=0,max_iter=10;
        while(fabs(u1-u0)>1e-15 && n_iter<max_iter){
            u0=u1;
            u1=u0+(rhs-(u0-e*sin(u0)))/(1.0-e*cos(u0));
            n_iter++;
        }
        double f=2.0*atan2(sqrt((1.0+e)/(1.0-e))*sin(u1/2.0),cos(u1/2.0));

        double cos_O=cos_Omega,cos_of=cos(omega+f),sin_O=sin_Omega,\
        sin_of=sin(omega+f),sin_i=sin(inc),sin_o=sin_omega,cos_o=cos_omega;
        double rr=a*(1.0-e*e)/(1.0+e*cos(f));

        out[i].rv[0]=rr*(cos_O*cos_of-cos_i*sin_O*sin_of);
        out[i].rv[1]=rr*(sin_O*cos_of+cos_i*cos_O*sin_of);
        out[i].rv[2]=rr*sin_i*sin_of;
        out[i].rv[3]=-vv*(cos_O*(sin_of+e*sin_o)+cos_i*sin_O*(cos_of+e*cos_o));
        out[i].rv[4]=-vv*(sin_O*(sin_of+e*sin_o)-cos_i*cos_O*(cos_of+e*cos_o));
        out[i].rv[5]=vv*sin_i*(cos_of+e*cos_o);
    }

    return;
}

void mot_evolve_T1PN(struct star_system star_sys, int n, double t[], struct mot_output out[]){
    struct PSR psr=star_sys.psr;
    struct SMBH BH=star_sys.BH;

    double x=psr.rv[0],y=psr.rv[1],z=psr.rv[2],vx=psr.rv[3],vy=psr.rv[4],vz=psr.rv[5];
    double M=BH.M;

    /*1PN的解析解采用DD模型，因此仅"精确"到1PN*/
    /*此时的对齐采用初始rv相同的对齐方式*/
    /*为使用DD形式，我们需要计算E和J，注意是测试粒子情形*/
    /*注意，除了E和J，实际上rv还将给定t0,inc,Omega,omega0*/
    double r=sqrt(x*x+y*y+z*z);
    double v2=vx*vx+vy*vy+vz*vz;
    double vr=(vx*x+vy*y+vz*z)/r;

    double E=v2/2.0-G*M/r+3.0*v2*v2/8.0/c/c+G*M/2.0/r/c/c*(3.0*v2+G*M/r);
    double JxN=y*vz-z*vy;
    double JyN=z*vx-x*vz;
    double JzN=x*vy-y*vx;
    double JN=sqrt(JxN*JxN+JyN*JyN+JzN*JzN);
    double J=JN*(1.0+v2/2.0/c/c+3.0*G*M/r/c/c);

    double cos_i=JzN/JN;
    double inc=acos(cos_i);
    double sin_i=sin(inc);

    double cos_Omega=-JyN/(JN*sin(inc));
    double sin_Omega=JxN/(JN*sin(inc));
    double Omega=atan2(sin_Omega,cos_Omega);

    /*需要计算一系列参数*/
    double nT=pow(-2.0*E,1.5)/G/M*(1.0+15.0/4.0*E/c/c);
    double et=sqrt(1.0+2.0*E/G/M/G/M*(1.0+17.0/2.0*E/c/c)*(J*J+2.0*G*G*M*M/c/c));
    double K=J/sqrt(J*J-6.0*G*M/c*G*M/c);
    double etheta=sqrt(1.0+2.0*E/G/M/G/M*(1.0-15.0/2.0*E/c/c)*(J*J-6.0*G*M*G*M/c/c));
    double aR=-G*M/2.0/E*(1.0+7.0/2.0*E/c/c);
    double eR=sqrt(1.0+2.0*E/G/M/G/M*(1.0-15.0/2.0*E/c/c)*(J*J-6.0*G*M*G*M/c/c));

    /*下面我们需要进行"对齐"，即利用rv确定t0和omega0*/
    /*我们约定rv是t=0时的，而t=t0时theta=omega0:尽管这仍不能完全确定t0和omega0,但我们只要任取一组自洽的即可*/
    double cos_u0=(1.0-r/aR)/eR;
    double sin_u0=vr*(1.0-et*cos_u0)/aR/eR/nT;
    double u0=atan2(sin_u0,cos_u0); //这个的取值范围在-pi到pi
    double t0=-(u0-et*sin_u0)/nT;

    double sin_theta0=z/r/sin_i;
    double cos_theta0=(x*cos_Omega+y*sin_Omega)/r;
    double theta0=atan2(sin_theta0,cos_theta0);
    double omega0=theta0-2.0*K*atan2(sqrt((1.0+etheta)/(1.0-etheta))*sin(u0/2.0),cos(u0/2.0));

    /*接下来就是生成解了*/
    for(int i=0;i<n;i++){
        out[i].t=t[i];

        /*计算t[i]时刻的u,注意最后使用u1*/
        double rhs=nT*(t[i]-t0);
        double u0=rhs;
        double u1=u0+(rhs-(u0-et*sin(u0)))/(1.0-et*cos(u0));
        int n_iter=0,max_iter=10;
        while(fabs(u1-u0)>1e-15 && n_iter<max_iter){
            u0=u1;
            u1=u0+(rhs-(u0-et*sin(u0)))/(1.0-et*cos(u0));
            n_iter++;
        }

        /*需要计算正确的theta,应该要先把u1归化到-pi到pi的取值范围内*/
        int n_2pi=0;
        n_2pi=(int)floor(u1/2.0/pi);
        u1-=n_2pi*2.0*pi;
        while(u1>pi){
            u1-=2.0*pi;
            n_2pi+=1;
        }
        while(u1<-pi){
            u1+=2.0*pi;
            n_2pi-=1;
        }

        double cos_u=cos(u1),sin_u=sin(u1);

        double r=aR*(1-eR*cos_u);
        double theta=omega0+2.0*K*(n_2pi*pi+atan2(sqrt((1.0+etheta)/(1.0-etheta))*sin(u1/2.0),cos(u1/2.0)));
        
        double cos_O=cos_Omega,sin_O=sin_Omega,cos_of=cos(theta),sin_of=sin(theta);

        double vr=aR*eR*sin_u*nT/(1.0-et*cos_u);
        double d_theta=2.0*K*sqrt((1.0+etheta)/(1.0-etheta))/(1.0+cos_u+((1.0+etheta)/(1.0-etheta))*(1.0-cos_u));

        out[i].rv[0]=r*(cos_O*cos_of-cos_i*sin_O*sin_of);
        out[i].rv[1]=r*(sin_O*cos_of+cos_i*cos_O*sin_of);
        out[i].rv[2]=r*sin_i*sin_of;
        out[i].rv[3]=vr*(cos_O*cos_of-cos_i*sin_O*sin_of)-r*(cos_O*sin_of+cos_i*sin_O*cos_of)*d_theta;
        out[i].rv[4]=vr*(sin_O*cos_of+cos_i*cos_O*sin_of)-r*(sin_O*sin_of-cos_i*cos_O*cos_of)*d_theta;
        out[i].rv[5]=vr*sin_i*sin_of+r*sin_i*cos_of*d_theta;
    }
}

/*最后一个解析模型考虑采用Wex的2PN with spin*/
/*但这里比较危险的是涉及到SSC和参考系的选取。。。*/
/*ADM坐标和协和坐标的关系是...？*/
/*在Damour & Schaeer 1988中提及了，是一个简单的变换关系，因此让我们最终都进入协和坐标中，注意转换即可*/
/*同样需要注意的是SSC可能带来的影响，但暂时先不考虑，先验证2PN结果*/
void mot_evolve_T2PN(struct star_system star_sys, int n, double t[], struct mot_output out[]){
    struct PSR psr=star_sys.psr;
    struct SMBH BH=star_sys.BH;

    double x=psr.rv[0],y=psr.rv[1],z=psr.rv[2],vx=psr.rv[3],vy=psr.rv[4],vz=psr.rv[5];
    double r_DD=sqrt(x*x+y*y+z*z);
    double vr_DD=(x*vx+y*vy+z*vz)/r_DD;
    double M=BH.M;

    /*第一步就是考虑坐标转换，这是因为我们的初值是在DD坐标下给的，而Wex解在ADM坐标下*/
    /*这不仅有坐标的变换还有速度的变换*/
    x*=1.0-G*M/r_DD/c/c*G*M/r_DD/c/c/4.0;
    y*=1.0-G*M/r_DD/c/c*G*M/r_DD/c/c/4.0;
    z*=1.0-G*M/r_DD/c/c*G*M/r_DD/c/c/4.0;
    vx+=-G*M/r_DD/c/c*G*M/r_DD/c/c/4.0*vx+G*M/r_DD/c/c*G*M/r_DD/c/c/2.0*vr_DD*x/r_DD;
    vy+=-G*M/r_DD/c/c*G*M/r_DD/c/c/4.0*vy+G*M/r_DD/c/c*G*M/r_DD/c/c/2.0*vr_DD*y/r_DD;
    vz+=-G*M/r_DD/c/c*G*M/r_DD/c/c/4.0*vz+G*M/r_DD/c/c*G*M/r_DD/c/c/2.0*vr_DD*z/r_DD;

    /*同样我们需要计算E和J*/
    /*麻烦的是p的计算，我们需要导出v和p的关系。。。这在E和J中都要用到*/
    /*好在他们间的关系只是差一个倍数*/
    double v2=vx*vx+vy*vy+vz*vz;
    double r=sqrt(x*x+y*y+z*z);
    double vr=(vx*x+vy*y+vz*z)/r;
    double C=G*M/r/c/c;
    double b2=v2/c/c;

    double v2p=1.0+b2/2.0+3.0*C+3.0/8.0*b2*b2+7.0/2.0*b2*C+4.0*C*C;
    double px=vx*v2p;
    double py=vy*v2p;
    double pz=vz*v2p;
    double p2=v2*v2p*v2p;

    double E=p2/2.0-G*M/r-p2*p2/8.0/c/c-3.0*p2/2.0*C+G*M/2.0/r*C+p2/c/c*p2*p2/16.0/c/c+5.0*p2*p2/c/c/8.0*C+5.0*p2/2.0*C*C-G*M/r/4.0*C*C;
    double Jx=(y*pz-z*py)/G/M;
    double Jy=(z*px-x*pz)/G/M;
    double Jz=(x*py-y*px)/G/M;
    double J=sqrt(Jx*Jx+Jy*Jy+Jz*Jz);

    /*需要计算一系列参数*/
    double nT=pow(-2.0*E,1.5)/G/M*(1.0+15.0/4.0*E/c/c+555.0/32.0*E/c/c*E/c/c-7.5*pow(-2.0*E,1.5)/c/c/c/(J*c));
    double et=sqrt(1.0+2.0*E*J*J+4.0*E/c/c+17.0*E/c/c*E*J*J+4.0*E/c/c*E/c/c-17.0*E/c/c/J/c/J/c+112.0*E/c/c*E/c/c*E*J*J-15.0*(1.0+2.0*\
    E*J*J)*pow(-2.0*E,1.5)/c/c/c/(J*c));
    double gt_c4=7.5*pow(-2.0*E,1.5)/c/c/c/(J*c);
    double a=-G*M/2.0/E*(1.0+3.5*E/c/c+E/c/c/4.0*E/c/c+17.0/2.0*E/c/c/(J*c)/(J*c));
    double er=sqrt(1.0+2.0*E*J*J-12.0*E/c/c-15.0*E/c/c*E*J*J+26.0*E/c/c*E/c/c-34.0*E/c/c/(J*c)/(J*c)+80.0*E/c/c*E/c/c*E*J*J);
    double k_c2=3.0/J/c/J/c*(1.0+2.5*E/c/c+35.0/4.0/J/c/J/c);
    double ep=sqrt(1.0+2.0*E*J*J-12.0*E/c/c-15.0*E*J*J*E/c/c-8.0*E/c/c*E/c/c+80.0*E/c/c*E/c/c*E*J*J-51.0*E/c/c/J/c/J/c);

double fix_term=+4.2e-7;
printf("fix_term/k_c2 = %g\n",fix_term/k_c2);
k_c2+=fix_term;

    /*剩下4个轨道参数也需要计算出来*/
    double cos_i=Jz/J;
    double inc=acos(cos_i);
    double sin_i=sin(inc);

    double cos_Omega=-Jy/(J*sin_i);
    double sin_Omega=Jx/(J*sin_i);
    double Omega=atan2(sin_Omega,cos_Omega);

    double cos_u0=(1.0-r/a)/er;
    double sin_u0=vr/a/er/nT*(1.0-et*cos_u0+gt_c4*(2.0*sqrt((1.0+ep)/(1.0-ep))/(1.0+cos_u0+(1.0+ep)/(1.0-ep)*(1.0-cos_u0))-1.0));
    double u0=atan2(sin_u0,cos_u0);
    double v0=star_u2v(u0,ep);
    double t0=-(u0-et*sin_u0+gt_c4*(v0-u0))/nT;

    double sin_theta0=z/r/sin_i;
    double cos_theta0=(x*cos_Omega+y*sin_Omega)/r;
    double theta0=atan2(sin_theta0,cos_theta0);
    double omega0=theta0-(1.0+k_c2)*v0;

    /*接下来即可计算运动*/
    /*计算完运动还需要转化回DD坐标*/
    for(int i=0;i<n;i++){
        out[i].t=t[i];

        /*计算t[i]时刻的u*/
        double rhs=nT*(t[i]-t0);
        double u0=rhs;
        double u1=u0+(rhs-(u0-et*sin(u0)+gt_c4*(star_u2v(u0,ep)-u0)))/(1.0-et*cos(u0)+gt_c4*(2.0*sqrt((1.0+ep)/(1.0-ep))/(1.0+\
        cos(u0)+(1.0+ep)/(1.0-ep)*(1.0-cos(u0)))-1.0));
        int n_iter=0,max_iter=15;
        while(fabs(u1-u0)>1e-15 && n_iter<max_iter){
            u0=u1;
            u1=u0+(rhs-(u0-et*sin(u0)+gt_c4*(star_u2v(u0,ep)-u0)))/(1.0-et*cos(u0)+gt_c4*(2.0*sqrt((1.0+ep)/(1.0-ep))/(1.0+\
            cos(u0)+(1.0+ep)/(1.0-ep)*(1.0-cos(u0)))-1.0));
            n_iter++;
        }

        double r=a*(1.0-er*cos(u1));
        double theta=omega0+(1.0+k_c2)*star_u2v(u1,ep);

        double cos_O=cos_Omega,sin_O=sin_Omega,cos_of=cos(theta),sin_of=sin(theta);

        double d_theta=(1.0+k_c2)*2.0*sqrt((1.0+ep)/(1.0-ep))/(1.0+cos(u1)+(1.0+ep)/(1.0-ep)*(1.0-cos(u1)))*nT/(1.0-et*cos(u1)+gt_c4*(\
        2.0*sqrt((1.0+ep)/(1.0-ep))/(1.0+cos(u1)+(1.0+ep)/(1.0-ep)*(1.0-cos(u1)))-1.0));

        double x_ADM=r*(cos_O*cos_of-cos_i*sin_O*sin_of);
        double y_ADM=r*(sin_O*cos_of+cos_i*cos_O*sin_of);
        double z_ADM=r*sin_i*sin_of;
        double vx_ADM=vr*(cos_O*cos_of-cos_i*sin_O*sin_of)-r*(cos_O*sin_of+cos_i*sin_O*cos_of)*d_theta;
        double vy_ADM=vr*(sin_O*cos_of+cos_i*cos_O*sin_of)-r*(sin_O*sin_of-cos_i*cos_O*cos_of)*d_theta;
        double vz_ADM=vr*sin_i*sin_of+r*sin_i*cos_of*d_theta;

        double r_ADM=sqrt(x_ADM*x_ADM+y_ADM*y_ADM+z_ADM*z_ADM);
        double ADM2DD=1.0+G*M/r_ADM/c/c*G*M/r_ADM/c/c/4.0;
        double vr_ADM=(vx_ADM*x_ADM+vy_ADM*y_ADM+vz_ADM*z_ADM)/r_ADM;

        out[i].rv[0]=x_ADM*ADM2DD;
        out[i].rv[1]=y_ADM*ADM2DD;
        out[i].rv[2]=z_ADM*ADM2DD;
        out[i].rv[3]=vx_ADM*ADM2DD-G*M/c/c/r_ADM*G*M/c/c/r_ADM/2.0*x_ADM/r_ADM*vr_ADM;
        out[i].rv[4]=vy_ADM*ADM2DD-G*M/c/c/r_ADM*G*M/c/c/r_ADM/2.0*y_ADM/r_ADM*vr_ADM;
        out[i].rv[5]=vz_ADM*ADM2DD-G*M/c/c/r_ADM*G*M/c/c/r_ADM/2.0*z_ADM/r_ADM*vr_ADM;
    }
}

