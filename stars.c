#include "stars.h"

#include <math.h>
#include <stdio.h>

#include "const.h"

// static long double c=vc;

/*oe[6]={Pb,e,omega,Omega,inc,T0}*/
void star_PSR_rv_ini(struct PSR *psr, struct SMBH BH, bool clear_op){
    double oe[6];
    for(int i=0;i<6;i++){
        oe[i]=psr->oe[i];
    }

    double M=BH.M;
    double Pb=oe[0],e=oe[1],omega=oe[2],Omega=oe[3],inc=oe[4],T0=oe[5];
    double a=pow(G*M*Pb*Pb/4.0/pi/pi,1.0/3.0);

    double rhs=-2.0*pi*T0/Pb;
    double u0=rhs;
    double u1=u0+(rhs-(u0-e*sin(u0)))/(1.0-e*cos(u0));
    int n_iter=0,max_iter=10;
    while(fabs(u1-u0)>1e-15 && n_iter<max_iter){
        u0=u1;
        u1=u0+(rhs-(u0-e*sin(u0)))/(1.0-e*cos(u0));
        n_iter++;
    }

    double f=2.0*atan2(sqrt((1.0+e)/(1.0-e))*sin(u1/2.0),cos(u1/2.0));
    // printf("f0 = %.16g\n",f);
    // double f=T0;

    double r=a*(1.0-e*e)/(1.0+e*cos(f));
    double v=sqrt(G*M/a/(1.0-e*e));
    double cos_O=cos(Omega),cos_of=cos(omega+f),cos_i=cos(inc),sin_O=sin(Omega),\
    sin_of=sin(omega+f),sin_i=sin(inc),sin_o=sin(omega),cos_o=cos(omega);

    psr->rv[0]=r*(cos_O*cos_of-cos_i*sin_O*sin_of);
    psr->rv[1]=r*(sin_O*cos_of+cos_i*cos_O*sin_of);
    psr->rv[2]=r*sin_i*sin_of;
    psr->rv[3]=-v*(cos_O*(sin_of+e*sin_o)+cos_i*sin_O*(cos_of+e*cos_o));
    psr->rv[4]=-v*(sin_O*(sin_of+e*sin_o)-cos_i*cos_O*(cos_of+e*cos_o));
    psr->rv[5]=v*sin_i*(cos_of+e*cos_o);

    if(clear_op==true){
        psr->op.Newton=0;
        psr->op.Test_1PN=0;
        psr->op.Test_2PN=0;
        psr->op.Test_Q=0;
        psr->op.Test_S=0;
        psr->op.Delta_S_2PN=0;
        psr->op.Delta_FD=0;
        psr->op.Delta_E_2PN=0;
        psr->op.proper_motion=0;
        psr->op.Delta_A1=0;
        psr->op.Delta_A2=0;
        psr->op.Delta_L=0;
        psr->op.IMBH=0;
        psr->op.star_cluster=0;
        psr->op.No_hair=0;
        psr->op.My_cluster=0;
    }

    double v2=psr->rv[3]*psr->rv[3]+psr->rv[4]*psr->rv[4]+psr->rv[5]*psr->rv[5];
    double E=v2/2.0-G*M/r+3.0*v2*v2/8.0/c/c+G*M/2.0/r/c/c*(3.0*v2+G*M/r);
    double a1PN=-G*M/2.0/E*(1.0+7.0/2.0*E/c/c);
    //这其实不对，1PN部分的积分会贡献一些讨厌的离心率依赖项在后面的2PN项里面
    //但我们干脆就用这个表达式也无妨
    //这跟把a1PN换成aN一样都是会在2PN阶上引入区别。。。
    psr->Delta_E_norm=1.0-2.0*G*M/a1PN/c/c-E/c/c+(E*E+2.0*G*M*G*M/a1PN/a1PN+2.0*E*G*M/a1PN)/c/c/c/c;

    return;
}

void oe_to_rv(double M, double t, double oe[6], double rv[6]){
    //在牛顿阶，我们暂时只算r
    double Pb=oe[0],e=oe[1],omega=oe[2],Omega=oe[3],inc=oe[4],t0=oe[5];
    double n=2.0*pi/Pb;
    double rhs=n*(t-t0);
    double u0=rhs;
    double u1=u0+(rhs-(u0-e*sin(u0)))/(1.0-e*cos(u0));
    int n_iter=0,max_iter=15;
    while(fabs(u1-u0)>1e-15 && n_iter<max_iter){
        u0=u1;
        u1=u0+(rhs-(u0-e*sin(u0)))/(1.0-e*cos(u0));
        n_iter++;
    }
    double f=2.0*atan2(sqrt((1.0+e)/(1.0-e))*sin(u1/2.0),cos(u1/2.0));
    double a=pow(G*M*Pb*Pb/4.0/pi/pi,1.0/3.0);

    double r=a*(1.0-e*e)/(1.0+e*cos(f));
    double v=sqrt(G*M/a/(1.0-e*e));
    double cos_O=cos(Omega),cos_of=cos(omega+f),cos_i=cos(inc),sin_O=sin(Omega),\
    sin_of=sin(omega+f),sin_i=sin(inc),sin_o=sin(omega),cos_o=cos(omega);

    rv[0]=r*(cos_O*cos_of-cos_i*sin_O*sin_of);
    rv[1]=r*(sin_O*cos_of+cos_i*cos_O*sin_of);
    rv[2]=r*sin_i*sin_of;
    rv[3]=-v*(cos_O*(sin_of+e*sin_o)+cos_i*sin_O*(cos_of+e*cos_o));
    rv[4]=-v*(sin_O*(sin_of+e*sin_o)-cos_i*cos_O*(cos_of+e*cos_o));
    rv[5]=v*sin_i*(cos_of+e*cos_o);
    // printf("%.16g\n",rv[0]);
}

double star_u2v(double u,double e){
    int n_2pi=0;
    n_2pi=(int)floor(u/2.0/pi);
    u-=n_2pi*2.0*pi;
    while(u>pi){
        u-=2.0*pi;
        n_2pi+=1;
    }
    while(u<-pi){
        u+=2.0*pi;
        n_2pi-=1;
    }

    return 2.0*atan2(sqrt((1.0+e)/(1.0-e))*sin(u/2.0),cos(u/2.0))+n_2pi*2.0*pi;
}

