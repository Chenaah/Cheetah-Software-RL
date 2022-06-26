/*
*author:Technician13
*date:2020.4.10
*/
#pragma once
#include<iostream>
#include<cmath>
#include<math.h>

class QuadrupedInverseKinematic
{
    public:
        QuadrupedInverseKinematic(float x, float y, float z);
        virtual ~QuadrupedInverseKinematic();
        //List the functions one by one for preserving the interface
        double calc_dyz();
        double calc_lyz();
        float calc_L_gamma();
        float calc_R_gamma();
        double calc_lxz();
        double calc_n();
        float calc_beta();
        float calc_alpha();

        float L_gamma;//The first joint angle rotation angle of left legs
        float R_gamma;//The first joint angle rotation angle of left legs
        float alpha;//The second joint angle rotation angle
        float beta;//The third joint angle rotation angle
    private:
        //given         
        float pos_x;//The position of the toe - X axis
        float pos_y;//The position of the toe - Y axis
        float pos_z;//The position of the toe - Z axis
        float h;//The length of the first link
        float hu;//The length of the second link
        float hl;//The length of the third link
        //find
        
        //middle parameters
        double dyz;//Length from O to P on Y-Z
        double lyz;//Length from top-pitch-joint to P on Y-Z
        double lxz;//Length from Qxz' to Pxz'
        double n;
};

QuadrupedInverseKinematic::QuadrupedInverseKinematic(float x, float y, float z)
{
    // std::cout<<"Please input parameters:";
    std::cout << "=============== [DEBUG] IK BEGINS ===============" << std::endl;
    std::cout<<"X-axis:  " <<x<<std::endl;
    std::cout<<"Y-axis:  " <<y<<std::endl;
    std::cout<<"Z-axis:  " <<z<<std::endl;

    pos_x = x;
    pos_y = y;
    pos_z = z;

    h = 0.008; // The length of the first link
    hu = 0.215; // The length of the second link
    hl = 0.215; // The length of the third link

}

QuadrupedInverseKinematic::~QuadrupedInverseKinematic()
{
    std::cout<<"The result is:  "<<std::endl
             <<"LEFT:"<<std::endl
             <<"The first joint angle rotation angle gamma:  "
             <<L_gamma<<std::endl
             <<"The second joint angle rotation angle alpha:  "
             <<alpha<<std::endl
             <<"The third joint angle rotation angle beta:  "
             <<beta<<std::endl

             <<"RIGHT:"<<std::endl
             <<"The first joint angle rotation angle gamma:  "
             <<R_gamma<<std::endl
             <<"The second joint angle rotation angle alpha:  "
             <<alpha<<std::endl
             <<"The third joint angle rotation angle beta:  "
             <<beta<<std::endl;
}

double QuadrupedInverseKinematic::calc_dyz()
{
    dyz = sqrt(pos_y*pos_y+pos_z*pos_z);
    std::cout<<"pos_y*pos_y: "<<pos_y*pos_y<<"  pos_z*pos_z: "<<pos_z*pos_z<<"    dyz: "<<dyz<<std::endl;
    // std::cout<<dyz<<std::endl;
    return dyz;
}

double QuadrupedInverseKinematic::calc_lyz()
{
    lyz = sqrt(dyz*dyz-h*h);
    // std::cout<<lyz<<std::endl;
    std::cout<< "[DEBUG] " << "dyz*dyz = " << dyz*dyz << "  h*h = " << h*h << "  dyz*dyz-h*h = " << dyz*dyz-h*h << "  sqrt(dyz*dyz-h*h) (lyz) = " << sqrt(dyz*dyz-h*h) <<std::endl;
    assert(dyz*dyz-h*h >= 0);
    return lyz;
}

float QuadrupedInverseKinematic::calc_R_gamma()
{
    float gamma_1, gamma_2;
    gamma_1=atan(pos_y/pos_z);
    gamma_2=atan(h/lyz);
    R_gamma = gamma_1 - gamma_2;
    // std::cout<<R_gamma<<std::endl;
    std::cout<<"[DEBUG] " << "gamma_1: " << gamma_1 << "  gamma_2: " << gamma_2 << "  R_gamma (gamma_1 - gamma_2): "<<R_gamma<<std::endl;

    if (R_gamma > 3.1415926/2) R_gamma -= 3.1415926;

    return R_gamma;
}

float QuadrupedInverseKinematic::calc_L_gamma()
{
    float gamma_1, gamma_2;
    gamma_1=atan(pos_y/pos_z);
    gamma_2=atan(h/lyz);
    L_gamma = gamma_1 + gamma_2;
    // std::cout<<L_gamma<<std::endl;
    std::cout<<"[DEBUG] " << "gamma_1: " << gamma_1 << "  gamma_2: " << gamma_2 << "  L_gamma (gamma_1 + gamma_2): "<<L_gamma<<std::endl;

    if (L_gamma > 3.1415926/2) L_gamma -= 3.1415926;

    return L_gamma;
}

double QuadrupedInverseKinematic::calc_lxz()
{
    lxz = sqrt(lyz*lyz+pos_x*pos_x);
    // std::cout<<lxz<<std::endl;
    return lxz;
}

double QuadrupedInverseKinematic::calc_n()
{
    n = (lxz*lxz-hl*hl-hu*hu)/(2*hu);
    // std::cout<<n<<std::endl;
    return n;
}

float QuadrupedInverseKinematic::calc_beta()
{
    // beta = acos(n/hl);
    beta = - acos(n/hl);
    // std::cout<<beta<<std::endl;
    return beta;
}

float QuadrupedInverseKinematic::calc_alpha()
{
    float alpha_1, alpha_2;
    alpha_1 = atan(pos_x/lyz);
    // std::cout << "[DEBUG] " << "pos_x = " << pos_x << "  lyz = " << lyz << "  pos_x/lyz = " << pos_x/lyz << "  atan(pos_x/lyz) = " << atan(pos_x/lyz) << std::endl;
    // alpha_2 = -atan((hu+n)/lxz);
    alpha_2 = acos((hu+n)/lxz);

    std::cout<<"alpha1:  "<<alpha_1<<std::endl;
    if (pos_z < 0)
        alpha_1 = - 3.14159 - alpha_1;
    std::cout<<"--> alpha1:  "<<alpha_1<<std::endl;

    alpha = alpha_1 + alpha_2;
    // std::cout<<"dyz: "<<dyz<<"    lyz: "<<lyz<<"    pos_x: "<<pos_x<<std::endl;
    // std::cout<<"hu+n: "<<hu+n<<"    lxz: "<<lxz<<"    (hu+n)/lxz: "<<(hu+n)/lxz<<std::endl;
    // std::cout<<"acos((hu+n)/lxz): " << acos((hu+n)/lxz) <<std::endl;
    // std::cout<<"=========================================================="<<std::endl;
    // alpha = 3.14159/2 - alpha_1 - acos((hu+n)/lxz);
    // std::cout<<alpha<<std::endl;
    return alpha;
}

void LegsIK(const float x, const float y, const float z, float &th0l, float &th0r, float &th1, float &th2)
{
    QuadrupedInverseKinematic *p=new QuadrupedInverseKinematic(x, y, z);
    p->calc_dyz();
    p->calc_lyz();
    p->calc_L_gamma();
    p->calc_R_gamma();
    p->calc_lxz();
    p->calc_n();
    p->calc_beta();
    p->calc_alpha();

    // std::cout<<"The result is:  "<<std::endl
    //          <<"LEFT:"<<std::endl
    //          <<"The first joint angle rotation angle gamma:  "
    //          <<p->L_gamma<<std::endl
    //          <<"The second joint angle rotation angle alpha:  "
    //          <<p->alpha<<std::endl
    //          <<"The third joint angle rotation angle beta:  "
    //          <<p->beta<<std::endl

    //          <<"RIGHT:"<<std::endl
    //          <<"The first joint angle rotation angle gamma:  "
    //          <<p->R_gamma<<std::endl
    //          <<"The second joint angle rotation angle alpha:  "
    //          <<p->alpha<<std::endl
    //          <<"The third joint angle rotation angle beta:  "
    //          <<p->beta<<std::endl;

    th0l = p->L_gamma;
    th0r = p->R_gamma;
    th1 = p->alpha;
    th2 = p->beta;

    // delete p;
    // p = nullptr;
    // system("pause");
    // return 0;
}
