//
// Created by user on 2020/3/25.
//

/*=========================== Balance Stand ===========================*/
/**
 * FSM State that forces all legs to be on the ground and uses the QP
 * Balance controller for instantaneous balance control.
 */

#include "FSM_State_FrontJump2.h"
#include <Controllers/WBC_Ctrl/LocomotionCtrl/LocomotionCtrl.hpp>

/**
 * Constructor for the FSM State that passes in state specific info to
 * the generic FSM State constructor.
 *
 * @param _controlFSMData holds all of the relevant control data
 */
template <typename T>
FSM_State_FrontJump2<T>::FSM_State_FrontJump2(
        ControlFSMData<T>* _controlFSMData)
        : FSM_State<T>(_controlFSMData, FSM_StateName::FRONTJUMP2,"FRONTJUMP2") {
    // Set the pre controls safety checks
    this->turnOffAllSafetyChecks();
    // Turn off Foot pos command since it is set in WBC as operational task
    this->checkPDesFoot = false;


    // Initialize GRF to 0s
    this->footFeedForwardForces = Mat34<T>::Zero();

    _wbc_ctrl = new LocomotionCtrl<T>(_controlFSMData->_quadruped->buildModel());
    _wbc_data = new LocomotionCtrlData<T>();

    _wbc_ctrl->setFloatingBaseWeight(1000.);
}

template <typename T>
void FSM_State_FrontJump2<T>::onEnter() {
    // Default is to not transition
    this->nextStateName = this->stateName;

    // Reset the transition data
    this->transitionData.zero();

    // Always set the gait to be standing in this state
    this->_data->_gaitScheduler->gaitData._nextGait = GaitType::STAND;

    _ini_body_pos = (this->_data->_stateEstimator->getResult()).position;
    _wbc_data->pBody_des=_ini_body_pos;
    if(_ini_body_pos[2] < 0.2) {
        _ini_body_pos[2] = 0.25;
    }
    //   _ini_body_pos[2]=0.26;

    for(size_t i(0); i < 4; ++i) {
        initial_jpos[i] = this->_data->_legController->datas[i].q;
    }

    last_height_command = _ini_body_pos[2];

    _ini_body_ori_rpy = (this->_data->_stateEstimator->getResult()).rpy;

    _wbc_data->pBody_RPY_des=_ini_body_ori_rpy;
    _body_weight = this->_data->_quadruped->_bodyMass * 9.81;
    _count=0;
    enter_once=0;
    enter_once_withdraw=0;
    enter_once_3=0;
    enter_once_2=0;
}

/**
 * Calls the functions to be executed on each control loop iteration.
 */
template <typename T>
void FSM_State_FrontJump2<T>::run() {
    Vec4<T> contactState;
    contactState<< 0.5, 0.5, 0.5, 0.5;
    this->_data->_stateEstimator->setContactPhase(contactState);
    BalanceStandStep();
}

/**
 * Manages which states can be transitioned into either by the user
 * commands or state event triggers.
 *
 * @return the enumerated FSM state name to transition into
 */
template <typename T>
FSM_StateName FSM_State_FrontJump2<T>::checkTransition() {
    // Get the next state
    _iter++;

    // Switch FSM control mode
    switch ((int)this->_data->controlParameters->control_mode) {
        case K_FRONTJUMP2:
            break;

        case K_RECOVERY_STAND:
            this->nextStateName = FSM_StateName::RECOVERY_STAND;
            break;

        case K_LOCOMOTION:
            this->nextStateName = FSM_StateName::LOCOMOTION;
            break;

        case K_PASSIVE:  // normal c
            this->nextStateName = FSM_StateName::PASSIVE;
            break;

        case K_BALANCE_STAND:
            this->nextStateName = FSM_StateName::BALANCE_STAND;
            break;

        default:
            std::cout << "[CONTROL FSM] Bad Request: Cannot transition from "
                      << K_FRONTJUMP << " to "
                      << this->_data->controlParameters->control_mode << std::endl;
    }

    // Return the next state name to the FSM
    return this->nextStateName;
}

/**
 * Handles the actual transition for the robot between states.
 * Returns true when the transition is completed.
 *
 * @return true if transition is complete
 */
template <typename T>
TransitionData<T> FSM_State_FrontJump2<T>::transition() {
    // Switch FSM control mode
    switch (this->nextStateName) {
        case FSM_StateName::PASSIVE:  // normal
            this->transitionData.done = true;
            break;

        case FSM_StateName::BALANCE_STAND:
            this->transitionData.done = true;
            break;

        case FSM_StateName::LOCOMOTION:
            this->transitionData.done = true;
            break;

        case FSM_StateName::RECOVERY_STAND:
            this->transitionData.done = true;
            break;


        default:
            std::cout << "[CONTROL FSM] Something went wrong in transition"
                      << std::endl;
    }

    // Return the transition data to the FSM
    return this->transitionData;
}

/**
 * Cleans up the state information on exiting the state.
 */
template <typename T>
void FSM_State_FrontJump2<T>::onExit() {
    _iter = 0;
}
template <typename T>
void FSM_State_FrontJump2<T>::BalanceStandStep() {

    _count++;


    if (_count < 1500) {
/*            if(_count<500)
//                _wbc_data->pBody_RPY_des[1] = -0.15 * _count / 500.0;
//            else
//                _wbc_data->pBody_RPY_des[1] = -0.15;
//
//            if (_wbc_data->pBody_des[2] > 0.1)
//                _wbc_data->pBody_des[2] -= 0.0001;
//            else
//                _wbc_data->pBody_des[2] = 0.1;
//
//            _wbc_data->vBody_Ori_des.setZero();
//            for (size_t i(0); i < 4; ++i) {
//                _wbc_data->Fr_des[i].setZero();
//                _wbc_data->contact_state[i] = true;
//            }
//            _wbc_ctrl->run(_wbc_data, *this->_data);
 */

//            fin<<0,-1.4,2.21;
        fin<<0,-1.0,2.275;
        ff_force<<0,0,-200;
        ff_force=ff_force*0.0;
        for(int leg=0;leg<2;leg++) //前腿
        {
            _SetJPosInterPts(_count,1200,leg,initial_jpos[leg],fin);
            this->_data->_legController->commands[leg].forceFeedForward=ff_force;
        }

        fin<<0,-1.6,2.5;
        ff_force<<-300,0,-250;
        ff_force=ff_force*0;
        for(int leg=2;leg<4;leg++) //后腿
        {
            _SetJPosInterPts(_count,1200,leg,initial_jpos[leg],fin);
            this->_data->_legController->commands[leg].forceFeedForward=ff_force;
        }


    } else if (_count < 1600) {
        if (enter_once == 0) {
            for (size_t i(0); i < 4; ++i) {
                initial_jpos[i] = this->_data->_legController->datas[i].q;
                initial_Ppos[i] = this->_data->_legController->datas[i].p;
            }
            enter_once = 1;
        }
        double delta_t=0.002;
//            Vec3<float> v_front(-3.5,0,-2.2);
//            Vec3<float> v_rear(-3.5,0,-0.8);
        Vec3<float> v_front(-3.5,0,-2.2);
        Vec3<float> v_rear(-3.5,0,-0.4);
        Vec3<float> pos;
        kpMat << 80, 0, 0, 0, 80, 0, 0, 0, 80;
        kdMat << 2, 0, 0, 0, 2, 0, 0, 0, 2;
        ff_force<<-0,0,-0;
        for (int leg = 0; leg < 4; ++leg) {
            if(leg<2)
                initial_Ppos[leg]=initial_Ppos[leg]+delta_t*v_front;
            else
                initial_Ppos[leg]=initial_Ppos[leg]+delta_t*v_rear;

            pos=initial_Ppos[leg];
            if(pos(0)<-0.350) {
                pos(0)=-0.350;
            }
            if(pos(2)<-0.300) {
                pos(2)=-0.300;
            }
            fin = InverseKinematics(pos);

            this->_data->_legController->commands[leg].qDes = fin;
            this->_data->_legController->commands[leg].kpJoint = kpMat;
            this->_data->_legController->commands[leg].kdJoint = kdMat;
            this->_data->_legController->commands[leg].forceFeedForward=ff_force;
        }
    } else if (_count < 1700) {
        if (enter_once_withdraw == 0) {
            for (size_t i(0); i < 4; ++i) {
                initial_jpos[i] = this->_data->_legController->datas[i].q;
                initial_Ppos[i] = this->_data->_legController->datas[i].p;
            }
            enter_once_withdraw = 1;
        }
        //ini << 0, -1.1, 0.8;
//            fin << 0, -0.55, 1.9;
        ff_force << 0, 0, 0;
        for (int leg = 0; leg < 4; leg++)
        {
            if(leg<2)
                fin <<0, -0.98, 2.2;// 0, -0.7, 1.9;
            else
                fin << 0, -0.98, 2.15;//0, -0.7, 1.75;

            if (leg % 2 == 0)
                fin(0) = -0.15;
            else
                fin(0) = 0.15;
            _SetJPosInterPts(_count - 1600, 100, leg, initial_jpos[leg], fin);
            this->_data->_legController->commands[leg].forceFeedForward = ff_force;
        }
    } else {

        Vec3<float> finp;
//            finp << 0, -0.8, 2.0;
        kpMat << 40, 0, 0, 0, 20, 0, 0, 0, 20;
        kdMat << 3, 0, 0, 0, 3, 0, 0, 0, 3;
        for (int leg = 0; leg < 4; ++leg) {

            if(leg<2)
                finp << 0, -0.98, 2.2;//0, -0.7, 1.9;//
            else
                finp << 0, -0.98, 2.15;//0, -0.7, 1.75;

            if (leg % 2 == 0)
                finp(0) = -0.15;
            else
                finp(0) = 0.15;
            this->_data->_legController->commands[leg].qDes = finp;
            this->_data->_legController->commands[leg].kpJoint = kpMat;
            this->_data->_legController->commands[leg].kdJoint = kdMat;
        }
    }

}



template <typename T>
Vec3<T> FSM_State_FrontJump2<T>::InverseKinematics(Vec3<T> pos)
{
    Vec3<T> ang;
    float hipx=pos(0);
//    double hipy=pos(1);
    float hipz=pos(2);
    float L1=0.209;
    float L2=0.195;
    float PI=3.1415926;
    ang(0)=0;
    float L12=sqrt((hipz)*(hipz)+hipx*hipx);
    if(L12>0.38) L12=0.38;
    else if(L12<0.06) L12=0.06;
//printf("IK:L12:%.2f\n",L12);
    float fai=acos((L1*L1+L12*L12-L2*L2)/2.0/L1/L12);
    if(isnan(fai))
    { printf("IK:fai is nan :%.2f\t%.2f\n",L1*L1+L12*L12-L2*L2,(L1*L1+L12*L12-L2*L2)/2.0/L1/L12);}
//    printf("IK:fai:%.2f\n",fai);
    ang(1)=atan2(hipx,-hipz)-fai;
    ang(2)=PI-acos((L1*L1+L2*L2-L12*L12)/2.0/L1/L2);
    return ang;
}
template <typename T>
void FSM_State_FrontJump2<T>::_SetJPosInterPts(
        const size_t & curr_iter, size_t max_iter, int leg,
        const Vec3<T> & inip, const Vec3<T> & finp){

    float a(0.f);
    float b(1.f);

    // if we're done interpolating
    if(curr_iter <= max_iter) {
        b = (float)curr_iter/(float)max_iter;
        a = 1.f - b;
    }

    // compute setpoints
    Vec3<T> inter_pos = a * inip + b * finp;

    Vec3<T> zero_vec3;
    zero_vec3.setZero();
    this->jointPDControl(leg, inter_pos, zero_vec3);
}

// template class FSM_State_BalanceStand<double>;
template class FSM_State_FrontJump2<float>;
