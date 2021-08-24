//
// Created by user on 2020/3/25.
//

#ifndef CHEETAH_SOFTWARE_FSM_STATE_FRONTJUMP2_H
#define CHEETAH_SOFTWARE_FSM_STATE_FRONTJUMP2_H
#include "FSM_State.h"

template<typename T> class WBC_Ctrl;
template<typename T> class LocomotionCtrlData;

/**
 *
 */
template <typename T>
class FSM_State_FrontJump2 : public FSM_State<T> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FSM_State_FrontJump2(ControlFSMData<T>* _controlFSMData);

    // Behavior to be carried out when entering a state
    void onEnter();// override;

    // Run the normal behavior for the state
    void run();

    // Checks for any transition triggers
    FSM_StateName checkTransition();

    // Manages state specific transitions
    TransitionData<T> transition();

    // Behavior to be carried out when exiting a state
    void onExit();


private:
    // Keep track of the control iterations
    int _iter = 0;

    // Parses contact specific controls to the leg controller
    void BalanceStandStep();
    void _SetJPosInterPts(
            const size_t & curr_iter, size_t max_iter, int leg,
            const Vec3<T> & inip, const Vec3<T> & finp);
    Vec3<T> InverseKinematics(Vec3<T> pos);
    WBC_Ctrl<T> * _wbc_ctrl;
    LocomotionCtrlData<T> * _wbc_data;

    T last_height_command = 0;

    Vec3<T> _ini_body_pos;
    Vec3<T> _ini_body_ori_rpy;
    Mat3 <T> kpMat,kdMat;
    T _body_weight;
    int _count;
    bool enter_once;
    bool enter_once_withdraw;
    Vec3<T> initial_jpos[4];
    Vec3<T> initial_Ppos[4];
    Vec3<T> ini,fin,ff_force;
    bool enter_once_2,enter_once_3;

};
#endif //CHEETAH_SOFTWARE_FSM_STATE_FRONTJUMP2_H
