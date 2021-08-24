//
// Created by user on 2020/3/24.
//

#ifndef CHEETAH_SOFTWARE_FRONTFLIPUPCTRL_H
#define CHEETAH_SOFTWARE_FRONTFLIPUPCTRL_H
#include "DataReader.hpp"
#include "DataReadCtrl.hpp"
#include <Dynamics/FloatingBaseModel.h>
#include <Controllers/LegController.h>

template <typename T>
class FrontFlipCtrl : public DataReadCtrl<T> {
public:
    FrontFlipCtrl(DataReader*, float _dt);
    virtual ~FrontFlipCtrl();

    virtual void OneStep(float _curr_time, bool b_preparation, LegControllerCommand<T>* command);

protected:
    void _update_joint_command();
};

#endif //CHEETAH_SOFTWARE_FRONTFLIPUPCTRL_H
