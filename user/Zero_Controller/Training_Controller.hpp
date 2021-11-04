#ifndef TRAINING_CONTROLLER
#define TRAINING_CONTROLLER

#include <RobotController.h>
#include "TrainingUserParameters.h"

#include <BackFlipTesting.hpp>
#include <Recovering.h>
#include "SafetyCheckerPro.hpp"
#include <glob.h>

/* #include "DataReader.hpp"
#include "DataReadCtrl.hpp"
#include <Dynamics/FloatingBaseModel.h>
#include <Controllers/LegController.h>


class Training_Controller:public RobotController, DataReadCtrl<float>{
  public:
    Training_Controller():RobotController(),DataReadCtrl(_data_reader, 0.002), _jpos_ini(cheetah::num_act_joint){
    _jpos_ini.setZero();

    // _data_reader = new DataReader(this->_quadruped->_robotType, FSM_StateName::BACKFLIP);

    // Testing_BF<float> backflip_ctrl_ = new Testing_BF<float>();
    // backflip_ctrl_->SetParameter();

    }
    DataReader* _data_reader = new DataReader(this->_quadruped->_robotType, FSM_StateName::BACKFLIP);


    void OneStep(){}
    void _update_joint_command();
    virtual void StepTest(LegControllerCommand<float>* command);

    virtual ~Training_Controller(){}

    virtual void initializeController(){}
    virtual void runController();
    virtual void updateVisualization(){}
    virtual ControlParameters* getUserControlParameters() {
      return &userParameters;
    }
  protected:

    DVec<float> _jpos_ini;
    TrainingUserParameters userParameters;

    // _Kp_joint


    
};



template <typename T>
class Testing_BF{
 public:
  Testing_BF() {}
  virtual ~Testing_BF() {}

  virtual void OneStep();

 protected:
  void _update_joint_command();

}; */




class Training_Controller:public RobotController{
  public:
    // Training_Controller():RobotController(), BF(_commands), recovering(_legController, _stateEstimator), _jpos_ini(cheetah::num_act_joint){
    Training_Controller():RobotController(), recovering(), _jpos_ini(cheetah::num_act_joint){

    _jpos_ini.setZero();
    // _controlParameters->cheater_mode = 1;
    
    //(void)BF;
    // delete _data_reader;  _legController->commands
    
    }
    // LegControllerCommand<float>* _commands = _legController->commands;
    // BackFlipTesting BF;
    Recovering recovering;
    Guard guard;

    
    virtual ~Training_Controller(){}

    virtual void initializeController(){}
    virtual void runController();
    virtual void updateVisualization(){}
    virtual ControlParameters* getUserControlParameters() {
      return &userParameters;
    }
    virtual void Estop(){ recovering._Passive(); }
    // virtual void recover();

  private:
    bool _firstrun = true;

  protected:
    DVec<float> _jpos_ini;
  TrainingUserParameters userParameters;
};





#endif

