#pragma once
#include<BackFlipTesting.hpp>
#include "DataManager.hpp"

enum plan_offsets {
  q0_offset = 0,     // x, z, yaw, front hip, front knee, rear hip, rear knee
  qd0_offset = 7,    // x, z, yaw, front hip, front knee, rear hip, rear knee
  tau_offset = 14,   // front hip, front knee, rear hip, rear knee
  force_offset = 18  // front x, front z, rear x, rear z
};

typedef Eigen::Matrix<float, 7, 1> Vector7f;

class BackFlipTesting : public DataManager {
 public:
  // LegController<float>* _legController = nullptr;
  LegControllerCommand<float>* _commands = nullptr;
  
  BackFlipTesting() : DataManager(){
    //_command = command;
    //_legController = command;
  }; //LegControllerCommand<float>* command

  BackFlipTesting(LegController<float>* command) : DataManager(){

    std::cout << "~~~ THE LEG CONTROLLER GOT :  " << command << std::endl;
    //_command = command;
    //_legController = command;
  };


  void OneStep();  // LegControllerCommand<float>*
  void _update_joint_command();

  
  
};