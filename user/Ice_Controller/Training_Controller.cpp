#include "Training_Controller.hpp"
// #include "BackFlipCtrl.hpp"
#include <iostream>

float pre_q[4][3];
int iter(0);

void Training_Controller::runController(){

  // std::cout << "THE RIGHT LEG CONTROLLER: " << _legController << std::endl;

  // *_done = false;


  //BF._commands = _legController->commands;
  //BF.OneStep(); //_legController->commands
  recovering._legController = _legController;
  recovering._stateEstimator = _stateEstimator;
  recovering._controlParameters = _controlParameters;
  // recovering._desiredStateCommand = _desiredStateCommand;

  guard._legController = _legController;
  //*_done = recovering._done;
  // bool TEST_ONLY_ON_SIM = true;
  //*_finishReset = true;
  // if (this->_firstrun){
  //   if (guard.initial_jpos_safe()){
  //     guard.global_safe = true;
  //     this->_firstrun = false;
  //     std::cout << " SHOW TIME ! " <<std::endl;
  //   }
  // // recovering.begin();
  
    
  // }



  // if (guard.global_safe){ //guard.jpos_safe() && guard.global_safe
  //   recovering.runtest();
  //   //std::cout << " RUNNING ~~~~~~~ " <<std::endl;

  // } else{
  //   recovering._Passive();
  // }  
  // recovering.runtest();
  // Test state estimation of leg 1;
  // int legi = 2; 
  
  //for (int i=0; i<4; i++){


  // if (iter % 500 == 0){
  //   std::cout << "J-POS LEG" << legi << "MONITORING : ";
  //   for (int j=0; j<3; j++){
  //     //if (abs(pre_q[i][j] - _legController->datas[i].q[j]) > 0.1)
  //     std::cout << _legController->datas[legi].q[j] << ", " ;
      
  //   }
  //   std::cout << std::endl;
  // }
  

  
  ++iter;

  // std::cout << "I AM RUNNING! " << iter << std::endl;

  // GET STATE INFORMATION

  for(size_t i(0); i < 4; ++i) {
        // XXX = _legController->datas[i].q;
    }

  //if(iter%200 ==0){
    //printf("value 1, 2: %f, %f\n", userParameters.testValue, userParameters.testValue2);
  //}


}
