#include "Training_Controller.hpp"
// #include "BackFlipCtrl.hpp"
#include <iostream>
#include <string>
#include <fstream>

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
  recovering._desiredStateCommand = _desiredStateCommand;

  guard._legController = _legController;
  //*_done = recovering._done;
  // bool TEST_ONLY_ON_SIM = true;
  //*_finishReset = true;

  if (this->_firstrun){
    if (guard.initial_jpos_safe()){
      guard.global_safe = true;
      this->_firstrun = false;
      std::cout << " SHOW TIME ! " <<std::endl;
    }
  // recovering.begin();

    char* homeDir = getenv("HOME");
    const char* stand_conf = "/dog_static.conf"; 
    char* stand_conf_dir_full = new char[strlen(homeDir) + strlen(stand_conf) + 1 + 1];
    strcpy(stand_conf_dir_full, homeDir);
    strcat(stand_conf_dir_full, stand_conf);
    const char* bound_conf = "/bounding.conf"; 
    char* bounding_conf_dir_full = new char[strlen(homeDir) + strlen(bound_conf) + 1 + 1];
    strcpy(bounding_conf_dir_full, homeDir);
    strcat(bounding_conf_dir_full, bound_conf);
    const char* walk_conf = "/walking.conf"; 
    char* walking_conf_dir_full = new char[strlen(homeDir) + strlen(walk_conf) + 1 + 1];
    strcpy(walking_conf_dir_full, homeDir);
    strcat(walking_conf_dir_full, walk_conf);
    const char* param_conf = "/DAM/param_opt.conf"; 
    char* param_conf_dir_full = new char[strlen(homeDir) + strlen(param_conf) + 1 + 1];
    strcpy(param_conf_dir_full, homeDir);
    strcat(param_conf_dir_full, param_conf);


    std::ifstream stand_config(stand_conf_dir_full);
    std::vector<float> data;

    std::string temp;
      while(std::getline(stand_config, temp))
        data.push_back(std::stof(temp));


    // std::ifstream myfile("/home/user/dog_static.conf");
    // std::vector<float> data;
    
    // if (!myfile.good()){
    //   std::cout << "I GUESS YOU ARE USING YOUR PC" << std::endl;
    //   std::ifstream myfile2("/home/chen/dog_static.conf");
      
    //   std::string temp;
    //   while(std::getline(myfile2, temp))
    //     data.push_back(std::stof(temp));

    // } else {
    //   std::string temp;
    //   while(std::getline(myfile, temp))
    //     data.push_back(std::stof(temp));
    // }
    
    recovering.stand_front_hip = data[0];
    recovering.stand_front_knee = data[1];
    recovering.stand_back_hip = data[2];
    recovering.stand_back_knee = data[3];

    recovering.pre_front_hip = data[4];
    recovering.pre_front_knee = data[5];
    recovering.pre_back_hip = data[6];
    recovering.pre_back_knee = data[7];

    if (data[4] == 999)
      recovering.pre1_enable = false;

    recovering.pre2_front_hip = data[8];
    recovering.pre2_front_knee = data[9];
    recovering.pre2_back_hip = data[10];
    recovering.pre2_back_knee = data[11];

    recovering.ad1 = data[12];
    recovering.ad2 = data[13];


    std::ifstream bounding_config(bounding_conf_dir_full);
    std::vector<int> data2;
    while(std::getline(bounding_config, temp))
        data2.push_back(std::stoi(temp));

    recovering.bound_length = data2[0];

    std::ifstream walking_config(walking_conf_dir_full);
    std::vector<float> data3;
    while(std::getline(walking_config, temp))
        // std::cout << "READ:::::::::::::::" << temp << std::endl;
        data3.push_back(std::stof(temp));

    recovering.param_a = data3[0];  
    recovering.param_b = data3[1];
    // these two variables would be overwritten by param_opt now

    std::ifstream param_config(param_conf_dir_full);
    std::vector<float> param_opt;
    std::string gait;
    std::string info;
    int line_counter = 0;
    while(std::getline(param_config, temp)){
      if (line_counter == 0)
        gait = temp;
      else if (line_counter >= 1 && line_counter <= 9)
        param_opt.push_back(std::stof(temp));
      else
        info = temp;
      line_counter += 1;
    }

    if (gait == "line") recovering.gait = Recovering::Gait::line;
    else if (gait == "sine") recovering.gait = Recovering::Gait::sine;
    else if (gait == "rose") recovering.gait = Recovering::Gait::rose;
    else if (gait == "triangle") recovering.gait = Recovering::Gait::triangle;
    else if (gait == "none") recovering.gait = Recovering::Gait::none;
    else std::cout << "WHAT THE F**K IS " << gait << " ???" << std::endl;

    for (unsigned int i=0; i<recovering.param_opt.size(); i++)
      recovering.param_opt[i] = param_opt[i];

    std::cout << "[LOADER] SUCCESSFULLY LOADED PARAMETERS FOR GAIT " << gait << " !" << std::endl;
    std::cout << "[LOADER] PARAMETERS: [" ;
    for (auto & p : param_opt)
      std::cout << p << "  ";
    std::cout << "]" << std::endl;
    std::cout << "[LOADER] " << info << std::endl;
  }

  if (guard.global_safe){ //guard.jpos_safe() && guard.global_safe
    recovering.runtest();
    //std::cout << " RUNNING ~~~~~~~ " <<std::endl;

  } else{
    recovering._Passive();
  }  
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
