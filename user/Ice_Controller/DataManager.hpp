#pragma once
#include "DataManager.hpp"
#include <cppTypes.h>
#include <Dynamics/FloatingBaseModel.h>
#include <Controllers/LegController.h>

class DataManager {
 public:
  static const int plan_cols = 22;

  DataManager();
  void load_control_plan(const char *filename);
  void unload_control_plan();
  float *get_initial_configuration();
  float *get_plan_at_time(int timestep);
  void FirstVisit(float _curr_time);
  void LastVisit();
  bool EndOfPhase(LegControllerData<float>* data);
  void SetParameter();


  int plan_timesteps = -1;

 private:
  // RobotType _type;
  float *plan_buffer;
  bool plan_loaded = false;

 protected:
  DVec<float> _Kp, _Kd;
  DVec<float> _des_jpos;
  DVec<float> _des_jvel;
  DVec<float> _jtorque;

  float dt;

  std::vector<float> _Kp_joint, _Kd_joint;

  bool _b_Preparation = false;

  bool _b_set_height_target;
  float _end_time = 5.5;
  int _dim_contact;

  float _ctrl_start_time;
  float _q_knee_max = 2.0;
  float _qdot_knee_max = 2.0;

  float _state_machine_time;

  int _key_pt_step = 2;
  int current_iteration, pre_mode_count;
};