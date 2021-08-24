#include "DataManager.hpp"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define THIS_COM "../"

DataManager::DataManager() {
  load_control_plan(THIS_COM "config/mc_flip.dat");
  printf("[Backflip DataManager] Constructed.\n");
  dt = 0.002;
  _key_pt_step = 2;
  printf("dt: %f, step:%d\n", dt, _key_pt_step);

  _Kp.resize(12);
  _Kd.resize(12);
  _des_jpos.resize(12);
  _des_jvel.resize(12);
  _jtorque.resize(12);
  _Kp_joint.resize(3);
  _Kd_joint.resize(3);
}

void DataManager::load_control_plan(const char* filename) {
  printf("[Backflip DataManager] Loading control plan %s...\n", filename);
  FILE* f = fopen(filename, "rb");
  if (!f) {
    printf("[Backflip DataManager] Error loading control plan!\n");
    return;
  }
  fseek(f, 0, SEEK_END);
  uint64_t file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  printf("[Backflip DataManager] Allocating %ld bytes for control plan\n",
         file_size);

  plan_buffer = (float*)malloc(file_size + 1);

  if (!plan_buffer) {
    printf("[Backflip DataManager] malloc failed!\n");
    return;
  }

  uint64_t read_success = fread(plan_buffer, file_size, 1, f);
  if (!read_success) {
    printf("[Backflip DataManager] Error: fread failed.\n");
  }

  if (file_size % sizeof(float)) {
    printf(
        "[Backflip DataManager] Error: file size isn't divisible by size of "
        "float!\n");
  }

  fclose(f);

  plan_loaded = true;
  plan_timesteps = file_size / (sizeof(float) * plan_cols);
  printf("[Backflip DataManager] Done loading plan for %d timesteps\n",
         plan_timesteps);
}

/* float* DataManager::get_initial_configuration() {
  if (!plan_loaded) {
    printf(
        "[Backflip DataManager] Error: get_initial_configuration called without "
        "a plan!\n");
    return nullptr;
  }

  return plan_buffer + 3;
}
 */
float* DataManager::get_plan_at_time(int timestep) {
  if (!plan_loaded) {
    printf(
        "[Backflip DataManager] Error: get_plan_at_time called without a "
        "plan!\n");
    return nullptr;
  }

  if (timestep < 0 || timestep >= plan_timesteps) {
    printf(
        "[Backflip DataManager] Error: get_plan_at_time called for timestep %d\n"
        "\tmust be between 0 and %d\n",
        timestep, plan_timesteps - 1);
    timestep = plan_timesteps - 1;
    // return nullptr; // TODO: this should estop the robot, can't really
    // recover from this!
  }

  // if(timestep < 0) { return plan_buffer + 3; }
  // if(timestep >= plan_timesteps){ timestep = plan_timesteps-1; }

  return plan_buffer + plan_cols * timestep;
}

void DataManager::unload_control_plan() {
  free(plan_buffer);
  plan_timesteps = -1;
  plan_loaded = false;
  printf("[Backflip DataManager] Unloaded plan.\n");
}

void DataManager::FirstVisit(float _curr_time) {
    _ctrl_start_time = _curr_time;
    current_iteration = 0;
    pre_mode_count = 0;
  }

void DataManager::LastVisit() {}

bool DataManager::EndOfPhase(LegControllerData<float>* data) {
    if (_state_machine_time > (_end_time - 2. * dt)) {
        return true;
    }
    for (int leg(0); leg < 4; ++leg) {
        if(_state_machine_time>2.7 && data[leg].q[1] > 
            _q_knee_max && data[leg].qd[1] > _qdot_knee_max){
        printf("Contact detected at leg [%d] => Switch to the landing phase !!! \n", leg); 
        printf("state_machine_time: %lf \n",_state_machine_time); 
        printf("Q-Knee: %lf \n",data[leg].q[1]);
        printf("Qdot-Knee: %lf \n",data[leg].qd[1]);
        return true;
        } 
    }
    return false;
}

void DataManager::SetParameter() {
    for (int i = 0;i < 12;i++) {
        _Kp[i] = 1000;
        _Kd[i] = 5.;
    }
    //_Kp_joint = {20.0, 20.0, 20.0};
    //_Kd_joint = {2.0, 2.0, 2.0};
    _Kp_joint = {10.0, 10.0, 10.0};
    _Kd_joint = {1.0, 1.0, 1.0};
}