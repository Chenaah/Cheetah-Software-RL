#include <Recovering.h>
// #include <pthread.h>
// #include <iostream>
// #include <unistd.h>
// #include "InverseKinematics.h"


Recovering::Recovering(LegController<float>* legController, StateEstimatorContainer<float>* stateEstimator){

    _legController = legController;
    _stateEstimator = stateEstimator;
}

Recovering::Recovering(){
    // goal configuration
    // Folding
    fold_jpos[0] << -0.0f, -1.0f, 2.0f;  // V2: SHOULD NOT BE GREATER THAN 2.0 ;   the third joint shouldn't be greater than 2.5 after adding the feet
    fold_jpos[1] << 0.0f, -1.0f, 2.0f;
    fold_jpos[2] << -0.0f, -1.4f, 2.0f;
    fold_jpos[3] << 0.0f, -1.4f, 2.0f;
    // Stand Up
    // for(size_t i(0); i<4; ++i){
    //     stand_jpos[i] << 0.f, -.8f, 1.6f;
    // }
    
    // Rolling
    rolling_jpos[0] << 1.5f, -1.6f, 2.77f;
    rolling_jpos[1] << 1.3f, -3.1f, 2.77f;
    rolling_jpos[2] << 1.5f, -1.6f, 2.77f;
    rolling_jpos[3] << 1.3f, -3.1f, 2.77f;

    f_ff << 0.f, 0.f, -25.f;
    zero_vec3 << 0.f, 0.f, -0.f;


    front_offset << 0.f, front_hip_offset, 0.f;
    back_offset << 0.f, back_hip_offset, 0.f;
    backl_offset <<  0.f, back_hip_offset - fix_leg7, 0.f;


    a_dim = 12;
    s_dim = 48;

    LoadOnnxModel();

    // std::cout << "0 !!!!!!!!!!!!!!!!!!!!!! HERE I AM. " << std::endl;

    
    //********* Read model
    // CHANGE TO LOAD PYTORCH MODEL !
    // graph = TF_NewGraph();
    // status = TF_NewStatus();
    // SessionOpts = TF_NewSessionOptions();
    // RunOpts = NULL;

    // char* homeDir = getenv("HOME");
    // glob_t globbuf;
    // const char* dam_folder = "/DAM*/"; 
    // char* dam_dir_full = new char[strlen(homeDir) + strlen(dam_folder) + 1 + 1];
    // strcpy(dam_dir_full, homeDir);
    // strcat(dam_dir_full, dam_folder);
    // glob(dam_dir_full, 0, NULL, &globbuf);
    // assert(globbuf.gl_pathc == 1);
    // DAM_path = globbuf.gl_pathv[0];
    // std::cout << "Deterministic Actro Model " << DAM_path << " FOUND!" <<std::endl;
    // const char* checking_file = "saved_model.pb"; // only used for checking if RL agent is enabled
    // char* checking_file_path = new char[strlen(DAM_path) + strlen(checking_file) + 1 + 1];
    // strcpy(checking_file_path, DAM_path);
    // strcat(checking_file_path, checking_file);

    // std::ifstream model_exist(checking_file_path);
    // if (model_exist.good()){
    //     std::cout << "TENSORFLOW MODEL FOUND! RL AGENT IS ENABLED. " << std::endl;
    //     agent_enable = true;

    //     const char* tags = "serve"; 
    //     int ntags = 1;
    //     sess = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, DAM_path, &tags, ntags, graph, NULL, status);
        
    //     if(TF_GetCode(status) == TF_OK)
    //         printf("TF_LoadSessionFromSavedModel OK\n");
    //     else
    //         printf("%s",TF_Message(status));

    //     //****** Get input tensor
    //     input_op = TF_Output{TF_GraphOperationByName(graph, "serving_default_input_1"), 0};
    //     if (input_op.oper == nullptr) {
    //         std::cout << "ERROR: Failed TF_GraphOperationByName serving_default_input_1" << std::endl;
    //     } else {
    //         printf("TF_GraphOperationByName serving_default_input_1 is OK\n");
    //     }
        
    //     //********* Get Output tensor
    //     out_op = TF_Output{TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};

    //     if(out_op.oper == nullptr)
    //         printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    //     else
    //         printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

    //     //********* Allocate data for inputs & outputs

    //     const std::vector<std::int64_t> input_dims = {1, s_dim};
    //     std::vector<float> input_vals(s_dim, 0.5);
    //     input_tensor = CreateTensor(TF_FLOAT, input_dims, input_vals);
    //     printf("Allocated data ~ \n");

    //     output_tensor = nullptr;


    // } else {
    //     agent_enable = false;
    //     std::cout << "TENSORFLOW MODEL NOT FOUND! RL AGENT IS DISABLED. " << std::endl;

    // }

    action.resize(a_dim);
    state.resize(s_dim);
    observation.resize(s_dim);
    state_sum = std::vector<float>(s_dim, 0);

    leg_offsets << 0.f, 0.f, 0.f, 0.f;
    for (int leg = 0; leg < 4; leg++)
      pos_impl[leg] << 0.f, 0.f, 0.f;

    #if DEBUG
    socket.bind("tcp://*:5555");
    #endif


    // open the connection
    std::cout << "Connecting to hello world server…" << std::endl;
    // socket.connect("tcp://localhost:4555");
    socket.connect("tcp://10.0.0.3:4555");
    // int request_nbr;

    // std::cout << std::setprecision(6) << std::fixed;


    // TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
    // std::cout << "=================== TEST =================" << std::endl;
    // Eigen::Vector4d test4d; test4d << 1, -2, 3, 4;
    // Eigen::Vector3d test3d; test3d << 5,-6,997;
    // Eigen::Vector3d testresult;
    // testresult = quat_rotate_inverse(test4d, test3d);
    // std::cout << testresult << std::endl;
    // std::cout << "=================== TEST =================" << std::endl;




}

void Recovering::runtest() {

    // std::cout << "TESTING!! PHASE: "<< _phase << "STEPS: " << _state_iter << std::endl;
    // _done = true;

    body_height = _stateEstimator->getResult().position[2];
    // std::cout << "HEIGHT: "<< body_height << std::endl;

    // if (!isSafe()){
    //   _phase = -1;
    //   std::cout << "SAFE CHECK FAIL! " << std::endl;
    // }
    // std::cout << "PHASE: " << _phase << std::endl;


    // #if DEBUG
    // wait_for_simulation_state();
    // #endif
    _process_remote_controller_signal(_state_iter - _motion_start_iter);


    switch(_phase){

      case -1:
          _Passive();
          break;
      case 0:
          _done = false;
          _FoldLegs(_state_iter - _motion_start_iter);  // _motion_start_iter = _state_iter + 1;
          break;
      case 1:
          _JustStandUp(_state_iter - _motion_start_iter);
          break;
    //   case 2:
    //       _Prepare(_state_iter - _motion_start_iter);
    //       break;
    //   case 3:
    //       _Prepare_2(_state_iter - _motion_start_iter);
    //       break;
    //   case 4:
    //       _RearLegsActions(_state_iter - _motion_start_iter);
    //       break;
      case 5:
          _Walk(_state_iter - _motion_start_iter);
          break;
    //   case 6:
    //       _SettleDown(_state_iter - _motion_start_iter);
    //       break;
    //   case 7:
    //       _InverseRearLegsActions(_state_iter - _motion_start_iter);
    //       break;
    //   case 8:
    //       _InversePrepare_2(_state_iter - _motion_start_iter);
    //       break;
    //   case 9:
    //       _InverseJustStandUp(_state_iter - _motion_start_iter);
    //       break;
    //   case 10:
    //       _PushUp(_state_iter - _motion_start_iter);
    //       break;
    //   case 11:
    //       _Bound(_state_iter - _motion_start_iter);
    //       break;
    //   case 12:
    //       _BoundToStand(_state_iter - _motion_start_iter);
    //       break;
    //   case 13:
    //       _ClimbPre(_state_iter - _motion_start_iter);
    //       break;
    //   case 14:
    //       _ClimbPre1(_state_iter - _motion_start_iter);
    //       break;
    //   case 15:
    //       _Climb(_state_iter - _motion_start_iter);
    //       break;
    //   case 16:
    //       _Climb1(_state_iter - _motion_start_iter);
    //       break;
    //   case 17:
    //       _Climb2(_state_iter - _motion_start_iter);
    //       break;
    //   case 18:
    //       _Climb3(_state_iter - _motion_start_iter);
    //       break;
    //   case 19:
    //       _Climb4(_state_iter - _motion_start_iter);
    //       break;
    //   case 20:
    //       _Climb5(_state_iter - _motion_start_iter);
    //       break;
    //   case 21:
    //       _Pull(_state_iter - _motion_start_iter);
    //       break;
    //   case 22:
    //       _Pull1(_state_iter - _motion_start_iter);
    //       break;
    //   case 23:
    //       _Pull2(_state_iter - _motion_start_iter);
    //       break;
    //   case 24:
    //       _Pull3(_state_iter - _motion_start_iter);
    //       break;
    //   case 25:
    //       _RecoverOnTable(_state_iter - _motion_start_iter);
    //       break;

    //   case 26:
    //       _ClimbM1(_state_iter - _motion_start_iter);
    //       break;
    //   case 27:
    //       _ClimbM1_1(_state_iter - _motion_start_iter);
    //       break;
    //   case 28:
    //       _ClimbM1_2(_state_iter - _motion_start_iter);
    //       break;
    //   case 29:
    //       _ClimbM1_3(_state_iter - _motion_start_iter);
    //       break;
    //   case 30:
    //       _ClimbM1_4(_state_iter - _motion_start_iter);
    //       break;
    //   case 31:
    //       _ClimbM2(_state_iter - _motion_start_iter);
    //       break;
    //   case 32:
    //       _ClimbM2_1(_state_iter - _motion_start_iter);
    //       break;
    //   case 33:
    //       _FastStand(_state_iter - _motion_start_iter);
    //       break;
    //   case 34:
    //       _ClimbLA0(_state_iter - _motion_start_iter);
    //       break;
    //   case 35:
    //       _ClimbLA1(_state_iter - _motion_start_iter);
    //       break;
    //   case 36:
    //       _ClimbLA2(_state_iter - _motion_start_iter);
    //       break;
    //   case 37:
    //       _PushTest(_state_iter - _motion_start_iter);
    //       break;
    //   case 38:
    //       _PushTest1(_state_iter - _motion_start_iter);
    //       break;
    //   case 39:
    //       _WalkTest(_state_iter - _motion_start_iter);
    //       break;
    //   case 40:
    //       _WalkTest1(_state_iter - _motion_start_iter);
    //       break;
      
    }

    #if DEBUG
    // send the reply to the client
    send_action_to_simulator();
    #endif
    ++_state_iter;
}

void Recovering::recover(){

    this->begin();
}


void Recovering::begin(){

    // initial configuration, position
    for(size_t i(0); i < 4; ++i) {
        initial_jpos[i] = _legController->datas[i].q;
    }
    body_height = this->_stateEstimator->getResult().position[2];
    max_body_height = this->_stateEstimator->getResult().position[2];

    _flag = FoldLegs;
    if( !_UpsideDown() ) { // Proper orientation
    if (  (0.2 < body_height) && (body_height < 0.45) ){
        printf("[Recovery Balance] body height is %f; Stand Up \n", body_height);
        _flag = StandUp;
    }else{
        printf("[Recovery Balance] body height is %f; Folding legs \n", body_height);
    }
    }else{
        printf("[Recovery Balance] UpsideDown (%d) \n", _UpsideDown() );
    }
    _motion_start_iter = 0;

    
    // py::object test_sys = py::module::import("sys");
    // std::cout << "!!!!!!!!!!!   SYS.PATH FROM PYTHON :" <<  test_sys.attr("path")();
    // py::list sys_list = test_sys.attr("path");
    // print_list(sys_list);
}


bool Recovering::_UpsideDown(){
    if(this->_stateEstimator->getResult().rBody(2,2) < 0){
        return true;
    }
    return false;
}


void Recovering::_SetJPosInterPts(
    const size_t & curr_iter, size_t max_iter, int leg, 
    const Vec3<float> & ini, const Vec3<float> & fin){

    float a(0.f);
    float b(1.f);

    // if we're done interpolating
    if(curr_iter <= max_iter) {
      b = (float)curr_iter/(float)max_iter;
      a = 1.f - b;
    }

    // compute setpoints
    Vec3<float> inter_pos = a * ini + b * fin;

    // do control
    this->jointPDControl(leg, inter_pos, zero_vec3);

    //if(curr_iter == 0){ 
      //printf("flag:%d, curr iter: %lu, state iter: %llu, motion start iter: %d\n", 
        //_flag, curr_iter, _state_iter, _motion_start_iter); 
      //printf("inter pos: %f, %f, %f\n", inter_pos[0], inter_pos[1], inter_pos[2]);
    //}
    //if(curr_iter == max_iter){ 
      //printf("flag:%d, curr iter: %lu, state iter: %llu, motion start iter: %d\n", 
        //_flag, curr_iter, _state_iter, _motion_start_iter); 
      //printf("inter pos: %f, %f, %f\n", inter_pos[0], inter_pos[1], inter_pos[2]);
    //}
}
void Recovering::jointPDControl(
    int leg, Vec3<float> qDes, Vec3<float> qdDes) {

    kpMat << 80, 0, 0, 0, 80, 0, 0, 0, 80;
    // kpMat << 120, 0, 0, 0, 120, 0, 0, 0, 120;
    // kdMat << 6, 0, 0, 0, 6, 0, 0, 0, 6;
    kdMat << 10, 0, 0, 0, 10, 0, 0, 0, 10;
    // kdMat << 1, 0, 0, 0, 1, 0, 0, 0, 1;

    this->_legController->commands[leg].kpJoint = kpMat;
    this->_legController->commands[leg].kdJoint = kdMat;

    // if (qDes[0]!=0){
    //    std::cout << "ACTIONS SENT TO _legController : " << qDes[0] << ", "<< qDes[1]<<", "<< qDes[2] <<std::endl;

    // }
    this->_legController->commands[leg].qDes = qDes;
    this->_legController->commands[leg].qdDes = qdDes;
}

void Recovering::jointPDControl(
    int leg, Vec3<float> qDes, Vec3<float> qdDes, Mat3<float> kpMat_arg, Mat3<float> kdMat_arg) {

    // kpMat << 80, 0, 0, 0, 80, 0, 0, 0, 80;
    // kdMat << 10, 0, 0, 0, 10, 0, 0, 0, 10;

    this->_legController->commands[leg].kpJoint = kpMat_arg;
    this->_legController->commands[leg].kdJoint = kdMat_arg;


    this->_legController->commands[leg].qDes = qDes;
    this->_legController->commands[leg].qdDes = qdDes;
}

void Recovering::_RollOver(const int & curr_iter){

    // std::cout << "_RollOver IS CALLED ! " << std::endl;

  for(size_t i(0); i<4; ++i){
    _SetJPosInterPts(curr_iter, rollover_ramp_iter, i, 
        initial_jpos[i], rolling_jpos[i]);
  }

  if(curr_iter > rollover_ramp_iter + rollover_settle_iter){
    _flag = FoldLegs;
    for(size_t i(0); i<4; ++i) initial_jpos[i] = rolling_jpos[i];
    _motion_start_iter = _state_iter+1;
  }
}




void Recovering::_FoldLegs(const int & curr_iter){
  if (curr_iter==0){
    for(size_t i(0); i < 4; ++i) {
      initial_jpos[i] = this->_legController->datas[i].q;
      
    }

    // for(size_t i(0); i < 2; ++i)
    //     initial_jpos[i] = initial_jpos[i] - front_offset;
    // initial_jpos[3] = initial_jpos[3] - backl_offset;
    // initial_jpos[4] = initial_jpos[4] - back_offset;


    std::cout<< "====================== FOLD LEG BEGINS! ======================" << curr_iter <<std::endl;

    // t_start = std::chrono::high_resolution_clock::now();
  }

  // t_end = std::chrono::high_resolution_clock::now();
  // double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  // std::cout << "THE TIME PERIOD IS " <<  elapsed_time_ms << std::endl;
  // t_start = t_end;      // 2ms !

    // _update_rpy();
    // pitch = rpy[1];
    // std::cout << " THE PITCH (ALPHA) IS " << pitch << std::endl;
    // #if DEBUG
    // std::cout << "   --  THE ORIENTATION IS " << bullet_orientation[0] << ", "<< bullet_orientation[1]<< ", "<< bullet_orientation[2]<< ", "<< bullet_orientation[3] << std::endl;
    // #else
    // std::cout << "   --  THE ORIENTATION IS " << _stateEstimator->getResult().orientation[0] << ", "<< _stateEstimator->getResult().orientation[1]<< ", "<< _stateEstimator->getResult().orientation[2]<< ", "<< _stateEstimator->getResult().orientation[3] << std::endl;
    // #endif

    // return;

  // if (curr_iter % 40 == 0){

  //   for (int j=0; j<3; j++){
  //       std::cout << "JOINT " << j << " : " ;
  //       for (int l=0; l<4; l++){
  //           std::cout << _legController->datas[l].q[j] << ", ";
  //       }
  //       std::cout << std::endl;
  //     }

  //     std::cout << "THE PITCH IS " << _stateEstimator->getResult().rpy[1]  << std::endl;
  //     std::cout << "THE ORIENTATION IS " << _stateEstimator->getResult().orientation[0] << ", "<< _stateEstimator->getResult().orientation[1]<< ", "<< _stateEstimator->getResult().orientation[2]<< ", "<< _stateEstimator->getResult().orientation[3] << std::endl;
  //     Eigen::Quaterniond _orientation(_stateEstimator->getResult().orientation[0], _stateEstimator->getResult().orientation[1], _stateEstimator->getResult().orientation[2], _stateEstimator->getResult().orientation[3]);

  //     Eigen::Vector3d euler = _orientation.matrix().eulerAngles(2,1,0);
  //     std::cout << "THE EULER ANGLES ARE " << euler.x() << ", " << euler.y() << ", " << euler.z() << std::endl;

  //     std::cout << "THE OMEGA IS " << _stateEstimator->getResult().omegaBody[0] << ", "<< _stateEstimator->getResult().omegaBody[1]<< ", "<< _stateEstimator->getResult().omegaBody[2]<< std::endl;


  // }

    // rc_mode = this->_desiredStateCommand->rcCommand->mode;
    // std::cout << " TEST, MODE: " << rc_mode << "  VALUE: " << this->_desiredStateCommand->rcCommand->v_des[0] << ", " 
    //                                                     << this->_desiredStateCommand->rcCommand->v_des[1] << ", " 
    //                                                     << this->_desiredStateCommand->rcCommand->omega_des[0] << ", " 
    //                                                     << this->_desiredStateCommand->rcCommand->omega_des[1] << ", " 
    //                                                     << this->_desiredStateCommand->rcCommand->rpy_des[0] << ", " 
    //                                                     << this->_desiredStateCommand->rcCommand->rpy_des[1] << ", " 
    //                                                     << this->_desiredStateCommand->rcCommand->rpy_des[2] 
    //                                                     << std::endl;
    // std::cout << this->_desiredStateCommand->returnBackGamepad()->leftTriggerAnalog << std::endl;
  

  // return;



  // std::cout << "[DEBUG] _FoldLegs IS CALLED ! (CURR_ITER " << curr_iter << ")" << std::endl;

  for(size_t i(0); i<4; ++i){
    _SetJPosInterPts(curr_iter, fold_ramp_iter, i, 
        initial_jpos[i], fold_jpos[i]);
  }

//   for(size_t i(0); i<4; ++i){
//     _Step(curr_iter, fold_ramp_iter, i, 
//         initial_jpos[i], fold_jpos[i]);
//   }



  if(curr_iter >= fold_ramp_iter + 100){
    if(_UpsideDown()){
      _flag = RollOver;
      for(size_t i(0); i<4; ++i) initial_jpos[i] = fold_jpos[i];
    }else{
      _flag = StandUp;
      for(size_t i(0); i<4; ++i) initial_jpos[i] = fold_jpos[i];
    }
    // #if !DEBUG
    // _phase = 11;  // DEVELOPING
    // #else
    // // _phase = 1;
    // if (just_bounding)
    //     _phase = 11;
    // else
    //     if (fast_stand)
    //         _phase = 33;  // FAST STANDING UP
    //     else
    //         _phase = 1;  // FAST STANDING UP

    //     // _phase = 37;  // PUSHINGTEST

    // #endif
    _phase = 1;
    _motion_start_iter = _state_iter + 1;
  }
}

void Recovering::_JustStandUp(const int & curr_iter){

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
            initial_jpos[i] = this->_legController->datas[i].q;

        stand_jpos[1] << default_dof_pos[0], -(default_dof_pos[1] + front_hip_offset), -default_dof_pos[2];  // FL
        stand_jpos[0] << default_dof_pos[3], -(default_dof_pos[4] + front_hip_offset), -default_dof_pos[5];  // FR
        stand_jpos[3] << default_dof_pos[6], -(default_dof_pos[7] + front_hip_offset), -default_dof_pos[8];  // RL
        stand_jpos[2] << default_dof_pos[9], -(default_dof_pos[10] + front_hip_offset), -default_dof_pos[11];  // RR

    }

    // _update_state();

    for(size_t leg(0); leg<4; ++leg){
        _SetJPosInterPts(curr_iter, standup_ramp_iter, 
                         leg, initial_jpos[leg], stand_jpos[leg]);
    }       

    if(curr_iter >= standup_ramp_iter+100){
        _phase = 5;
        _motion_start_iter = _state_iter+1;
    } 

}



void* Recovering::_Test(void* args){
    (void) args;
    _update_action();
    return 0;}

// static void* _update_action_worker(void* args){
//     ((Recovering*)args)->_update_action();
//     return 0;
// }

#if DEBUG
void Recovering::receive_state_from_bullet(const lcm::ReceiveBuffer* rbuf,
                                        const std::string& chan, 
                                        const state_estimator_lcmt* msg){

    (void)rbuf;
    (void)chan;

    for(int i=0; i<4; i++)
        bullet_orientation[i] = msg->quat[i];

    for(int i=0; i<3; i++)
        bullet_omegaWorld[i] = msg->omegaWorld[i];

}

void Recovering::receive_leg_state_from_bullet(const lcm::ReceiveBuffer* rbuf,
                                        const std::string& chan, 
                                        const leg_control_data_lcmt* msg){
    (void)rbuf;
    (void)chan;

    bullet_q.clear();
    for(int i=0; i<12; i++)
        bullet_q.push_back(msg->q[i]);
    // std::cout << "[DEBUG] POSITION OF JOIN[0][1] and JOIN[1][1] FROM BULLET  " << bullet_q[1] << ", " <<  bullet_q[4] << std::endl;
}

#endif

void Recovering::_Walk(const int & curr_iter){

    

    #if DEBUG
    // state_estimator_lcmt debug_data;
    // debug_data.p[0] = curr_iter;
    // lcm.publish("debug_channel", &debug_data);
    // debug_info_stream << "SUB STEP " << curr_iter << "\n";
    std::cout << "[DEBUG] SUB STEP " << curr_iter << "\n";
    #endif

    if (curr_iter == 0){
        std::cout << "[WALKING] INITIAL STATE: [" << state[0] << ", " << state[1] << ", " << state[2] << ", " << state[3] << ", " << state[4] << ", " << state[5] << ", " << state[6] << ", " << state[7] << ", " << state[8] << ", " << state[9] << ", " << state[10] << ", " << state[11] << ", " << state[12] << ", " << state[13] << "]" << std::endl;
        // std::cout << "QUAT: [" << bullet_orientation[0] << ", " << bullet_orientation[1] << ", "<< bullet_orientation[2] << ", "<< bullet_orientation[3] << "]" << std::endl;
    }

    _update_state();
    _update_state_buffer();
    
    if (curr_iter % num_sub_steps == 0){
        
        for(size_t i(0); i < 4; ++i)
            initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

        // _update_state();
        _update_observation();

        if (verbose){
            std::cout << std::endl << "============================================================================" << std::endl << std::endl;
            std::cout << "[AGENT] STATE: ";
            for(int i=0; i<s_dim; i++){
                std::cout << state[i];
                if (i != s_dim-1)
                    std::cout << ", ";
            }
            std::cout << std::endl << std::endl;
        }


        // if (curr_iter != 0 && !_done && curr_iter <= 10000)
        //     _update_reward();
        // else if (curr_iter != 0)
        //     _log_return();
        
        #if DEBUG
        // debug_info_stream << "STATE: [" << state[0] << ", " << state[1] << ", " << state[2] << ", " << state[3] << ", " << state[4] << ", " << state[5] << ", \n        " << 
        //                                    state[6] << ", " << state[7] << ", " << state[8] << ", " << state[9] << ", \n        " << 
        //                                    state[10] << ", " << state[11] << ", " << state[12] << ", " << state[13] << "]" << std::endl;
        #endif
        
        // _update_action();
        _update_action_remote();

        if (verbose){
            std::cout << "[AGENT] ACTION: ";
            for(int i=0; i<a_dim; i++){
                std::cout << action[i];
                if (i != a_dim-1)
                    std::cout << ", ";
            }
            std::cout << std::endl << std::endl;
        }

        
        #if DEBUG
        // debug_info_stream << "AGENT ACTION: [" <<  action[0] << ", " <<  action[1] << ", " <<  action[2] << ", "<<  action[3] << ", "<<  action[4] << ", "<<  action[5]  << "] " << "\n";
        #endif

        // _update_leg_offsets();  // leg_offsets would be overwritten

    }

    pos_impl[1] << action[0]*action_scale + default_dof_pos[0], -(action[1]*action_scale + default_dof_pos[1]), -(action[2]*action_scale + default_dof_pos[2]);
    pos_impl[0] << action[3]*action_scale + default_dof_pos[3], -(action[4]*action_scale + default_dof_pos[4]), -(action[5]*action_scale + default_dof_pos[5]);
    pos_impl[3] << action[6]*action_scale + default_dof_pos[6], -(action[7]*action_scale + default_dof_pos[7]), -(action[8]*action_scale + default_dof_pos[8]);
    pos_impl[2] << action[9]*action_scale + default_dof_pos[9], -(action[10]*action_scale + default_dof_pos[10]), -(action[11]*action_scale + default_dof_pos[11]);

    if (verbose){
        std::cout << "[CONTROL] JOINT BEFORE CLIPPING: [";
        for(int i=0; i<4; i++){
            for(int j=0; j<3; j++){
                std::cout << pos_impl[i][j];
                // if (i != 3 && j != 2)
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl << std::endl;
    }

    joint_pos_clip();
    // pos_impl[0][0] = 0;
    // pos_impl[1][0] = 0;
    // pos_impl[2][0] = 0;
    // pos_impl[3][0] = 0;
    // assert((unsigned int)pos_impl[2][0] == 0 && (unsigned int)pos_impl[3][0] == 0);
    // std::cout << "[CONTROL] FINAL ABAD: [" <<  pos_impl[0][0] << ", " <<  pos_impl[1][0] << ", " <<  pos_impl[2][0] << ", "<<  pos_impl[3][0] << "] " << std::endl;

    if (verbose){
        std::cout << "[CONTROL] JOINT AFTER CLIPPING: [";
        for(int i=0; i<4; i++){
            for(int j=0; j<3; j++){
                std::cout << pos_impl[i][j];
                // if (i != 3 && j != 2)
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl << std::endl;
    }

    for (size_t leg(0); leg<4; ++leg)
        for (size_t j(0); j<3; ++j)
            assert(!std::isnan(pos_impl[leg][j]));

    // if (curr_iter % 50){
    //     std::cout << "[CONTROL] FINAL ARM ACTION (OBJ): [" <<  pos_impl[0][0] << ", " <<  pos_impl[0][1] << ", " <<  pos_impl[1][0] << ", "<<  pos_impl[1][1] << "] " << std::endl;
    //     std::cout << "[CONTROL] FINAL LEG ACTION: [" <<  pos_impl[2][1] << ", " <<  pos_impl[2][2] << ", " <<  pos_impl[3][1] << ", "<<  pos_impl[3][2] << "] " << std::endl;
    // }


    #if DEBUG
    // debug_info_stream << "FINAL ACTION: [" <<  pos_impl[2][1] << ", " <<  pos_impl[2][2] << ", " <<  pos_impl[3][1] << ", "<<  pos_impl[3][2] << "] " << "\n";
    // std::cout << "[DEBUG] LEG FINAL ACTION: [" <<  pos_impl[2][1] << ", " <<  pos_impl[2][2] << ", " <<  pos_impl[3][1] << ", "<<  pos_impl[3][2] << "] " << "\n";
    std::cout << "[DEBUG] ARM FINAL ACTION: [" <<  pos_impl[0][0] << ", " <<  pos_impl[0][1] << ", " <<  pos_impl[1][0] << ", "<<  pos_impl[1][1] << "] " << "\n";
    #endif

    // for(size_t leg(0); leg<4; ++leg){
    //     pos_impl[leg][2] *= 1.05;
    // }

    if (_within_limits()){  // SECURITY CHECK:
    // if (true){  // SECURITY CHECK:

        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, 9, 
            leg, initial_jpos[leg], pos_impl[leg]);
        }

        // TESTING BROKEN LEG !!!
        // Mat3<float> kpMat_broken;
        // Mat3<float> kdMat_broken;
        // // if pos_impl[3][2] <
        // kpMat_broken << 80, 0, 0, 0, 80, 0, 0, 0, 0;
        // kdMat_broken << 10, 0, 0, 0, 10, 0, 0, 0, 0;
        // _Step(curr_iter, 9, 
        //     3, initial_jpos[3], pos_impl[3], kpMat_broken, kdMat_broken);

    } else {

        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";

    }
    

    if (stopping && _has_stopped()){
        _phase = 6;
        _motion_start_iter = _state_iter + 1;
    }

    // if(curr_iter > 11000 + additional_step || _done){

    //     #if DEBUG
    //     // debug_file.open("states_actions.txt", std::ios_base::app);

    //     // if (!debug_file)
    //     //   std::cout << "can't open output file" << std::endl;
        
    //     // debug_file << debug_info_stream.str();
    //     // debug_file.flush();

    //     #endif
    //     _phase = -5;

    // }

}

bool Recovering::_has_stopped(){
    bool actions_stop = true;
    for (auto & a : action)
        if (fabs(a) > 0.001)
            actions_stop = false;
    if (fabs(param_c_buffered) > 0.001)
        actions_stop = false;
    if (fabs(param_d_buffered) > 0.001)
        actions_stop = false;
    if (fabs(param_a) > 0.001)
        actions_stop = false;
    // if (!actions_stop)
    //     std::cout << "[STOP] STOPPING CHECK DIDN'T PASS ..." << std::endl;
    return actions_stop;

}

void Recovering::_update_state(){

    // FRONT
    // 1 0  RIGHT
    // 3 2
    // BACK

    // FL, FR, BL, BR --> 1, 0, 3, 2

    base_quat << _stateEstimator->getResult().orientation[1], 
                 _stateEstimator->getResult().orientation[2], 
                 _stateEstimator->getResult().orientation[3], 
                 _stateEstimator->getResult().orientation[0];
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec);

    std::cout<< std::endl << "[DEBUG] X VELOCITY: " << _stateEstimator->getResult().vBody[0] << std::endl<< std::endl;

    state[0] = (_stateEstimator->getResult().vBody[0]+ 0.2) * 1.5 * lin_vel_scale ;
    state[1] = _stateEstimator->getResult().vBody[1] * lin_vel_scale;
    state[2] = _stateEstimator->getResult().vBody[2] * lin_vel_scale;

    state[3] = _stateEstimator->getResult().omegaBody[0] * ang_vel_scale;
    state[4] = _stateEstimator->getResult().omegaBody[1] * ang_vel_scale;
    state[5] = _stateEstimator->getResult().omegaBody[2] * ang_vel_scale;

    state[6] = projected_gravity[0];
    state[7] = projected_gravity[1];
    state[8] = projected_gravity[2];

    state[9] = 0.5*2;
    state[10] = 0*2;
    state[11] = 0*0.25;

    state[12] = (_legController->datas[1].q[0] - default_dof_pos[0])*dof_pos_scale;
    state[13] = (-(_legController->datas[1].q[1]- front_hip_offset) - default_dof_pos[1])*dof_pos_scale;
    state[14] = (-_legController->datas[1].q[2] - default_dof_pos[2])*dof_pos_scale;
    state[15] = (_legController->datas[0].q[0] - default_dof_pos[3])*dof_pos_scale;
    state[16] = (-(_legController->datas[0].q[1]- front_hip_offset) - default_dof_pos[4])*dof_pos_scale;
    state[17] = (-_legController->datas[0].q[2] - default_dof_pos[5])*dof_pos_scale;
    state[18] = (_legController->datas[3].q[0] - default_dof_pos[6])*dof_pos_scale;
    state[19] = (-(_legController->datas[3].q[1] - back_hip_offset) - default_dof_pos[7])*dof_pos_scale;
    state[20] = (-_legController->datas[3].q[2] - default_dof_pos[8])*dof_pos_scale;
    state[21] = (_legController->datas[2].q[0] - default_dof_pos[9])*dof_pos_scale;
    state[22] = (-(_legController->datas[2].q[1] - back_hip_offset) - default_dof_pos[10])*dof_pos_scale;
    state[23] = (-_legController->datas[2].q[2] - default_dof_pos[11])*dof_pos_scale;

    state[24] = _legController->datas[1].v[0]*dof_vel_scale;
    state[25] = -_legController->datas[1].v[1]*dof_vel_scale;
    state[26] = -_legController->datas[1].v[2]*dof_vel_scale;
    state[27] = _legController->datas[0].v[0]*dof_vel_scale;
    state[28] = -_legController->datas[0].v[1]*dof_vel_scale;
    state[29] = -_legController->datas[0].v[2]*dof_vel_scale;
    state[30] = _legController->datas[3].v[0]*dof_vel_scale;
    state[31] = -_legController->datas[3].v[1]*dof_vel_scale;
    state[32] = -_legController->datas[3].v[2]*dof_vel_scale;
    state[33] = _legController->datas[2].v[0]*dof_vel_scale;
    state[34] = -_legController->datas[2].v[1]*dof_vel_scale;
    state[35] = -_legController->datas[2].v[2]*dof_vel_scale;

    for(int i=0; i<12; i++)
        state[36+i] = action[i];

    // clipping
    for(int i=0; i<48; i++)
        state[i] = std::max(std::min(state[i], clip_obs), -clip_obs);

    // if(fabs(p_error) > 0.65){
    //     _done = true;
    //     std::cout << "[DONE] I AM DEAD !!!" << std::endl;
    // }

    
    

    
}


void Recovering::_update_action(){

    for (size_t idx = 0; idx < 48; idx++)
        pdData[idx] = observation[idx];

    t_start = std::chrono::high_resolution_clock::now();

    OnnxInference();

    t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "THE TIME PERIOD IS " <<  elapsed_time_ms << " /1000 SEC" << std::endl;

    for (size_t idx = 0; idx < 12; idx++)
        action[idx] = pfOutputData[idx];

    // // std::shuffle(action_test.begin(), action_test.end(), std::default_random_engine(233));
    // // action = action_test;
    // std::cout << "[AGENT] STATE: ";
    // for(int i=0; i<s_dim; i++){
    //     std::cout << state[i];
    //     if (i != s_dim-1)
    //         std::cout << ", ";
    // }
    // std::cout << std::endl;
    

    // std::memcpy(TF_TensorData(input_tensor), state.data(), std::min(state.size() * sizeof(float), TF_TensorByteSize(input_tensor)));
    // // printf("DATA WERE GIVEN TO THE TENSOR !!! \n");

    // // Run the Session
    // TF_SessionRun(sess, 
    //               nullptr, 
    //               &input_op, &input_tensor, 1,  // Input tensors, input tensor values, number of inputs.
    //               &out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
    //               nullptr, 0,
    //               nullptr, 
    //               status);
    
    // // if(TF_GetCode(status) == TF_OK)
    // //   printf("Session is OK\n");
    // // else
    // //   printf("%s",TF_Message(status));

    // auto data = static_cast<float*>(TF_TensorData(output_tensor));
    // // std::cout << "ACTION: " << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << std::endl;
    // for (int i=0; i<a_dim; i++){
    //     if (progressing)
    //         action[i] = data[i] * progressing_agent_factor;
    //     else
    //         action[i] = data[i] * agent_factor;
    // }

    // std::cout << "[AGENT] ACTION: ";
    // for(int i=0; i<a_dim; i++){
    //     std::cout << action[i];
    //     if (i != a_dim-1)
    //         std::cout << ", ";
    // }
    // std::cout << std::endl;

    

}

void Recovering::_update_action_remote(){

    t_start = std::chrono::high_resolution_clock::now();

    std::ostringstream obs_to_agent;
    // action_to_bullet << std::setprecision(4) << std::fixed;
    for (int i=0; i<s_dim; i++)
        obs_to_agent << observation[i] << ", ";

    // send the request message
    std::cout << "Sending State ..." << std::endl;
    socket.send(zmq::buffer(obs_to_agent.str()), zmq::send_flags::none);
    
    // wait for reply from server
    zmq::message_t reply{};
    socket.recv(reply, zmq::recv_flags::none);

    std::cout << "Received " << reply.to_string() << std::endl;



    std::stringstream ss(reply.to_string());
    int i_a = 0;
    for (float i; ss >> i;) {
        state_from_bullet.push_back(i);    
        action[i_a] = i;
        if (ss.peek() == ',' || ss.peek() == ' ')
            ss.ignore();
        i_a += 1;
    }
    // std::cout << "STATE DIMENSION IS "<< state_from_bullet.size() << std::endl;
    assert(i_a == a_dim);


    // for (size_t idx = 0; idx < 48; idx++)
    //     pdData[idx] = observation[idx];


    // OnnxInference();

    t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "THE TIME PERIOD IS " <<  elapsed_time_ms << " /1000 SEC" << std::endl;

    // for (size_t idx = 0; idx < 12; idx++)
    //     action[idx] = pfOutputData[idx];

}

void Recovering::_update_state_buffer(){
    
    state_buffer.push_back(state);
    assert(state_sum.size() == state.size());
    for(unsigned int i=0; i < state_sum.size(); i++)
        state_sum[i] += state[i];

    if (state_buffer.size() > state_buffer_size){
        for(unsigned int i=0; i < state_sum.size(); i++)
            state_sum[i] -= state_buffer[0][i];

        state_buffer.erase(state_buffer.begin());
    } 

}

void Recovering::_update_observation(){
    // std::cout << "STATE BUFFER SIZE: " << state_buffer.size() << std::endl;
    for(unsigned int i=0; i < state_sum.size(); i++)
        observation[i] = state_sum[i] / state_buffer.size();
}

void Recovering::_Bound(const int & curr_iter){

    ini_bound_length = 500;
    // bound_length = 130;

    if (curr_iter == 0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

        // initial_jpos[3] = initial_jpos[3] - backl_offset;
        // initial_jpos[4] = initial_jpos[4] - back_offset;

        prepare_jpos[0] << 0.0f, -0.75f, 1.5f;
        prepare_jpos[1] << 0.0f, -0.75f, 1.5f;
        prepare_jpos[2] << 0.0f, -0.62f, 1.3f;
        prepare_jpos[3] << 0.0f, -0.62f, 1.3f;

        bound_phase = 1;

        std::cout << "I AM BOUNDING AT TIME LENGTH " << bound_length << std::endl;
    } 
    if (curr_iter < ini_bound_length){
        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, ini_bound_length-1, 
            leg, initial_jpos[leg], prepare_jpos[leg]);
        }

    } else {
        if ((curr_iter-ini_bound_length) % bound_length == 0){
            for(size_t i(0); i < 4; ++i)
                initial_jpos[i] = prepare_jpos[i];
            if (bound_phase == 1){  // high -> low
                // prepare_jpos[0] << 0.0f, -1.1f, 2.1f; // low
                // prepare_jpos[1] << 0.0f, -0.75f, 1.5f; // high
                // prepare_jpos[2] << 0.0f, -0.62f, 1.3f; // high
                // prepare_jpos[3] << 0.0f, -0.95f, 2.0f; // low

                prepare_jpos[0] << 0.0f, -1.1f, 2.1f; // low
                prepare_jpos[1] << 0.0f, -1.1f, 2.1f;
                prepare_jpos[2] << 0.0f, -0.95f, 2.0f; // low
                prepare_jpos[3] << 0.0f, -0.95f, 2.0f;

            } else {  // low -> high
                // prepare_jpos[0] << 0.0f, -0.75f, 1.5f; // high
                // prepare_jpos[1] << 0.0f, -1.1f, 2.1f; // low
                // prepare_jpos[2] << 0.0f, -0.95f, 2.0f; // low
                // prepare_jpos[3] << 0.0f, -0.62f, 1.3f; // high

                prepare_jpos[0] << 0.0f, -0.75f, 1.5f; // high
                prepare_jpos[1] << 0.0f, -0.75f, 1.5f; // high
                prepare_jpos[2] << 0.0f, -0.62f, 1.3f; // high
                prepare_jpos[3] << 0.0f, -0.62f, 1.3f; // high
            }

            bound_phase *= -1;
            
        }
        // for(size_t leg(0); leg<4; ++leg)
        //     _Step((curr_iter-1000) % time_length, time_length-1, leg, initial_jpos[leg], prepare_jpos[leg]);

        float a, b;
        Vec3<float> inter_pos;
        // std::cout << " bound_phase:" << bound_phase << std::endl;
        if (bound_phase == 1){
            b = (sin((float)((curr_iter-ini_bound_length)%bound_length)/(float)(bound_length-1)*PI/2));
            // std::cout << " b:" << b << std::endl;
            a = 1.f - b;
        } else {
            b = (-cos((float)((curr_iter-ini_bound_length)%bound_length)/(float)(bound_length-1)*PI/2))+1;
            // std::cout << " b:" << b << std::endl;
            a = 1.f - b;
        }

        for(size_t leg(0); leg<4; ++leg){

            inter_pos = a * initial_jpos[leg] + b * prepare_jpos[leg];

            _Step(curr_iter, 0, leg, initial_jpos[leg], inter_pos);
        }

    }

}


void Recovering::_BoundToStand(const int & curr_iter){

    // std::cout << "[DEBUG] TRANFERING TO STANDING UP... (CURR_ITER " << curr_iter << ")" << std::endl;

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i) {
            initial_jpos[i] = this->_legController->datas[i].q;
        }

    }

     for(size_t i(0); i<4; ++i){
        _SetJPosInterPts(curr_iter, 600, i, 
            initial_jpos[i], fold_jpos[i]);
    }



    if(curr_iter >= 600){
        if (!fast_stand)
            _phase = 1;
        else
            _phase = 33;

        _motion_start_iter = _state_iter+1;
    } 

}

void Recovering::_Step(
    const size_t & curr_iter, size_t max_iter, int leg, 
    const Vec3<float> & ini, const Vec3<float> & fin){

    float a(0.f);
    float b(1.f);
    Vec3<float> inter_pos;

    if (max_iter != 0){

        // if we're done interpolating
        if(curr_iter <= max_iter) {
          b = (float)curr_iter/(float)max_iter;
          // b = (-cos((float)curr_iter/(float)max_iter*PI)+1)/2;
          a = 1.f - b;
        }

        // compute setpoints
        if (leg == 0 || leg == 1)
            inter_pos = a * ini + b * fin + front_offset;
            // #if DEBUG
            // if (leg == 0){
            //     std::cout << "[DEBUG] COMMAND TO JOINT[0][1]: "<< inter_pos[1] << std::endl;
            // }
            // #endif
        // else if (leg == 3)
        //     inter_pos = a * ini + b * fin + backl_offset;
        else 
            inter_pos = a * ini + b * fin + back_offset;

        // if (leg == 0)
        //     std::cout << "[DEBUG] LEG 0: " << inter_pos[0] << ", " << inter_pos[1] << ", " << inter_pos[2] << std::endl;

        

    } else {
        if (leg == 0 || leg == 1)
            inter_pos = fin + front_offset;
        // else if (leg == 3)
        //     inter_pos = fin + backl_offset;
        else
            inter_pos = fin + back_offset;
    }

    // if (leg == 0)
    //     std::cout << "[DEBUG] INTER_POS[0][1]: " << a * ini[1] + b * fin[1] << " + "<< front_offset[1] << " = " <<  inter_pos[1] << std::endl;

    //  if (leg == 1)
    //     std::cout << "[DEBUG] INTER_POS[1]: " << inter_pos[0] << ", " << inter_pos[1] << ", " << inter_pos[2] << std::endl;
    #if !DEBUG
    if (leg == 2)
        inter_pos[2] += 0.02;   // FIX THE FUCKING KNEE ERROR
    #endif

    Vec3<float> vec3;
    // vec3 << 0.0, -20.0, 20.0;
    vec3 << 0.0, 0.0, 0.0;

    // do control
    this->jointPDControl(leg, inter_pos, vec3);


}

void Recovering::_Step(
    const size_t & curr_iter, size_t max_iter, int leg, 
    const Vec3<float> & ini, const Vec3<float> & fin,
    const Mat3<float> kpMat_arg, const Mat3<float> kdMat_arg){

    float a(0.f);
    float b(1.f);
    Vec3<float> inter_pos;

    if (max_iter != 0){

        if(curr_iter <= max_iter) {
          b = (float)curr_iter/(float)max_iter;
          a = 1.f - b;
        }

        if (leg == 0 || leg == 1)
            inter_pos = a * ini + b * fin + front_offset;
        else 
            inter_pos = a * ini + b * fin + back_offset;


    } else {
        if (leg == 0 || leg == 1)
            inter_pos = fin + front_offset;
        else
            inter_pos = fin + back_offset;
    }

    #if !DEBUG
    if (leg == 2)
        inter_pos[2] += 0.02;   // FIX THE FUCKING KNEE ERROR
    #endif

    Vec3<float> vec3;
    vec3 << 0.0, 0.0, 0.0;

    // do control
    this->jointPDControl(leg, inter_pos, vec3, kpMat_arg, kdMat_arg);


}

void Recovering::_MoveHands(const int & curr_iter){
  std::cout << "TRY TO KEEP BALANCE !!!  " << std::endl;

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i) {
            initial_jpos[i] = this->_legController->datas[i].q;
        }
    }

    // int length = front_factor*90.5; //80<x<100, around 90, 80<89<89.5<x<90<100
    // for(size_t i(0); i<2; ++i){
    //     front_jpos[i] << 0.f, -1.6, 1.6*0.96;  
    // }

    // for(size_t leg(0); leg<2; ++leg){
    //   _Step(curr_iter, length, 
    //       leg, initial_jpos[leg], front_jpos[leg]);
    // }


    // Vec4<float> se_contactState(0.5,0.5,0.5,0.5);

    // if(curr_iter >= 1500){
    //     _phase = -4; // move the hand
    //     _motion_start_iter = _state_iter + 1;
    // }


}



void Recovering::_PushTest(const int & curr_iter){
    // PHASE 38

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

        climb_th1_p = -1.5;
        climb_th2_p = 2.6;

    }

    Mat3<float> kpMat_drop;
    Mat3<float> kdMat_drop;

    kpMat_drop << 0, 0, 0, 0, 0, 0, 0, 0, 0;
    kdMat_drop << 0, 0, 0, 0, 0, 0, 0, 0, 0;

    pos_impl[0] << 0, climb_th1_p, climb_th2_p;
    pos_impl[1] << 0, climb_th1_p, climb_th2_p;
    pos_impl[2] << 0, -PI/2, PI/2;
    pos_impl[3] << 0, -PI/2, PI/2;

    if (_within_limits()){

        for(size_t leg(0); leg<2; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg], kpMat_drop, kdMat_drop);
        } 

        for(size_t leg(2); leg<4; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 
    } else {
        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
    }

    if(curr_iter >= 2000 + 1000){
        _phase = -28;

        _motion_start_iter = _state_iter+1;
    } 
    
}


void Recovering::_PushTest1(const int & curr_iter){
    // PHASE 37
    // prepare for the actions

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

        climb_th1_p = -1.5;
        climb_th2_p = 2.6;

    }

    pos_impl[0] << 0, climb_th1_p, climb_th2_p;
    pos_impl[1] << 0, climb_th1_p, climb_th2_p;
    pos_impl[2] << 0, X_climb[1+0], -1.000f;
    pos_impl[3] << 0, X_climb[1+0], -1.000f;

    if (_within_limits()){
        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 
    } else {
        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
    }

    if(curr_iter >= 2000 + 1000){
        if (full_climb)
            _phase = 28;
        else
            _phase = -28;
        _motion_start_iter = _state_iter+1;
    } 
    
}

// void Recovering::_WalkTest(const int & curr_iter){
//     // PHASE 39
//     // prepare for the actions

//     if (curr_iter==0){
//         for(size_t i(0); i < 4; ++i)
//           initial_jpos[i] = this->_legController->datas[i].q;
//         for(size_t i(0); i < 2; ++i)
//             initial_jpos[i] = initial_jpos[i] - front_offset;
//         for(size_t i(2); i < 4; ++i)
//             initial_jpos[i] = initial_jpos[i] - back_offset;

//         // climb_th1_p = -1.5;
//         // climb_th2_p = 2.6;

//         // theta1 = -1.156;
//         // theta2 = -1.000;
//         // ture: -0.976, -1


//         th1_ = theta1;
//         th2_ = theta2;
//         // CORRESPONDING FK:   x: -0.375666    z: 0.0357188


//         float test_fk_x, test_fk_z;
//         std::cout << "[DEBUG] th1: " << th1_ << "    th2: " << th2_ << std::endl;
//         _LegsFK(th1_, th2_, test_fk_x, test_fk_z);

//         std::cout << "[DEBUG FK] test_fk_x: " << test_fk_x << std::endl;
//         std::cout << "[DEBUG FK] test_fk_z: " << test_fk_z << std::endl;
//         std::cout << "==============================================" << std::endl;
//         std::cout << "USE test_fk_x, test_fk_z FOR IK: " << std::endl;
//         LegsIK(test_fk_x, 0, test_fk_z, th0_l, th0_r, th1_, th2_); // right leg
//         std::cout << "[DEBUG] IK --> th1: " << th1_ << "    th2: " << th2_ << std::endl;


//     }

//     pos_impl[0] << th0p_l, th1p_, th2p_;
//     pos_impl[1] << th0p_r, th1p_, th2p_;


//     pos_impl[2] << th0_l, th1_, th2_;
//     pos_impl[3] << th0_r, th1_, th2_;

//     // pos_impl[2] = pos_impl[2] - back_offset;
//     // pos_impl[3] = pos_impl[3] - back_offset;


//     // theta1 = -1.156;
//     //     theta2 = -1.000;
        

//     // }

//     // // float delta_stretch = 1.3;
    

//     // // pos_impl[0] << 0, -1.571f, 0.998f;
//     // // pos_impl[1] << 0, -1.571f, 0.998f;
//     // // pos_impl[2] << 0, -PI/2, 0;
//     // // pos_impl[3] << 0, -PI/2, 0;
//     // // pos_impl[3] << 1.3, -1.1f, climb_up_th2;
//     // std::cout << "===================================" << std::endl;
//     // std::cout << "[DEBUG] ! DESIRED POSITION: " << pos_impl[0][1] << ", " << pos_impl[0][2] << ", " << pos_impl[2][1] << ", " << pos_impl[2][2] << std::endl;
//     // std::cout << "[DEBUG] ! CURRENT POSITION: " << this->_legController->datas[0].q[1] << ", " << this->_legController->datas[0].q[2] << ", " << this->_legController->datas[2].q[1] << ", " << this->_legController->datas[2].q[2] << std::endl;

//     // pos_impl[2] = pos_impl[2] - back_offset;
//     // pos_impl[3] = pos_impl[3] - back_offset;

//     bool joint_check = false;

//     if (_within_limits() || ~joint_check){
//         for(size_t leg(0); leg<4; ++leg){
//             _Step(curr_iter, 2000, 
//             leg, initial_jpos[leg], pos_impl[leg]);
//         } 
//     } else {
//         _phase = -911;
//         std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
//     }

//     if(curr_iter >= 2000 + 2500){

//         _phase = -28;
//         _motion_start_iter = _state_iter+1;
//     } 
    
// }

// void Recovering::_WalkTest1(const int & curr_iter){
//     // PHASE 40
//     // try trajectory

//     if (curr_iter==0){
//         for(size_t i(0); i < 4; ++i)
//           initial_jpos[i] = this->_legController->datas[i].q;
//         for(size_t i(0); i < 2; ++i)
//             initial_jpos[i] = initial_jpos[i] - front_offset;
//         for(size_t i(2); i < 4; ++i)
//             initial_jpos[i] = initial_jpos[i] - back_offset;

//         // climb_th1_p = -1.5;
//         // climb_th2_p = 2.6;


//         th1_ = theta1;
//         th2_ = theta2;


//         // ture: -0.976, -1

//         float fk_x, fk_z;
//         _LegsFK(th1_, th2_, fk_x, fk_z);
//         std::cout << "[WALK] INITIAL POS:  th1: " << th1_ << "    th2: " << th2_ << std::endl;
//         std::cout << "[WALK] FK RESULTS:  x: " << fk_x << "    z: " << fk_z << std::endl;

        

//         // cnpy::NpyArray arr = cnpy::npy_load("/home/chen/fun_v0.3/user/Zero_Controller/left_traj.npy");
//         cnpy::NpyArray arr = cnpy::npy_load("/home/chen/_LAB/Cheetah-ZMP/data/left_traj.npy");
//         loaded_data = arr.data<double>();
//         traj_length = arr.shape[0];
//         pose_dim = arr.shape[1];
//         std::cout << "[TRAJ] SHAPE 0: " << traj_length << std::endl;
//         std::cout << "[TRAJ] SHAPE 1: " << pose_dim << std::endl;
//         left_traj = std::vector<double>(loaded_data, loaded_data+traj_length*pose_dim);

//         arr = cnpy::npy_load("/home/chen/_LAB/Cheetah-ZMP/data/right_traj.npy");
//         loaded_data = arr.data<double>();
//         // assert(arr.shape[0] == traj_length && arr.shape[1] == pose_dim);
//         right_traj = std::vector<double>(loaded_data, loaded_data+traj_length*pose_dim);
//         // std::cout << "[DEBUG]: z_0: " << left_traj[2] << std::endl;
//         // std::cout << "[DEBUG]: z_0: " << loaded_data[2] << std::endl;

//         lipm_z_offset = fk_z;  // - 0.0357188
//         lipm_x_offset = - right_traj[2] + fk_x; // + 0.395/2   (lipm height - ini height)
//         std::cout << "[WALK] Z OFFSET: " << lipm_z_offset << std::endl;
//         std::cout << "[WALK] X OFFSET: " << lipm_x_offset << std::endl;

//     }

//     if (curr_iter*pose_dim+2 < traj_length*pose_dim){
//     // if (curr_iter == 50){

//         // STANDING UP:
//         // X-axis:  -0.375666
//         // Y-axis:  0
//         // Z-axis:  0.0357188

//         float left_traj_x = left_traj[curr_iter*pose_dim] + lipm_z_offset;
//         float left_traj_y = left_traj[curr_iter*pose_dim+1];// + ab_offset;
//         float left_traj_z = left_traj[curr_iter*pose_dim+2] + lipm_x_offset;


//         float right_traj_x = right_traj[curr_iter*pose_dim] + lipm_z_offset;
//         float right_traj_y = right_traj[curr_iter*pose_dim+1];// + ab_offset;
//         float right_traj_z = right_traj[curr_iter*pose_dim+2] + lipm_x_offset;
        
//         std::cout<< std::endl;

//         std::cout<<"========================PURE DEBUG============================="<<std::endl;

//         LegsIK(-0.3, -0.03, 0.04, th0_l, th0_r, th1_l, th2_l);
//         std::cout<<"==============================================================="<<std::endl;
//         std::cout<< std::endl;
//         // std::cout << "[DEBUG]: curr_iter: " << curr_iter << std::endl;
//         // std::cout << "[DEBUG] curr_iter\%pose_dim: " << curr_iter%pose_dim << ", curr_iter\%pose_dim+1: " << curr_iter%pose_dim+1 << ", curr_iter\%pose_dim+1: " << curr_iter%pose_dim+2 << std::endl;
//         std::cout<<"========================DEBUG============================="<<std::endl;
        
//         // LegsIK(left_traj_z, left_traj_y, left_traj_x, th0_l, th0_r, th1_l, th2_l); // left leg
//         _LeftLegIK(left_traj_x, left_traj_y, left_traj_z, th0_l, th1_l, th2_l);
//         std::cout << "======LEFT TRAJECTORY: " << std::endl;
//         std::cout << "[DEBUG] traj_x: " << left_traj_x << ", traj_y: " << left_traj_y << ", traj_z: " << left_traj_z << std::endl;
//         std::cout << "======LEFT IK RESULT: " << std::endl;
//         std::cout << "[*DEBUG] [th0_l]: " << th0_l << ", th1_l: " << th1_l << ", th2_l: " << th2_l << std::endl;
//         // if (fabs(th0_l) > 0.2) exit(1);
//         // LegsIK(right_traj_z, right_traj_y, right_traj_x, th0_l, th0_r, th1_r, th2_r); // right leg
//         _RightLegIK(right_traj_x, right_traj_y, right_traj_z, th0_r, th1_r, th2_r);
//         std::cout << "======RIGHT TRAJECTORY: " << std::endl;
//         std::cout << "[DEBUG] traj_x: " << right_traj_x << ", traj_y: " << right_traj_y << ", traj_z: " << right_traj_z << std::endl;
//         std::cout << "======RIGHT IK RESULT: " << std::endl;
//         std::cout << "[*DEBUG] [th0_r]: " << th0_r << ", th1_r: " << th1_r << ", th2_r: " << th2_r << std::endl;

//         // float right_traj_x = 0; // left_traj[curr_iter%pose_dim];
//         // float right_traj_y = 0 - ab_offset; //left_traj[curr_iter%pose_dim+1];
//         // float right_traj_z = -0.23; //left_traj[curr_iter%pose_dim+2];
//         // LegsIK(0.1, 0, 0.26, th0_l, th0_r, th1_r, th2_r); // right leg
//         // th0_r = 0, th1_r = 0, th2_r = 0;

//         // th0_r = 0, th0_l = 0;

//         th1p_ = initial_jpos[0][1];
//         th2p_ = initial_jpos[0][2];

//         // if (th0_l > 0.4)
//         //     _phase = -120;
//         _phase = -120; // DEBUG: VIEW FIRST STEP

//     } 
//     // else {
//     //     exit(1);
//     // }

//     // std::cout << "[DEBUG] th0_l: " << th0_l << ", th0_r: " << th0_r << std::endl;

//     pos_impl[0] << 0, th1p_, th2p_;  //right
//     pos_impl[1] << 0, th1p_, th2p_;  // left


//     pos_impl[2] << th0_r, th1_r, th2_r;  // right
//     pos_impl[3] << th0_l, th1_l, th2_l;  //left

//     // pos_impl[2] = pos_impl[2] - back_offset;
//     // pos_impl[3] = pos_impl[3] - back_offset;

//     // if (th0_l > 1) exit(1);


//     bool joint_check = false;

//     // if (_within_limits() || ~joint_check){
//     if (~joint_check){
//         for(size_t leg(0); leg<4; ++leg){
//             _Step(curr_iter, 0, 
//             leg, initial_jpos[leg], pos_impl[leg]);
//         } 
//     } else {
//         _phase = -911;
//         std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
//     }

//     // if (curr_iter > 20)
//     //     exit(1);


//     if(curr_iter >= 2000 + 250000){

//         _phase = -28;
//         _motion_start_iter = _state_iter+1;
//     } 
    
// }



bool Recovering::isSafe(){
  // Check RC mode
  // More checks should be added
  // std::cout << "conrol mode: " << _controlParameters->control_mode << std::endl; // 0
  if (_controlParameters->control_mode != K_RECOVERY_STAND){
    return false;
  }else{
    return true;
  }
}

void Recovering::_Passive(){

  _motion_start_iter = _state_iter + 1;

}


void Recovering::_Finish(){
  std::cout << "I was finished! "  << std::endl; 
  _phase = 5;
  _motion_start_iter = _state_iter + 1;

}

float Recovering::_Box(const float & num, const float & min, const float & max){
    reach_limit = false;
    if (num > max) {reach_limit = true; return max;}
    else if (num < min) {reach_limit = true; return min;}
    else return num;
}

float Recovering::_toPitch(const Eigen::Quaterniond& q){

    // double roll_, pitch_, yaw_;

    // // roll (x-axis rotation)
    // double sinr_cosp = +2.0 * (q.w() * q.x() + q.y() * q.z());
    // double cosr_cosp = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
    // roll_ = atan2(sinr_cosp, cosr_cosp);

    // // pitch (y-axis rotation)
    
    // double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
    // if (fabs(sinp) >= 1)
    // pitch_ = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    // else
    // pitch_ = asin(sinp);

    // // yaw (z-axis rotation)
    // double siny_cosp = +2.0 * (q.w() * q.z() + q.x() * q.y());
    // double cosy_cosp = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
    // yaw_ = atan2(siny_cosp, cosy_cosp);

    // if (fabs(roll_) > 3.1415/2 || fabs(yaw_) > 3.1415/2)
    //     pitch_ = -3.1415 - pitch_;
    return _toRPY(q)[1];


}

std::vector<float> Recovering::_toRPY(const Eigen::Quaterniond& q){
    

    float q_x = q.x(), q_y = q.y(), q_z = q.z(), q_w = q.w();
    float x2 = q_x + q_x,   y2 = q_y + q_y, z2 = q_z + q_z;
    float xx = q_x * x2, xy = q_x * y2, xz = q_x * z2;
    float yy = q_y * y2, yz = q_y * z2, zz = q_z * z2;
    float wx = q_w * x2, wy = q_w * y2, wz = q_w * z2;

    rot_mat[ 0 ] = ( 1 - ( yy + zz ) ) * 1;
    rot_mat[ 1 ] = ( xy + wz ) * 1;
    rot_mat[ 2 ] = ( xz - wy ) * 1;
    rot_mat[ 3 ] = 0;

    rot_mat[ 4 ] = ( xy - wz ) * 1;
    rot_mat[ 5 ] = ( 1 - ( xx + zz ) ) * 1;
    rot_mat[ 6 ] = ( yz + wx ) * 1;
    rot_mat[ 7 ] = 0;

    rot_mat[ 8 ] = ( xz + wy ) * 1;
    rot_mat[ 9 ] = ( yz - wx ) * 1;
    rot_mat[ 10 ] = ( 1 - ( xx + yy ) ) * 1;
    rot_mat[ 11 ] = 0;

    rot_mat[ 12 ] = 0;
    rot_mat[ 13 ] = 0;
    rot_mat[ 14 ] = 0;
    rot_mat[ 15 ] = 1;

    float RPY_x, RPY_y, RPY_z;
    RPY_z = asin(std::max(std::min(rot_mat[1], 1.f), -1.f));

    if (abs(rot_mat[1]) < 0.9999999){
        RPY_x = atan2(-rot_mat[9], rot_mat[5]);
        RPY_y = atan2(-rot_mat[2], rot_mat[0]);
    } else {
        RPY_x = 0;
        RPY_y = atan2(rot_mat[8], rot_mat[1]);
    }
       
    std::vector<float> ans = {RPY_x, RPY_y, RPY_z};

    return ans;
}

void Recovering::_update_rpy(){
    
    
    Eigen::Quaterniond q(_stateEstimator->getResult().orientation[3], 
                         _stateEstimator->getResult().orientation[0], 
                         _stateEstimator->getResult().orientation[1], 
                         _stateEstimator->getResult().orientation[2]);
   // THIS ORDER MAY BE WRONG!!!

    float q_x = q.x(), q_y = q.y(), q_z = q.z(), q_w = q.w();
    float x2 = q_x + q_x,   y2 = q_y + q_y, z2 = q_z + q_z;
    float xx = q_x * x2, xy = q_x * y2, xz = q_x * z2;
    float yy = q_y * y2, yz = q_y * z2, zz = q_z * z2;
    float wx = q_w * x2, wy = q_w * y2, wz = q_w * z2;

    rot_mat[ 0 ] = ( 1 - ( yy + zz ) ) * 1;
    rot_mat[ 1 ] = ( xy + wz ) * 1;
    rot_mat[ 2 ] = ( xz - wy ) * 1;
    rot_mat[ 3 ] = 0;

    rot_mat[ 4 ] = ( xy - wz ) * 1;
    rot_mat[ 5 ] = ( 1 - ( xx + zz ) ) * 1;
    rot_mat[ 6 ] = ( yz + wx ) * 1;
    rot_mat[ 7 ] = 0;

    rot_mat[ 8 ] = ( xz + wy ) * 1;
    rot_mat[ 9 ] = ( yz - wx ) * 1;
    rot_mat[ 10 ] = ( 1 - ( xx + yy ) ) * 1;
    rot_mat[ 11 ] = 0;

    rot_mat[ 12 ] = 0;
    rot_mat[ 13 ] = 0;
    rot_mat[ 14 ] = 0;
    rot_mat[ 15 ] = 1;

    float RPY_x, RPY_y, RPY_z;
    RPY_z = asin(std::max(std::min(rot_mat[1], 1.f), -1.f));

    if (abs(rot_mat[1]) < 0.9999999){
        RPY_x = atan2(-rot_mat[9], rot_mat[5]);
        RPY_y = atan2(-rot_mat[2], rot_mat[0]);
    } else {
        RPY_x = 0;
        RPY_y = atan2(rot_mat[8], rot_mat[1]);
    }
    // std::cout << "[DEBUG] QUAT2RPY: [" << q_x << ", " << q_y << ", "<< q_z << ", "<< q_w << "]  -->  [" << RPY_x << ", "<<  RPY_y << ", "<<  RPY_z << "]" << std::endl;
    rpy = {-RPY_x, - RPY_y, - RPY_z};
    // THE PITCH AND YAW SHOULD BE NEGATIVE. I DONT KNOW WHY. MAYBE ALSO ROLL

    #if DEBUG
    rpy = {state_from_bullet[0], state_from_bullet[1], state_from_bullet[2]};
    #endif


}

void Recovering::_FK(const float & th1, const float & th2, float & x, float & y){
    x = c_*cos(th1) + b_*cos(th1+th2);
    y = c_*sin(th1) + b_*sin(th1+th2);
}

void Recovering::_IK(const float & x, const float & y, float & th1, float & th2){
    th2 = atan2( - sqrt(1 - pow(((x*x + y*y - c_*c_ - b_*b_) / (2*c_*b_)), 2.0)), (x*x + y*y - c_*c_ - b_*b_) / (2*c_*b_));
    th1 = atan2(y, x) - atan2(b_*sin(th2), c_ + b_*cos(th2));
}

void Recovering::_LegsFK(const float & th1, const float & th2, float & x, float & z){
    // this is with different frame with _FK
    x = b_*sin(th1) + c_*sin(th1+th2);
    z = b_*cos(th1) + c_*cos(th1+th2);
    std::cout << "[DEBUG] b_*cos(th1): " << b_*cos(th1) << "    - c_*sin(th1+th2): " << - c_*sin(th1+th2) << std::endl;
}

// void Recovering::_RightLegIK(const float x, const float y, const float z, float &th0, float &th1, float &th2){
//     // Humanoid frame
//     float dummy;
//     // LegsIK(z, y+torso_y/2, -x, dummy, th0, th1, th2);
//     LegsIK(z, -y-torso_y/2, -x, dummy, th0, th1, th2);
    
// }

// void Recovering::_LeftLegIK(const float x, const float y, const float z, float &th0, float &th1, float &th2){
//     // Humanoid frame
//     float dummy;
//     // LegsIK(z, y-torso_y/2, -x, dummy, th0, th1, th2);
//     LegsIK(z, -y+torso_y/2, -x, dummy, th0, th1, th2);
    
// }

void Recovering::_process_remote_controller_signal(const int & curr_iter){
    // if phase == 0 and mode == 12:
    //     do nothing, the show will go on
    // if phase == 0 and mode == 11:
    //     do nothing, the show will go on
    // if phase == 5 and mode == 11:
    //     do nothing, but listen to the controller values
    // if phase == 5 and mode == 12:
    //     phase = 6, go back to sleep
    // if phase == 6 and mode == 12:
    //     do nothing, the show will go on
    // if phase == 6 and mode == 11:
    //    phase = 0
    // (void)curr_iter;

    rc_mode = this->_desiredStateCommand->rcCommand->mode;
    // float rc_value = (this->_desiredStateCommand->rcCommand->v_des[0] + 0.7) / 2.1 *2 - 1 + 0.333333;  // -1 ~ 1

    // if (_phase == 11 && (rc_mode == 11 || rc_mode == 13)) {
    //     // _phase = 12;
    //     _phase = 1;
    //     _motion_start_iter = _state_iter;

    // } else {
    //     _phase = -1;
    // }
    
    // else if (_phase == 5 && (rc_mode == 11 || rc_mode == 13)){
    //     // delta_x_buffer.push_back(param_opt[8] + 0.1*rc_value);
    //     // delta_x_sum += param_opt[8] + 0.1*rc_value;

    //     // if (delta_x_buffer.size() > 100){
    //     //     delta_x_sum -= delta_x_buffer[0];
    //     //     delta_x_buffer.erase(delta_x_buffer.begin());
    //     // } 
    //     // delta_x_buffered = delta_x_sum / delta_x_buffer.size();

    //     // std::cout << "BUFFERED DELTA: " << delta_x_buffered << std::endl;

    //     // param_b_buffered = std::min(std::max(param_opt[0] + rc_value*0.03f, B_range[0]), B_range[1]); 
    //     // std::cout << "BUFFERED PARAM B: " << param_opt[0] << " + " << rc_value << " * 0.03 = " << param_b_buffered << std::endl;

    //     // NOTE THAT IN SOME CASES THIS WILL BE OVERWRITTEN 
    // } else if (_phase == 5 && rc_mode == 12) {
    //     stopping = true;

    // } else if (_phase >= 6 && _phase < 10 && (rc_mode == 11 || rc_mode == 13)) {
    //     _phase = -1;

    // }

    if (curr_iter % 500 == 0)
        std::cout << "[RC] PHASE: " << _phase << "  RC MODE: " << rc_mode << std::endl;
    
    

    // std::cout << " TEST, MODE: " << rc_mode << "  VALUE: " << this->_desiredStateCommand->rcCommand->v_des[0] << ", " 
    //                                                     << this->_desiredStateCommand->rcCommand->v_des[1] << ", " 
    //                                                     << this->_desiredStateCommand->rcCommand->omega_des[0] << ", " 
    //                                                     << this->_desiredStateCommand->rcCommand->omega_des[1] << ", " 
    //                                                     << this->_desiredStateCommand->rcCommand->rpy_des[0] << ", " 
    //                                                     << this->_desiredStateCommand->rcCommand->rpy_des[1] << ", " 
    //                                                     << this->_desiredStateCommand->rcCommand->rpy_des[2] 
    //                                                     << std::endl;

}

void Recovering::_update_reward(){
    if (reach_limit) limit_cost = 0.1;
    else limit_cost = 0;
    action_cost = 0.25 * std::max({fabs(action[0]), fabs(action[1]), fabs(action[2]), fabs(action[3])});
    reward = _stateEstimator->getResult().position[2] - limit_cost + _stateEstimator->getResult().position[0] - action_cost + 0.5;
    episode_return_h += _stateEstimator->getResult().position[2];
    episode_return_limit_cost += limit_cost;
    episode_return_x += _stateEstimator->getResult().position[0];
    episode_return_action_cost += action_cost;
    episode_return_alive += 0.5;
    episode_return += reward;
    std::cout << "[RETURN] REWARD: " << _stateEstimator->getResult().position[2] << " - " << limit_cost << " + " << _stateEstimator->getResult().position[0] << " - " << action_cost << " + 0.5 = " << reward << "    RETURN: " << episode_return << std::endl;; 
}

void Recovering::_log_return(){
    std::cout << "[RETURN] ==================  FINAL RETURN: " << episode_return << "  ==================" << std::endl;

    if (!has_logged){

        std::ofstream debug_file;
        std::string DAM_path_str(DAM_path);

        if (progressing)
            debug_file.open(DAM_path_str + "robot_returns_pro_prog", std::ios_base::app);
        else
            debug_file.open(DAM_path_str + "robot_returns_pro_wprog", std::ios_base::app);

        if (!debug_file)
          std::cout << "Can't open output file " << std::endl;

        
        // std::ostringstream debug_info_stream;
        // debug_info_stream << episode_return << "\n";
        // debug_file << debug_info_stream.str();

        debug_file << episode_return_h << " " << episode_return_limit_cost << " " << episode_return_x << " " << episode_return_action_cost << " " << episode_return_alive << " " << episode_return << "\n";
        // debug_file << episode_return << "\n";

        debug_file.close();

        if (progressing)
            std::cout << "[RETURN] LOGGED TO " << DAM_path_str + "robot_returns_pro_prog" << std::endl;
        else
            std::cout << "[RETURN] LOGGED TO " << DAM_path_str + "robot_returns_pro_wprog" << std::endl;


        has_logged = true;
    }

}

bool Recovering::_within_limits(){


    for (size_t leg(0); leg<4; ++leg){
        if (pos_impl[leg][0] < -PI/2 || pos_impl[leg][0] > PI/2 || std::isnan(pos_impl[leg][0])){    
            std::cout << "[SECURITY] JOINT 0 OF LEG " << leg << " GOT A CRAZY ACTION: "<< pos_impl[leg][0] << "\n";
            return false;
        }
        else if (pos_impl[leg][1] < -PI || pos_impl[leg][1] > PI || std::isnan(pos_impl[leg][1])){    
            std::cout << "[SECURITY] JOINT 1 OF LEG " << leg << " GOT A CRAZY ACTION: "<< pos_impl[leg][1] << "\n";
            return false;
        }
        else if (pos_impl[leg][2] < -2.8 || pos_impl[leg][2] > 2.8 || std::isnan(pos_impl[leg][2])){    
            std::cout << "[SECURITY] JOINT 2 OF LEG " << leg << " GOT A CRAZY ACTION: "<< pos_impl[leg][2] << "\n";
            return false;
        }
    }

    return true;
}

void Recovering::CheckOrtStatus(OrtStatus* status, const OrtApi* ortApi){
    if (status != NULL) {
        const char* sMsg = ortApi->GetErrorMessage(status);
        std::cerr << "ONNX Runtime error: " << sMsg << std::endl;
        ortApi->ReleaseStatus(status);
        exit(1);
    }
}

void Recovering::PrintTensorInfo(std::string Type, size_t Idx, const char* Name, OrtTypeInfo* TypeInfo, const OrtApi* ortApi)
{
    const OrtTensorTypeAndShapeInfo* pTensorInfo;
    ONNXTensorElementDataType type;
    size_t iNumDims;
    std::vector<int64_t> NodeDims;

    CheckOrtStatus(ortApi->CastTypeInfoToTensorInfo(TypeInfo, &pTensorInfo), ortApi);
    CheckOrtStatus(ortApi->GetTensorElementType(pTensorInfo, &type), ortApi);
    CheckOrtStatus(ortApi->GetDimensionsCount(pTensorInfo, &iNumDims), ortApi);
    NodeDims.resize(iNumDims);
    CheckOrtStatus(ortApi->GetDimensions(pTensorInfo, (int64_t*)NodeDims.data(), iNumDims), ortApi);
    std::cout << "- " << Type << " " << Idx << ": name=" << Name << ", type=" << type << ", ndim=" << iNumDims << ", shape=[";
    for (size_t dimIdx = 0; dimIdx < iNumDims; dimIdx++)
    {
        std::cout << NodeDims[dimIdx] << ((dimIdx >= iNumDims-1)?"":",");
    }
    std::cout << "]" << std::endl;
}

void Recovering::LoadOnnxModel(){
    
    pOrtApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    CheckOrtStatus(pOrtApi->CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "onnx.log", &pOnnxEnv), pOrtApi);
    assert(pOnnxEnv != NULL);
    std::cout << "ONNX environment created successfully." << std::endl;

    CheckOrtStatus(pOrtApi->CreateSessionOptions(&pOnnxSessionOpts), pOrtApi);
    CheckOrtStatus(pOrtApi->SetSessionGraphOptimizationLevel(pOnnxSessionOpts, ORT_ENABLE_ALL), pOrtApi);

    // Create a session from the saved model using default settings, so it'll run on the CPU.

    // const char* sModelName = "./data/policy_1.onnx";
    const char* sModelName = "./data/a1_18nm_80kp.onnx";

    CheckOrtStatus(pOrtApi->CreateSession(pOnnxEnv, sModelName, pOnnxSessionOpts, &pOnnxSession), pOrtApi);
    assert(pOnnxSession != NULL);
    std::cout << "ONNX session create successfully." << std::endl;

    // Print info about the inputs and outputs of the model.
    // Shouldd be one input and one output, both having dimensions of [-1,10] and data type equal to 1 (float).
    CheckOrtStatus(pOrtApi->SessionGetInputCount(pOnnxSession, &iNumInputs), pOrtApi);
    std::cout << "Model has " << iNumInputs << " input(s)." << std::endl;
    CheckOrtStatus(pOrtApi->GetAllocatorWithDefaultOptions(&pOnnxMemAllocator), pOrtApi);
    psInputNames = new const char*[iNumInputs]; // You need to keep track of the names of the inputs to the network so you can specify multiple inputs when runningt the network, if it has multiple inputs.
    for (size_t inIdx = 0; inIdx < iNumInputs; inIdx++)
    {
        char* sName;
        OrtTypeInfo* pTypeinfo;
        CheckOrtStatus(pOrtApi->SessionGetInputName(pOnnxSession, inIdx, pOnnxMemAllocator, &sName), pOrtApi);
        CheckOrtStatus(pOrtApi->SessionGetInputTypeInfo(pOnnxSession, inIdx, &pTypeinfo), pOrtApi);
        psInputNames[inIdx] = sName;
        PrintTensorInfo("Input", inIdx, sName, pTypeinfo, pOrtApi);
        pOrtApi->ReleaseTypeInfo(pTypeinfo);
    }
    CheckOrtStatus(pOrtApi->SessionGetOutputCount(pOnnxSession, &iNumOutputs), pOrtApi);
    std::cout << "Model has " << iNumOutputs << " output(s)." << std::endl;
    psOutputNames = new const char*[iNumOutputs]; // Also need to get the name of the network's output so that one can tell the runtime which outputs to calculate.
    for (size_t outIdx = 0; outIdx < iNumOutputs; outIdx++)
    {
        char* sName;
        OrtTypeInfo* pTypeinfo;
        CheckOrtStatus(pOrtApi->SessionGetOutputName(pOnnxSession, outIdx, pOnnxMemAllocator, &sName), pOrtApi);
        CheckOrtStatus(pOrtApi->SessionGetOutputTypeInfo(pOnnxSession, outIdx, &pTypeinfo), pOrtApi);
        psOutputNames[outIdx] = sName;
        PrintTensorInfo("Output", outIdx, sName, pTypeinfo, pOrtApi);
        pOrtApi->ReleaseTypeInfo(pTypeinfo);
    }

    // Create the input tensor and fill it with data.
    CheckOrtStatus(pOrtApi->CreateCpuMemoryInfo(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault, &pOnnxMemAllocInfo), pOrtApi);
    // std::cout << "Created CPU memory allocator." << std::endl;
    CheckOrtStatus(pOrtApi->CreateTensorWithDataAsOrtValue(pOnnxMemAllocInfo, pdData, sizeof(float) * 48, piShape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pOnnxInput), pOrtApi);

    
}

void Recovering::OnnxInference(){
    // for (size_t idx = 0; idx < 48; idx++)
    //     pdData[idx] = 1.0f;

    // // Create the input tensor and fill it with data.
    // CheckOrtStatus(pOrtApi->CreateCpuMemoryInfo(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault, &pOnnxMemAllocInfo), pOrtApi);
    // // std::cout << "Created CPU memory allocator." << std::endl;
    // CheckOrtStatus(pOrtApi->CreateTensorWithDataAsOrtValue(pOnnxMemAllocInfo, pdData, sizeof(float) * 48, piShape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pOnnxInput), pOrtApi);
    assert(pOnnxInput != NULL);
    CheckOrtStatus(pOrtApi->IsTensor(pOnnxInput, &iFlag), pOrtApi);
    assert(iFlag);
    // std::cout << "Allocated input tensor." << std::endl;

    // Finally execute the neural network.
    CheckOrtStatus(pOrtApi->Run(pOnnxSession, NULL, (const char* const*)psInputNames, (const OrtValue* const*)&pOnnxInput, 1, (const char* const*)psOutputNames, 1, &pOnnxOutput), pOrtApi);

    CheckOrtStatus(pOrtApi->GetTensorMutableData(pOnnxOutput, (void**)&pfOutputData), pOrtApi);
    // std::cout << "Output tensor: [";
    // for (size_t idx = 0; idx < 12; idx++)
    //     std::cout << pfOutputData[idx] << ((idx==12-1)?"":",");
    // std::cout << "]" << std::endl;

    // std::cout << "Verifying results: ";
    // bool bSuccess = true;
    // for (size_t idx = 0; idx < 12; idx++)
    // {
    //     bSuccess &= pfOutputData[idx] == 561.0;
    // }
    // std::cout << (bSuccess?"PASS":"FAIL") << std::endl;


}

void Recovering::OnnxCleanup(){
    // Cleanup time.
    delete psInputNames;
    delete psOutputNames;
    pOrtApi->ReleaseValue(pOnnxInput);
    pOrtApi->ReleaseValue(pOnnxOutput);
    pOrtApi->ReleaseMemoryInfo(pOnnxMemAllocInfo);
    // pOrtApi->ReleaseAllocator(pOnnxMemAllocator);
    // OrtReleaseSession(pOnnxSession);
    pOrtApi->ReleaseEnv(pOnnxEnv);
    std::cout << "ONNX Runtime variables released." << std::endl;
}

Eigen::Vector3d Recovering::quat_rotate_inverse(Eigen::Vector4d &q, Eigen::Vector3d &v){
    float q_w = q[3];
    Eigen::Vector3d q_vec, a, b, c;
    q_vec << q[0], q[1], q[2];
    // a = v * (2.0 * q_w ** 2 - 1.0)
    a = (2. * pow(q_w, 2) - 1.) * v;
    // b = np.cross(q_vec, v) * q_w * 2.0
    b = q_vec.cross(v) * q_w * 2.0;
    // c = q_vec * np.matmul(q_vec.reshape(1, 3), v.reshape(3, 1)) * 2.0
    c = q_vec * v.transpose() * 2. * q_vec;
    // cwiseProduct
    return a - b + c;
}

void Recovering::joint_pos_clip(){

    for (size_t leg = 0; leg < 4; leg++){
        pos_impl[leg][0] = std::min(std::max(pos_impl[leg][0], -0.802851455917f), 0.802851455917f);
        pos_impl[leg][1] = std::min(std::max(pos_impl[leg][1], -4.18879020479f), 1.0471975512f);
        // pos_impl[leg][2] = std::min(std::max(pos_impl[leg][2], 0.916297857297f), 2.69653369433f);
        pos_impl[leg][2] = std::min(std::max(pos_impl[leg][2], 0.0f), 2.69653369433f);
    }

}


void Recovering::wait_for_simulation_state(){
    // action_to_bullet << "[PHASE" << _phase << "] ";

    // std::ostringstream action_to_bullet;
    // std::cout << "SENDING....." << std::endl;
    // socket.send(zmq::buffer(action_to_bullet.str()), zmq::send_flags::none);
    // std::cout << "STATE DIMENSION IS "<< state_from_bullet.size() << std::endl;
    // assert(int(state_from_bullet.size()) == s_dim);
    const std::string data{"Hello"};

    for (auto request_num = 0; request_num < 10; ++request_num) 
    {
        // send the request message
        std::cout << "Sending Hello " << request_num << "..." << std::endl;
        socket.send(zmq::buffer(data), zmq::send_flags::none);
        
        // wait for reply from server
        zmq::message_t reply{};
        socket.recv(reply, zmq::recv_flags::none);

        std::cout << "Received " << reply.to_string(); 
        std::cout << " (" << request_num << ")";
        std::cout << std::endl;
        
    }


    socket.send(zmq::buffer(data), zmq::send_flags::none);


    // std::cout << "RECEIVING STATE......................" << std::endl;
    // socket.recv(request, zmq::recv_flags::none);
    // std::cout << "Received " << request.to_string() << std::endl;


    // state_from_bullet.clear();

    // std::stringstream ss(request.to_string());

    // for (float i; ss >> i;) {
    //     state_from_bullet.push_back(i);    
    //     if (ss.peek() == ',' || ss.peek() == ' ')
    //         ss.ignore();
    // }

    

	// std::cout << "[DEBUG] RECEIVED JPOS[0][1]: " << state[7] << std::endl;


}


// void Recovering::send_action_to_simulator(){
//     std::ostringstream action_to_bullet;
//     if (_phase >= 0)
//         action_to_bullet << "[PHASE" << _phase << "] ";
//     else
//         action_to_bullet << "[PHASE-] ";
//     // action_to_bullet << std::setprecision(4) << std::fixed;
//     for (int i=0; i<4; i++){
//         for (int j=0; j<3; j++){
//             action_to_bullet << this->_legController->commands[i].qDes[j];
//             if (!(i==3 && j==2))
//                 action_to_bullet << ", ";
//         }
//     }

//     socket.send(zmq::buffer(action_to_bullet.str()), zmq::send_flags::none);
//     // std::cout << "ACTION SENT TO BULLET: " << action_to_bullet.str() << std::endl;

// }


#if DEBUG
void Recovering::wait_for_simulation_state(){
    socket.recv(request, zmq::recv_flags::none);
    // std::cout << "Received " << request.to_string() << std::endl;


    state_from_bullet.clear();

    std::stringstream ss(request.to_string());

    for (float i; ss >> i;) {
        state_from_bullet.push_back(i);    
        if (ss.peek() == ',' || ss.peek() == ' ')
            ss.ignore();
    }
    // std::cout << "STATE DIMENSION IS "<< state_from_bullet.size() << std::endl;
    assert(int(state_from_bullet.size()) == s_dim);

	// std::cout << "[DEBUG] RECEIVED JPOS[0][1]: " << state[7] << std::endl;


}


void Recovering::send_action_to_simulator(){
    std::ostringstream action_to_bullet;
    if (_phase >= 0)
        action_to_bullet << "[PHASE" << _phase << "] ";
    else
        action_to_bullet << "[PHASE-] ";
    // action_to_bullet << std::setprecision(4) << std::fixed;
    for (int i=0; i<4; i++){
        for (int j=0; j<3; j++){
            action_to_bullet << this->_legController->commands[i].qDes[j];
            if (!(i==3 && j==2))
                action_to_bullet << ", ";
        }
    }

    socket.send(zmq::buffer(action_to_bullet.str()), zmq::send_flags::none);
    // std::cout << "ACTION SENT TO BULLET: " << action_to_bullet.str() << std::endl;

}

#endif