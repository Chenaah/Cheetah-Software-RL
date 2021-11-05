#include <Recovering.h>
// #include <pthread.h>
// #include <iostream>
// #include <unistd.h>


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

    if (action_mode == ActionMode::partial)
        a_dim = 4;
    else if (action_mode == ActionMode::whole)
        a_dim = 6;
    else if (action_mode == ActionMode::residual){
        if (leg_action_mode == LegActionMode::none)
            a_dim = 4;
        else if (leg_action_mode == LegActionMode::parameter)
            a_dim = 6;
        else if (leg_action_mode == LegActionMode::parallel_offset)
            a_dim = 6;
        else if (leg_action_mode == LegActionMode::hips_offset || leg_action_mode == LegActionMode::knees_offset)
            a_dim = 6;
        else if (leg_action_mode == LegActionMode::hips_knees_offset)
            a_dim = 8;
    }

    if (state_mode == StateMode::h_body_arm)
        s_dim = 14;
    else if (state_mode == StateMode::body_arm)
        s_dim = 10;
    else if (state_mode == StateMode::body_arm_p)
        s_dim = 12;
    else if (state_mode == StateMode::body_arm_leg_full)
        s_dim = 14;
    else if (state_mode == StateMode::body_arm_leg_full_p)
        s_dim = 16;
    
    //********* Read model
    graph = TF_NewGraph();
    status = TF_NewStatus();
    SessionOpts = TF_NewSessionOptions();
    RunOpts = NULL;

    char* homeDir = getenv("HOME");
    // const char* saved_model_dir = "/DAM/"; 
    // char* saved_model_dir_full = new char[strlen(homeDir) + strlen(saved_model_dir) + 1 + 1];
    // strcpy(saved_model_dir_full, homeDir);
    // strcat(saved_model_dir_full, saved_model_dir);
    glob_t globbuf;
    const char* dam_folder = "/DAM*/"; 
    char* dam_dir_full = new char[strlen(homeDir) + strlen(dam_folder) + 1 + 1];
    strcpy(dam_dir_full, homeDir);
    strcat(dam_dir_full, dam_folder);
    glob(dam_dir_full, 0, NULL, &globbuf);
    assert(globbuf.gl_pathc == 1);
    DAM_path = globbuf.gl_pathv[0];
    std::cout << "Deterministic Actro Model " << DAM_path << " FOUND!" <<std::endl;
    const char* checking_file = "saved_model.pb"; // only used for checking if RL agent is enabled
    char* checking_file_path = new char[strlen(DAM_path) + strlen(checking_file) + 1 + 1];
    strcpy(checking_file_path, DAM_path);
    strcat(checking_file_path, checking_file);

    std::ifstream model_exist(checking_file_path);
    if (model_exist.good()){
        std::cout << "TENSORFLOW MODEL FOUND! RL AGENT IS ENABLED. " << std::endl;
        agent_enable = true;

        const char* tags = "serve"; 
        int ntags = 1;
        sess = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, DAM_path, &tags, ntags, graph, NULL, status);
        
        if(TF_GetCode(status) == TF_OK)
            printf("TF_LoadSessionFromSavedModel OK\n");
        else
            printf("%s",TF_Message(status));

        //****** Get input tensor
        input_op = TF_Output{TF_GraphOperationByName(graph, "serving_default_input_1"), 0};
        if (input_op.oper == nullptr) {
            std::cout << "ERROR: Failed TF_GraphOperationByName serving_default_input_1" << std::endl;
        } else {
            printf("TF_GraphOperationByName serving_default_input_1 is OK\n");
        }
        
        //********* Get Output tensor
        out_op = TF_Output{TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};

        if(out_op.oper == nullptr)
            printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
        else
            printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

        //********* Allocate data for inputs & outputs

        const std::vector<std::int64_t> input_dims = {1, s_dim};
        std::vector<float> input_vals(s_dim, 0.5);
        input_tensor = CreateTensor(TF_FLOAT, input_dims, input_vals);
        printf("Allocated data ~ \n");

        output_tensor = nullptr;


    } else {
        agent_enable = false;
        std::cout << "TENSORFLOW MODEL NOT FOUND! RL AGENT IS DISABLED. " << std::endl;

    }

    action.resize(a_dim);
    state.resize(s_dim);

    leg_offsets << 0.f, 0.f, 0.f, 0.f;
    for (int leg = 0; leg < 4; leg++)
      pos_impl[leg] << 0.f, 0.f, 0.f;

    #if DEBUG
    socket.bind("tcp://*:5555");
    #endif

    std::cout << std::setprecision(3) << std::fixed;




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

    #if DEBUG
    wait_for_simulation_state();
    #endif
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
      case 2:
          _Prepare(_state_iter - _motion_start_iter);
          break;
      case 3:
          _Prepare_2(_state_iter - _motion_start_iter);
          break;
      case 4:
          _RearLegsActions(_state_iter - _motion_start_iter);
          break;
      case 5:
          _Walk(_state_iter - _motion_start_iter);
          break;
      case 6:
          _SettleDown(_state_iter - _motion_start_iter);
          break;
      case 7:
          _InverseRearLegsActions(_state_iter - _motion_start_iter);
          break;
      case 8:
          _InversePrepare_2(_state_iter - _motion_start_iter);
          break;
      case 9:
          _InverseJustStandUp(_state_iter - _motion_start_iter);
          break;
      case 10:
          _PushUp(_state_iter - _motion_start_iter);
          break;
      case 11:
          _Bound(_state_iter - _motion_start_iter);
          break;
      case 12:
          _BoundToStand(_state_iter - _motion_start_iter);
          break;
      case 13:
          _ClimbPre(_state_iter - _motion_start_iter);
          break;
      case 14:
          _ClimbPre1(_state_iter - _motion_start_iter);
          break;
      case 15:
          _Climb(_state_iter - _motion_start_iter);
          break;
    //   case 16:
    //       _Climb1(_state_iter - _motion_start_iter);
    //       break;
      case 17:
          _Climb2(_state_iter - _motion_start_iter);
          break;
      case 18:
          _Climb3(_state_iter - _motion_start_iter);
          break;
      case 19:
          _Climb4(_state_iter - _motion_start_iter);
          break;
    //   case 20:
    //       _Climb5(_state_iter - _motion_start_iter);
    //       break;
      case 21:
          _Pull(_state_iter - _motion_start_iter);
          break;
      case 22:
          _Pull1(_state_iter - _motion_start_iter);
          break;
      case 23:
          _Pull2(_state_iter - _motion_start_iter);
          break;
      case 24:
          _Pull3(_state_iter - _motion_start_iter);
          break;
      case 25:
          _RecoverOnTable(_state_iter - _motion_start_iter);
          break;
      
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


    // std::cout<< "====================== FOLD LEG BEGINS! ======================" << curr_iter <<std::endl;

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
    #if !DEBUG
    _phase = 11;  // DEVELOPING
    #else
    _phase = 1;
    #endif
    _motion_start_iter = _state_iter + 1;
  }
}

void Recovering::_JustStandUp(const int & curr_iter){

    // std::cout<< "STAND UP STEP: NO." << curr_iter <<std::endl;
    // std::cout<< "[1] STAND UP INI: " << initial_jpos[0][0] <<", "<< initial_jpos[0][1] <<", "<<  initial_jpos[0][2] << std::endl;
    // std::cout<< "[2] NOW: " << this->_legController->datas[0].q[0] <<", "<< this->_legController->datas[0].q[1] <<", "<<  this->_legController->datas[0].q[2] << std::endl;
    // std::cout<< "[3] STAND UP FIN: " << stand_jpos[0][0] <<", "<< stand_jpos[0][1] << ", "<< stand_jpos[0][2] << std::endl;

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
            initial_jpos[i] = this->_legController->datas[i].q;
        stand_jpos[0] << 0.f, stand_front_hip + front_hip_offset, stand_front_knee;
        stand_jpos[1] << 0.f, stand_front_hip + front_hip_offset, stand_front_knee;
        stand_jpos[2] << 0.f, stand_back_hip + back_hip_offset, stand_back_knee;
        stand_jpos[3] << 0.f, stand_back_hip + back_hip_offset, stand_back_knee;
    }

    for(size_t leg(0); leg<4; ++leg){
        _SetJPosInterPts(curr_iter, standup_ramp_iter, 
                         leg, initial_jpos[leg], stand_jpos[leg]);
    }       
    // feed forward mass of robot.
    //for(int i = 0; i < 4; i++)
    //this->_data->_legController->commands[i].forceFeedForward = f_ff;
    //Vec4<T> se_contactState(0.,0.,0.,0.);
    Vec4<float> se_contactState(0.5,0.5,0.5,0.5);
    this->_stateEstimator->setContactPhase(se_contactState);

    if(curr_iter >= standup_ramp_iter+100){
        // std::cout<< "====================== STAND UP FINISHED! ======================" << curr_iter <<std::endl;
        if (pre1_enable)
            _phase = 2;
        else
            _phase = 3;

        // _phase = 0; // JUST FOR TEST
        _motion_start_iter = _state_iter+1;
    } 

}


void Recovering::_Prepare(const int & curr_iter){

    // std::cout << "PREPARE STEP:  " << curr_iter << std::endl;

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i) {
        initial_jpos[i] = this->_legController->datas[i].q;
        }
    }

    if (pre_back_knee == 999) pre_back_knee = knee_angle;
    if (pre_back_hip == 999) pre_adaptive_hip = true;

    if (pre_adaptive_hip){

        Eigen::Quaterniond _orientation(_stateEstimator->getResult().orientation[0], _stateEstimator->getResult().orientation[1], _stateEstimator->getResult().orientation[2], _stateEstimator->getResult().orientation[3]);
        pitch = _toPitch(_orientation);
        adaptive_hip = -1.7 + PI/2 + pitch;
        adaptive_hip = _Box(adaptive_hip, -PI, 0.8);
        pre_back_hip = adaptive_hip;

    }


    prepare_jpos[0] << -ad1, pre_front_hip + front_hip_offset, pre_front_knee; //  -1.4f, 2.45f; 
    prepare_jpos[1] << ad1, pre_front_hip + front_hip_offset, pre_front_knee;  // -1.4f, 2.45f; 
    prepare_jpos[2] << -0.0f, pre_back_hip + back_hip_offset, pre_back_knee;
    prepare_jpos[3] << 0.0f, pre_back_hip + back_hip_offset, pre_back_knee;

    // std::cout << "[DEBUG] FRONT HIP: " << pre_front_hip <<  " FRONT KNEE: " << pre_front_knee << std::endl;
    // std::cout << "[DEBUG] BACK HIP: " << pre_back_hip <<  " BACK KNEE: " << pre_back_knee << std::endl;

    for(size_t i(0); i<4; ++i){
      _SetJPosInterPts(curr_iter, 1500, i, 
      initial_jpos[i], prepare_jpos[i]);
    }
    

    if(curr_iter >= 1500 + 100){
        _phase = 3;
        for(size_t i(0); i<4; ++i){
          this->jointPDControl(i, prepare_jpos[i], zero_vec3);
        }
        _motion_start_iter = _state_iter + 1;
    }
}

void Recovering::_Prepare_2(const int & curr_iter){

    // std::cout << "PREPARE 2" << std::endl;


    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i) {
            initial_jpos[i] = this->_legController->datas[i].q;
        }
    }


    if (curr_iter % 200 == 0){
        // Eigen::Quaterniond _orientation(_stateEstimator->getResult().orientation[0], _stateEstimator->getResult().orientation[1], _stateEstimator->getResult().orientation[2], _stateEstimator->getResult().orientation[3]);
        // W, X, Y, Z
        // pitch = _toPitch(_orientation);
        _update_rpy();
        pitch = rpy[1];
        std::cout << "[PRE2] THE PITCH (ALPHA) IS " << pitch << std::endl;
        std::cout << "[PRE2] THE POSITION OF THE BACK HIP (THETA1) IS " << _legController->datas[2].q[1] << ", " << _legController->datas[3].q[1]  << std::endl;
        std::cout << "[PRE2] THE POSITION OF THE BACK KNEE (THETA2) IS " << _legController->datas[2].q[2] << ", " << _legController->datas[3].q[2]  << std::endl;
        std::cout << "[PRE2] THE POSITION OF THE FRONT HIP (THETA1_PRIME) IS " << _legController->datas[0].q[1] << ", " << _legController->datas[1].q[1]  << std::endl;
        std::cout << "[PRE2] THE POSITION OF THE FRONT KNEE (THETA2_PRIME) IS " << _legController->datas[0].q[2] << ", " << _legController->datas[1].q[2]  << std::endl;
    }
    


    prepare_jpos[0] << -ad1, pre2_front_hip + front_hip_offset, pre2_front_knee; //  -1.4f, 2.45f; 
    prepare_jpos[1] << ad1, pre2_front_hip + front_hip_offset, pre2_front_knee;  // -1.4f, 2.45f; 
    prepare_jpos[2] << -0.f, pre2_back_hip + back_hip_offset, pre2_back_knee;
    prepare_jpos[3] << 0.f, pre2_back_hip + back_hip_offset, pre2_back_knee;



    for(size_t i(0); i<4; ++i){
      _SetJPosInterPts(curr_iter, 1200, i, 
      initial_jpos[i], prepare_jpos[i]);
    }

    if(curr_iter >= 1200 + 100){
        _phase = 4;
        for(size_t i(0); i<4; ++i){
          this->jointPDControl(i, prepare_jpos[i], zero_vec3);
        }
        _motion_start_iter = _state_iter + 1;
    }
    
}



void Recovering::_RearLegsActions(const int & curr_iter){

    if (curr_iter==0){
      #if !DEBUG
      for(size_t i(0); i < 4; ++i) {
          initial_jpos[i] = this->_legController->datas[i].q;
      }
      #else
        // TODO: THIS ONLY WORKS FOR 14-DIM STATE
        initial_jpos[0] << state_from_bullet[6], state_from_bullet[7], _theta2_prime_hat(state_from_bullet[7]);
        initial_jpos[1] << state_from_bullet[8], state_from_bullet[9], _theta2_prime_hat(state_from_bullet[9]);
        initial_jpos[2] << 0.f, state_from_bullet[10], _theta2_prime_hat(state_from_bullet[11]);
        initial_jpos[2] << 0.f, state_from_bullet[12], _theta2_prime_hat(state_from_bullet[13]);
      #endif

      for(size_t i(0); i < 2; ++i)
        initial_jpos[i] = initial_jpos[i] - front_offset;
      for(size_t i(2); i < 4; ++i)
        initial_jpos[i] = initial_jpos[i] - back_offset;
      
      

      theta1 = pre2_back_hip;  
      theta2 = pre2_back_knee;  // -1
    }

    if (theta2 < -1.0)
        theta2 += 0.0005;
    // if (theta2 < -1.5)
    //     theta2 += 0.0005;
    if (k_ < k_final)
        k_ += 0.0003;
    // std::cout << "K: " << k_ << std::endl;
    // std::cout << "CALCULATING THE ACOS OF " << (l_/2 + b_*cos(PI - (-theta2) - gamma_) -c_*cos(gamma_)) <<  "/" << k_ << "/" << a_ << " = "<< acos((l_/2 + b_*cos(PI - (-theta2) - gamma_) -c_*cos(gamma_)) / k_ / a_)  << std::endl;

    theta1 = _theta1_hat(theta2); // -PI/2 + (-theta2) + gamma_ - acos((l_/2 + b_*cos(PI - (-theta2) - gamma_) -c_*cos(gamma_)) / k_ / a_);
   //  std::cout << "THETA1: " << theta1 << "    THETA2: " << theta2 << std::endl;
    // Eigen::Quaterniond _orientation(_stateEstimator->getResult().orientation[0], _stateEstimator->getResult().orientation[1], _stateEstimator->getResult().orientation[2], _stateEstimator->getResult().orientation[3]);
    _update_rpy();
    pitch = rpy[1];

    std::cout << "[DEBUG] THE CALCULATED ALPHA: " << _alpha() << std::endl; // THSI SHOULD BE THE NEGATIVE OF THE PITCH

    if (curr_iter % 200 == 0){
        std::cout << "[STANDING] THE PITCH (ALPHA) IS " << pitch << std::endl;
        std::cout << "[STANDING]   --  THE ORIENTATION IS " << _stateEstimator->getResult().orientation[0] << ", "<< _stateEstimator->getResult().orientation[1]<< ", "<< _stateEstimator->getResult().orientation[2]<< ", "<< _stateEstimator->getResult().orientation[3] << std::endl;
        std::cout << "[STANDING] THE POSITION OF THE BACK HIP (THETA1) IS " << _legController->datas[2].q[1] << ", " << _legController->datas[3].q[1]  << std::endl;
        std::cout << "[STANDING] THE POSITION OF THE BACK KNEE (THETA2) IS " << _legController->datas[2].q[2] << ", " << _legController->datas[3].q[2]  << std::endl;
    }
    

    // std::cout << "TH1: " << bullet_q[10] - back_hip_offset << ", TH2: " << bullet_q[11] << " ( k = " << k_ << ")  -->  PITCH: " << pitch << "    HEIGHT: " << _stateEstimator->getResult().position[2] << std::endl;
    // std::cout << "JPOS: [" << bullet_q[0] << ", " << bullet_q[1] << ", " << bullet_q[2] << ", " << bullet_q[3] << ", " << bullet_q[4] << ", "<< bullet_q[5] << ", "<< bullet_q[6] << ", "<< bullet_q[7] << ", "<< bullet_q[8] << ", "<< bullet_q[9] << ", "<< bullet_q[10] << ", " << bullet_q[11] << "]" << std::endl;
    prepare_jpos[2] << -0.0f, theta1 + back_hip_offset, theta2;
    prepare_jpos[3] << 0.0f, theta1 + back_hip_offset, theta2;

    for(size_t i(0); i<4; ++i){
      this->jointPDControl(i, prepare_jpos[i], zero_vec3);
    }

    prepare_jpos[0] << -0.0f, -PI/2, _theta2_prime_hat(-PI/2);
    prepare_jpos[1] << 0.0f, -PI/2, _theta2_prime_hat(-PI/2);

    for(size_t leg(0); leg<2; ++leg){
        _Step(curr_iter, 1000, 
        leg, initial_jpos[leg], prepare_jpos[leg]);
    }
    // #if DEBUG
    // std::cout << "[DEBUG] ARMS: " << bullet_q[0]  << ", "<< bullet_q[1] - front_hip_offset<< ", "<< bullet_q[3] <<", "<< bullet_q[4] - front_hip_offset <<"] "<<std::endl; 
    // #endif
    int stand_time = 250;   // TODO DEBUG
    // if (!walk_enable)
    //     stand_time = 9999;

    if(curr_iter >= 2000 + stand_time){
        // Eigen::Quaterniond _orientation_(_stateEstimator->getResult().orientation[0], _stateEstimator->getResult().orientation[1], _stateEstimator->getResult().orientation[2], _stateEstimator->getResult().orientation[3]);
        // std::vector<float> rpy = _toRPY(_orientation_);
        // _update_rpy();
        // pitch_ref = rpy[1];
        _update_state();
        pitch_ref = state[1];
        // pitch_ref = -1.34066211;  // DEVELOPING: TRY TO CORRECT REFERENCE PITCH
        std::cout << "[STANDING] ORIGINAL PITCH REFERENCE: " << state[1] << "  AFTER CORRECTION: " << pitch_ref << std::endl;
        if (walk_enable)
            _phase = 5;
        else
            // _phase = 6;
            _phase = 13;   // TODO DEBUG


        _motion_start_iter = _state_iter + 1;
    }


}

float Recovering::_theta2_prime_hat(const float & th1_prime){
    if (th1_prime > -2.3023)
        return -0.6354*th1_prime;
    else
        return 1.7430*th1_prime+5.4759;
}

float Recovering::_theta1_hat(const float & th2){
    return -PI/2 + (-th2) + gamma_ - acos((l_/2 + b_*cos(PI - (-th2) - gamma_) -c_*cos(gamma_)) / k_ / a_);
}

float Recovering::_beta(){
    // for(size_t i(0); i < 4; ++i)
    //     initial_jpos[i] = this->_legController->datas[i].q;
    // for(size_t i(0); i < 2; ++i)
    //     initial_jpos[i] = initial_jpos[i] - front_offset;
    // for(size_t i(2); i < 4; ++i)
    //     initial_jpos[i] = initial_jpos[i] - back_offset;
            
    float th2_curr = (this->_legController->datas[2].q[2] + this->_legController->datas[3].q[2]) / 2;
    return PI + th2_curr - gamma_;
}

float Recovering::_alpha(){

    float th1_curr = (this->_legController->datas[2].q[1] + this->_legController->datas[3].q[1]) / 2;
    return PI/2 - th1_curr - _beta();

}

float Recovering::_beta(const float & th2){

    return PI + th2 - gamma_;

}

float Recovering::_alpha(const float & th1, const float & th2){

    return PI/2 - th1 - _beta(th2);

}



float Recovering::_head_height(){

    float h_1 = c_*sin(gamma_);
    float h_2 = b_*sin(_beta());
    float h_3 = a_*sin(_alpha());
    float table = table_global;
    
    // return (h_1 + h_2 + h_3) / cos(arm_ab) - table;
    return (h_1 + h_2 + h_3 - table) / cos(arm_ab);

}

float Recovering::_head_height(const float & th1, const float & th2, const float & arm_ab_arg, const float & table){

    float h_1 = c_*sin(gamma_);
    float h_2 = b_*sin(_beta(th2));
    float h_3 = a_*sin(_alpha(th1, th2));

    // return (h_1 + h_2 + h_3) / cos(arm_ab_arg) - table;
    return (h_1 + h_2 + h_3 - table) / cos(arm_ab_arg);

}



void Recovering::_arms_IK(const float & x, float & th1_prime, float & th2_prime){

    float y = _head_height();
    float foracos = (pow(x, 2) + pow(y, 2) - pow(b_, 2) - pow(c_, 2)) / (2*b_*c_);
    // std::cout << "[DEBUG] foracos: " << foracos << std::endl;
    assert(foracos > -1 && foracos < 1);
    float theta2_ik = -acos(foracos);
    float eq_b = 2*c_*sin(theta2_ik)*y;
    float eq_a = pow(c_, 2)*pow(sin(theta2_ik), 2) + pow((c_*cos(theta2_ik) + b_), 2);
    float eq_c = pow(y, 2) - pow((c_*cos(theta2_ik) + b_), 2);
    float theta1_ik;

    if (pow(eq_b, 2) - 4*eq_a*eq_c >= 0){
        // float sin_theta1_ik_1 = (-eq_b + sqrt(pow(eq_b, 2) - 4*eq_a*eq_c)) / (2*eq_a);
        float sin_theta1_ik_2 = (-eq_b - sqrt(pow(eq_b, 2) - 4*eq_a*eq_c)) / (2*eq_a);
        // float theta1_ik_1 = asin(sin_theta1_ik_1);
        float theta1_ik_2 = asin(sin_theta1_ik_2);
        // # print("TWO THETA1_PRIME: {:.2f} AND {:.2f}".format(theta1_ik_1, theta1_ik_2))
        theta1_ik = theta1_ik_2;  // CHHOSE A RIGHT ONE !
    } else if (fabs(pow(eq_b, 2) - 4*eq_a*eq_c) < 0.001){
        float sin_theta1_ik = (-eq_b + sqrt(0)) / (2*eq_a);
        theta1_ik = asin(sin_theta1_ik);

    } else {
        // print(f"{eq_b}**2 - 4*{eq_a}*{eq_c} = {eq_b**2 - 4*eq_a*eq_c} < 0 !!")
        std::cout << "[ARM IK] SHIT! IK CAN NOT BE SOLVED..." << std::endl;
        theta1_ik = 0;
    }

    th2_prime = -theta2_ik;
    th1_prime = -(_alpha() + theta1_ik);

}

void Recovering::_arms_IK(const float & x, const float & ab, const float & th1, const float & th2, const float & table, float & th1_prime, float & th2_prime){

    float y = _head_height(th1, th2, ab, table);
    float foracos = (pow(x, 2) + pow(y, 2) - pow(b_, 2) - pow(c_, 2)) / (2*b_*c_);
    // std::cout << "[DEBUG] foracos: " << foracos << std::endl;
    assert(foracos > -1 && foracos < 1);
    float theta2_ik = -acos(foracos);
    float eq_b = 2*c_*sin(theta2_ik)*y;
    float eq_a = pow(c_, 2)*pow(sin(theta2_ik), 2) + pow((c_*cos(theta2_ik) + b_), 2);
    float eq_c = pow(y, 2) - pow((c_*cos(theta2_ik) + b_), 2);
    float theta1_ik;

    if (pow(eq_b, 2) - 4*eq_a*eq_c >= 0){
        // float sin_theta1_ik_1 = (-eq_b + sqrt(pow(eq_b, 2) - 4*eq_a*eq_c)) / (2*eq_a);
        float sin_theta1_ik_2 = (-eq_b - sqrt(pow(eq_b, 2) - 4*eq_a*eq_c)) / (2*eq_a);
        // float theta1_ik_1 = asin(sin_theta1_ik_1);
        float theta1_ik_2 = asin(sin_theta1_ik_2);
        // # print("TWO THETA1_PRIME: {:.2f} AND {:.2f}".format(theta1_ik_1, theta1_ik_2))
        theta1_ik = theta1_ik_2;  // CHHOSE A RIGHT ONE !
    } else if (fabs(pow(eq_b, 2) - 4*eq_a*eq_c) < 0.001){
        float sin_theta1_ik = (-eq_b + sqrt(0)) / (2*eq_a);
        theta1_ik = asin(sin_theta1_ik);

    } else {
        // print(f"{eq_b}**2 - 4*{eq_a}*{eq_c} = {eq_b**2 - 4*eq_a*eq_c} < 0 !!")
        std::cout << "[ARM IK] SHIT! IK CAN NOT BE SOLVED..." << std::endl;
        theta1_ik = 0;
    }

    th2_prime = -theta2_ik;
    th1_prime = -(_alpha(th1, th2) + theta1_ik);

}

void Recovering::_legs_IK(const float & x, const float & ab, const float & head_height, const float & pitch_arg, float & th1, float & th2){

    float alpha_ = -pitch_arg;
    float y = (head_height - a_ * sin(alpha_)) / cos(ab);
    float foracos = (pow(x, 2) + pow(y, 2) - pow(b_, 2) - pow(c_, 2)) / (2*b_*c_);
    // std::cout << "[DEBUG] foracos: " << foracos << std::endl;
    assert(foracos > -1 && foracos < 1);
    float theta2_ik = -acos(foracos);
    float eq_b = 2*c_*sin(theta2_ik)*y;
    float eq_a = pow(c_, 2)*pow(sin(theta2_ik), 2) + pow((c_*cos(theta2_ik) + b_), 2);
    float eq_c = pow(y, 2) - pow((c_*cos(theta2_ik) + b_), 2);
    float theta1_ik;

    if (pow(eq_b, 2) - 4*eq_a*eq_c >= 0){
        // float sin_theta1_ik_1 = (-eq_b + sqrt(pow(eq_b, 2) - 4*eq_a*eq_c)) / (2*eq_a);
        float sin_theta1_ik_2 = (-eq_b - sqrt(pow(eq_b, 2) - 4*eq_a*eq_c)) / (2*eq_a);
        // float theta1_ik_1 = asin(sin_theta1_ik_1);
        float theta1_ik_2 = asin(sin_theta1_ik_2);
        // # print("TWO THETA1_PRIME: {:.2f} AND {:.2f}".format(theta1_ik_1, theta1_ik_2))
        theta1_ik = theta1_ik_2;  // CHHOSE A RIGHT ONE !
    } else if (fabs(pow(eq_b, 2) - 4*eq_a*eq_c) < 0.001){
        float sin_theta1_ik = (-eq_b + sqrt(0)) / (2*eq_a);
        theta1_ik = asin(sin_theta1_ik);

    } else {
        // print(f"{eq_b}**2 - 4*{eq_a}*{eq_c} = {eq_b**2 - 4*eq_a*eq_c} < 0 !!")
        std::cout << "[ARM IK] SHIT! IK CAN NOT BE SOLVED..." << std::endl;
        theta1_ik = 0;
    }

    th2 = -theta2_ik;
    th1 = -(alpha_ + theta1_ik);

}

float Recovering::_delta_th(const float & butt_h, const float & delta_h){

    // from delta h to delta theta
    // h = -b sin(theta -gamma)  (when the foot are fully touching the floor)
    // theta = asin(-h/b) + gamma
    return asin(-(butt_h-c_*sin(gamma_) + delta_h)/b_) - asin(-(butt_h - c_*sin(gamma_))/b_);

}


float Recovering::_butt_height(const float & th2){

    return b_ * cos(-PI/2 - th2 + gamma_) + c_*sin(gamma_);
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
        
            
        t_start = std::chrono::high_resolution_clock::now();

        if (!progressing)
            param_a_buffered = std::min(std::max(param_opt[3]*param_opt[0], A_range[0]), A_range[1]);
        else 
            param_a_buffered = 0;

        if (progressing){
            std::cout << "[PROGRESSING] PROGRESSING CONTROL IS ENABLE" << std::endl;
            assert(param_a_buffered == 0);
        }

        
        param_b_buffered = std::min(std::max(param_opt[0], B_range[0]), B_range[1]);
        std::cout << "[WALKING] A: " << param_a_buffered << "  B: " << param_b_buffered << std::endl;

        param_a = param_a_buffered;
        param_b = param_b_buffered;

        Kp_pitch = param_opt[4];
        Kd_pitch = param_opt[5];
        Kp_yaw = param_opt[6];
        Kd_yaw = param_opt[7];
        delta_x = param_opt[8];
        delta_x_buffered = delta_x;
        // variables above should be initialised only after updating the param_opt from files

        _FK(theta1, theta2, x_original, y_original);
        // exit(1);

        t_start = std::chrono::high_resolution_clock::now();

    }

    t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "THE TIME PERIOD IS " <<  elapsed_time_ms << " /1000 SEC" << std::endl;
    t_start = t_end;

    // t_end = std::chrono::high_resolution_clock::now();
    // double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    // std::cout << "CONTROL FREQUENCY IS " <<  (float)elapsed_time_ms/1000 << " S" << std::endl;
    // t_start = t_end; 

    #if !DEBUG
    for(size_t i(0); i < 4; ++i) {
      initial_jpos[i] = this->_legController->datas[i].q;
    }
    #else
    // TODO: THIS ONLY WORKS FOR 14-DIM STATE
    initial_jpos[0] << state_from_bullet[6], state_from_bullet[7], _theta2_prime_hat(state_from_bullet[7]);
    initial_jpos[1] << state_from_bullet[8], state_from_bullet[9], _theta2_prime_hat(state_from_bullet[9]);
    initial_jpos[2] << 0.f, state_from_bullet[10], _theta2_prime_hat(state_from_bullet[11]);
    initial_jpos[2] << 0.f, state_from_bullet[12], _theta2_prime_hat(state_from_bullet[13]);
    #endif

    for(size_t i(0); i < 2; ++i)
        initial_jpos[i] = initial_jpos[i] - front_offset;
    for(size_t i(2); i < 4; ++i)
        initial_jpos[i] = initial_jpos[i] - back_offset;
    
    // _update_state_buffer();
    
    if (curr_iter % num_sub_steps == 0){
        if (curr_iter == 0){
            std::cout << "[WALKING] FIRST STATE: [" << state[0] << ", " << state[1] << ", " << state[2] << ", " << state[3] << ", " << state[4] << ", " << state[5] << ", " << state[6] << ", " << state[7] << ", " << state[8] << ", " << state[9] << ", " << state[10] << ", " << state[11] << ", " << state[12] << ", " << state[13] << "]" << std::endl;
            // std::cout << "QUAT: [" << bullet_orientation[0] << ", " << bullet_orientation[1] << ", "<< bullet_orientation[2] << ", "<< bullet_orientation[3] << "]" << std::endl;
        
        }
        _update_state();

        if (curr_iter != 0 && !_done && curr_iter <= 10000)
            _update_reward();
        else if (curr_iter != 0)
            _log_return();
        
        // std::cout << "PITCH AFTER RESET:::::::::::::::::::: " <<  state[1] << std::endl;
        // pthread_t tid[1];
        // pthread_create(&tid[0], NULL, _update_action_worker, this);
        #if DEBUG
        // debug_info_stream << "STATE: [" << state[0] << ", " << state[1] << ", " << state[2] << ", " << state[3] << ", " << state[4] << ", " << state[5] << ", \n        " << 
        //                                    state[6] << ", " << state[7] << ", " << state[8] << ", " << state[9] << ", \n        " << 
        //                                    state[10] << ", " << state[11] << ", " << state[12] << ", " << state[13] << "]" << std::endl;
        #endif
        

        if (agent_enable){
            if (!stopping)
                _update_action();
            else {
                for (auto & a: action){
                    if (fabs(a) >= 0.001)
                        a = a/fabs(a)*(fabs(a)-0.001);
                    else
                        a = 0;
                }
            }
        } else
            action = {0, 0, 0, 0, 0, 0};

        
        #if DEBUG
        // debug_info_stream << "AGENT ACTION: [" <<  action[0] << ", " <<  action[1] << ", " <<  action[2] << ", "<<  action[3] << ", "<<  action[4] << ", "<<  action[5]  << "] " << "\n";
        #endif

        leg_offsets_old = leg_offsets;
        _update_leg_offsets();  // leg_offsets would be overwritten

        if (arm_pd_control){
            if (!stopping){
                param_c_buffered = - (Kp_pitch * p_error + Kd_pitch * d_error);
                param_d_buffered = - (Kp_yaw * p_error_yaw + Kd_yaw * d_error_yaw);
            } else {
                param_c_buffered = std::max(param_c_buffered-0.001, 0.0);
                param_d_buffered = std::max(param_d_buffered-0.001, 0.0);
            }
            std::cout << "[DEBUG] ACTION C: "<< param_c_buffered << " (ERROR P=" << p_error<< ") (ERROR D=" << d_error << ")" <<std::endl;
            std::cout << "[DEBUG] ACTION D: "<< param_d_buffered << " (ERROR P=" << p_error_yaw<< ") (ERROR D=" << d_error_yaw << ")" <<std::endl;
        }

    }

    if (fast_error_update){

        std::cout << "[DEBUG] D ERROR = " << _stateEstimator->getResult().omegaBody[1] << std::endl;
        assert(!std::isnan(_stateEstimator->getResult().omegaBody[1]));

        d_error_buffer.push_back(_stateEstimator->getResult().omegaBody[1]);
        d_error_sum += _stateEstimator->getResult().omegaBody[1];
        // std::cout << "[DEBUG] d_error_sum: "<< d_error_sum << " after adding " << _stateEstimator->getResult().omegaBody[1] << std::endl;
        if (d_error_buffer.size() > error_buffer_size){
            d_error_sum -= d_error_buffer[0];
            d_error_buffer.erase(d_error_buffer.begin());
        } 
        if (d_error_buffer.size() > 2) {
            float d_max_error = *std::max_element(d_error_buffer.begin(), d_error_buffer.end());
            float d_min_error = *std::min_element(d_error_buffer.begin(), d_error_buffer.end());
            d_error = (d_error_sum - d_max_error - d_min_error) / (d_error_buffer.size() - 2);
        } else {
            d_error = d_error_sum / d_error_buffer.size();
        }
        
        std::cout << "[DEBUG] P ERROR = " << state[1] << " - " << pitch_ref << " = " << state[1] - pitch_ref << std::endl;

        p_error_buffer.push_back(state[1] - pitch_ref);
        p_error_sum += state[1] - pitch_ref;
        if (p_error_buffer.size() > error_buffer_size){
            p_error_sum -= p_error_buffer[0];
            p_error_buffer.erase(p_error_buffer.begin());
        } 
        if (p_error_buffer.size() > 2) {
            float p_max_error = *std::max_element(p_error_buffer.begin(), p_error_buffer.end());
            float p_min_error = *std::min_element(p_error_buffer.begin(), p_error_buffer.end());
            p_error = (p_error_sum - p_max_error - p_min_error) / (p_error_buffer.size() - 2);
        } else {
            p_error = p_error_sum / p_error_buffer.size();
        }

        if (progressing){

            // PROGRESSING CONTROL
            abs_d_error_buffer.push_back(fabs(_stateEstimator->getResult().omegaBody[1]));
            abs_d_error_sum += fabs(_stateEstimator->getResult().omegaBody[1]);
            if (abs_d_error_buffer.size() > progressing_buffer_size){
                abs_d_error_sum -= abs_d_error_buffer[0];
                abs_d_error_buffer.erase(abs_d_error_buffer.begin());
            } 
            abs_d_error = abs_d_error_sum / abs_d_error_buffer.size();

            abs_p_error_buffer.push_back(fabs(state[1] - pitch_ref));
            abs_p_error_sum += fabs(state[1] - pitch_ref);
            if (abs_p_error_buffer.size() > progressing_buffer_size){
                abs_p_error_sum -= abs_p_error_buffer[0];
                abs_p_error_buffer.erase(abs_p_error_buffer.begin());
            } 
            abs_p_error = abs_p_error_sum / abs_p_error_buffer.size();

            if (curr_iter > 200 && abs_d_error < 0.35 && abs_p_error < 0.15){
                if (progressing_param_a_multiplier < param_opt[3]) {
                    if (progressing_param_a_multiplier < param_opt[3]*0.2)
                        progressing_param_a_multiplier += 0.001;
                    else if (progressing_param_a_multiplier < param_opt[3]*0.4)
                        progressing_param_a_multiplier += 0.004; // DEVELOPING
                    else
                        progressing_param_a_multiplier += 0.008;

                }
                if (progressing_agent_factor < agent_factor && 
                    progressing_param_a_multiplier > param_opt[3]*0.8)  // otherwise the progressing_param_a_multiplier may stop increasing due to unstability
                    progressing_agent_factor += 0.01;

                else if (progressing_agent_factor < agent_factor &&  gait == Gait::none)
                    progressing_agent_factor += 0.01;
                
                if (progressing_agent_factor > agent_factor)
                    progressing_agent_factor = agent_factor;

                if (curr_iter % 50)
                    std::cout << "[PROGRESSING] I FEEL GOOD! THE PROGRESSING A MULTIPLIER IS NOW " << progressing_param_a_multiplier << " AND THE AGENT FACTOR IS " << progressing_agent_factor << std::endl;

            } else {
                if (curr_iter % 50)
                    std::cout << "[PROGRESSING] EMMM..... THE D ERROR IS NOW " << abs_d_error << " AND THE P ERROR IS NOW " << abs_p_error <<  " THE PROGRESSING A MULTIPLIER IS NOW " << progressing_param_a_multiplier << " AND THE AGENT FACTOR IS " << progressing_agent_factor << std::endl;
            }
            // progressing_param_a_multiplier = 0;  
            ////////

        }

        // std::cout << "[DEBUG] param_a: " << param_a << " delta_x: "<< delta_x << std::endl;

        d_error_yaw_buffer.push_back(_stateEstimator->getResult().omegaBody[2]);
        d_error_yaw_sum += _stateEstimator->getResult().omegaBody[2];
        if (d_error_yaw_buffer.size() > error_buffer_size){
            d_error_yaw_sum -= d_error_yaw_buffer[0];
            d_error_yaw_buffer.erase(d_error_yaw_buffer.begin());
        } 
        d_error_yaw = d_error_yaw_sum / d_error_yaw_buffer.size();

        p_error_yaw_buffer.push_back(state[2]);
        p_error_yaw_sum += state[2];
        if (p_error_yaw_buffer.size() > error_buffer_size){
            p_error_yaw_sum -= p_error_yaw_buffer[0];
            p_error_yaw_buffer.erase(p_error_yaw_buffer.begin());
        } 
        p_error_yaw = p_error_yaw_sum / p_error_yaw_buffer.size();

        // std::cout << "P ERROR: " << p_error << "  D ERROR: " << d_error << std::endl;

    }
    if (!stopping){
        if (action_mode == ActionMode::whole){
            param_a_buffered = (action[4]+1)/2*0.9+0.1;
            param_a_buffered = std::min(std::max(param_a_buffered, A_range[0]), A_range[1]);
            param_b_buffered = (action[5]+1)/2*0.09+0.01;
            param_b_buffered = std::min(std::max(param_b_buffered, B_range[0]), B_range[1]);
        } else if (action_mode == ActionMode::partial){
            param_b_buffered = param_opt[0] + param_opt[1]*p_error + param_opt[2]*d_error;
            param_b_buffered = std::min(std::max(param_b_buffered, B_range[0]), B_range[1]);
            param_a_buffered = param_b_buffered*param_opt[3];
            param_a_buffered = std::min(std::max(param_a_buffered, A_range[0]), A_range[1]);
            std::cout << "ACTION A: " << param_a << "  ACTION B: " << param_b << std::endl;
        } else if (action_mode == ActionMode::residual && leg_action_mode == LegActionMode::parameter){
            param_b_buffered = param_opt[0] + param_opt[1]*p_error + param_opt[2]*d_error + action[4]*residual_multiplier;
            param_b_buffered = std::min(std::max(param_b_buffered, B_range[0]), B_range[1]);
            param_a_buffered = param_b_buffered*param_opt[3]  + action[5]*residual_multiplier;
            param_a_buffered = std::min(std::max(param_a_buffered, A_range[0]), A_range[1]);
        } else if (action_mode == ActionMode::open_loop){
            std::cout << "ACTION A: " << param_a << "  ACTION B: " << param_b << std::endl;
        }

        if (progressing)
            // param_a_buffered = param_b_buffered*progressing_param_a_multiplier + action[5]*residual_multiplier;  // TODO: THE RESIDUAL PART CAN CHANGE THE RANGE
            // note that if "param_opt[0]" is used as a factor than paramter A will not change with parameter B
            param_a_buffered = std::min(std::max(param_opt[0]*progressing_param_a_multiplier + action[5]*residual_multiplier, A_range[0]), A_range[1]);


    } else {
        param_a_buffered = std::max(param_a_buffered-0.001, 0.0);
        param_b_buffered = std::max(param_a_buffered-0.001, 0.0);
        if (curr_iter % 50)
            std::cout << "[STOP] Trying to stop, the param_a_buffered has decreased to " << param_a_buffered << std::endl; 
    }

    param_a_buffer.push_back(param_a_buffered);
    param_b_buffer.push_back(param_b_buffered);

    float normalised_abduct_right, normalised_abduct_left;

    if (initial_jpos[0][1] + action_multiplier*action[1] > -PI/2 - 0.174){
        ad_right_min = -PI/3;
        ad_right_max = 0;
        normalised_abduct_right = initial_jpos[0][0] + (action_multiplier*action[0] + arm_pd_multiplier*param_d_buffered);
    } else {
        ad_right_min = 0;
        ad_right_max = PI/3;
        normalised_abduct_right = initial_jpos[0][0] - (action_multiplier*action[0] + arm_pd_multiplier*param_d_buffered);
    }
        
    if (initial_jpos[1][1] + action_multiplier*action[3] > -PI/2 - 0.174){
        ad_left_min = 0;
        ad_left_max = PI/3;
        normalised_abduct_left = initial_jpos[1][0] + (action_multiplier*action[2] + arm_pd_multiplier*param_d_buffered);
    } else {
        ad_left_min = -PI/3;
        ad_left_max = 0;
        normalised_abduct_left = initial_jpos[1][0] - (action_multiplier*action[2] + arm_pd_multiplier*param_d_buffered);
    }
    
    // std::cout << "[DEBUG] [I] pos_impl[0][1] = " << initial_jpos[0][1] << " + " << action_multiplier << " * " << action[1] << " + " << arm_pd_multiplier*param_c_buffered << " = " << initial_jpos[0][1] + action_multiplier*action[1] + arm_pd_multiplier*param_c_buffered << std::endl;
    
    pos_impl[0] << _Box(normalised_abduct_right, ad_right_min, ad_right_max), 
                   _Box(initial_jpos[0][1] + action_multiplier*action[1] + arm_pd_multiplier*param_c_buffered, -PI, 0), 
                   _Box(_theta2_prime_hat(_Box(initial_jpos[0][1] + action_multiplier*action[1] + arm_pd_multiplier*param_c_buffered, -PI, 0)), -PI/2, PI/2);
    pos_impl[1] << _Box(normalised_abduct_left, ad_left_min, ad_left_max),
                   _Box(initial_jpos[1][1] + action_multiplier*action[3] + arm_pd_multiplier*param_c_buffered, -PI, 0), 
                   _Box(_theta2_prime_hat(_Box(initial_jpos[1][1] + action_multiplier*action[3] + arm_pd_multiplier*param_c_buffered, -PI, 0)), -PI/2, PI/2);
    pos_impl[2] << 0, theta1, theta2;
    pos_impl[3] << 0, theta1, theta2;
    // pos_impl[2] << 0, theta1 + param_a*sin(param_b*curr_iter), theta2 - param_a*sin(param_b*curr_iter);

    // if (old_sin_value*sin(param_b*curr_iter-PI/2) <= 0){
    //     param_a = std::accumulate(param_a_buffer.begin(), param_a_buffer.end(), 0.0) / (float)param_a_buffer.size();
    //     param_b = std::accumulate(param_b_buffer.begin(), param_b_buffer.end(), 0.0) / (float)param_b_buffer.size();
    //     param_a_buffer.clear();
    //     param_b_buffer.clear();
    // }

    old_sin_value = sin(param_b*curr_iter-PI/2);

    

    if (gait != Gait::none)
        _update_basic_leg_actions(curr_iter); 

    // this will rewrite pos_impl[2][1], pos_impl[2][2], pos_impl[3][1], and pos_impl[3][2]

    _process_leg_offsets(curr_iter);
    // this will change the values of pos_impl[2][1], pos_impl[2][2], pos_impl[3][1], and pos_impl[3][2]

    assert((unsigned int)pos_impl[2][0] == 0 && (unsigned int)pos_impl[3][0] == 0);

    for (size_t leg(0); leg<4; ++leg)
        for (size_t j(0); j<3; ++j)
            assert(!std::isnan(pos_impl[leg][j]));

    if (curr_iter % 50){
        std::cout << "[CONTROL] FINAL ARM ACTION (OBJ): [" <<  pos_impl[0][0] << ", " <<  pos_impl[0][1] << ", " <<  pos_impl[1][0] << ", "<<  pos_impl[1][1] << "] " << std::endl;
        std::cout << "[CONTROL] FINAL LEG ACTION: [" <<  pos_impl[2][1] << ", " <<  pos_impl[2][2] << ", " <<  pos_impl[3][1] << ", "<<  pos_impl[3][2] << "] " << std::endl;
    }


    #if DEBUG
    // debug_info_stream << "FINAL ACTION: [" <<  pos_impl[2][1] << ", " <<  pos_impl[2][2] << ", " <<  pos_impl[3][1] << ", "<<  pos_impl[3][2] << "] " << "\n";
    // std::cout << "[DEBUG] LEG FINAL ACTION: [" <<  pos_impl[2][1] << ", " <<  pos_impl[2][2] << ", " <<  pos_impl[3][1] << ", "<<  pos_impl[3][2] << "] " << "\n";
    std::cout << "[DEBUG] ARM FINAL ACTION: [" <<  pos_impl[0][0] << ", " <<  pos_impl[0][1] << ", " <<  pos_impl[1][0] << ", "<<  pos_impl[1][1] << "] " << "\n";
    #endif

    if (_within_limits()){  // SECURITY CHECK:

        for(size_t leg(0); leg<2; ++leg){
            _Step(curr_iter, 9, 
            leg, initial_jpos[leg], pos_impl[leg]);
        }

        for(size_t leg(2); leg<4; ++leg){
            _Step(curr_iter, 0, 
            leg, initial_jpos[leg], pos_impl[leg]);
        }

    } else {

        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";

    }
    

    if (stopping && _has_stopped()){
        _phase = 6;
        _motion_start_iter = _state_iter + 1;
    }

    if(curr_iter > 11000 + additional_step || _done){

        #if DEBUG
        // debug_file.open("states_actions.txt", std::ios_base::app);

        // if (!debug_file)
        //   std::cout << "can't open output file" << std::endl;
        
        // debug_file << debug_info_stream.str();
        // debug_file.flush();

        #endif
        _phase = -5;

    }

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

    if (agent_ver == 2 || state_mode == StateMode::h_body_arm_p){
        state[0] = _stateEstimator->getResult().position[2];  // position z of torso
        state[1] = _stateEstimator->getResult().orientation[1];  // orientation x of torso
        state[2] = _stateEstimator->getResult().orientation[2];  // orientation y of torso
        state[3] = _stateEstimator->getResult().orientation[3];  // orientation z of torso
        state[4] = _stateEstimator->getResult().orientation[0];  // orientation w of torso
        state[5] = _stateEstimator->getResult().omegaBody[0];  // angular velocity r of torso   //TODO: THIS MAY NOT RIGHT
        state[6] = _stateEstimator->getResult().omegaBody[1];  // angular velocity p of torso
        state[7] = _stateEstimator->getResult().omegaBody[2];  // angular velocity y of torso
        state[8] = _legController->datas[0].q[0];  // joint position
        state[9] = _legController->datas[0].q[1] - front_hip_offset;  // joint position
        state[10] = _legController->datas[1].q[0];  // joint position
        state[11] = _legController->datas[1].q[1] - front_hip_offset;  // joint position
        state[12] = param_a;
        state[13] = param_b;
    } else if (state_mode == StateMode::body_arm_p){
        assert(state.size() == 12);
        _update_rpy();
        state[0] = rpy[0];
        state[1] = rpy[1];
        state[2] = rpy[2];
        state[3] = _stateEstimator->getResult().omegaWorld[0];  // angular velocity r of torso   //TODO: THIS MAY NOT RIGHT
        state[4] = _stateEstimator->getResult().omegaWorld[1];  // angular velocity p of torso
        state[5] = _stateEstimator->getResult().omegaWorld[2];  // angular velocity y of torso
        state[6] = _legController->datas[0].q[0];  // joint position
        state[7] = _legController->datas[0].q[1] - front_hip_offset;  // joint position
        state[8] = _legController->datas[1].q[0];  // joint position
        state[9] = _legController->datas[1].q[1] - front_hip_offset;  // joint position
        state[10] = param_a;
        state[11] = param_b;
        std::cout << " PARAM A: " << param_a << " PARAM B: " << param_b << std::endl;

    } else if (state_mode == StateMode::body_arm){
        assert(state.size() == 10);
        _update_rpy();
        state[0] = rpy[0];
        state[1] = rpy[1];
        state[2] = rpy[2];
        state[3] = _stateEstimator->getResult().omegaWorld[0];  // angular velocity r of torso   //TODO: THIS MAY NOT RIGHT
        state[4] = _stateEstimator->getResult().omegaWorld[1];  // angular velocity p of torso
        state[5] = _stateEstimator->getResult().omegaWorld[2];  // angular velocity y of torso
        state[6] = _legController->datas[0].q[0];  // joint position
        state[7] = _legController->datas[0].q[1] - front_hip_offset;  // joint position
        state[8] = _legController->datas[1].q[0];  // joint position
        state[9] = _legController->datas[1].q[1] - front_hip_offset;  // joint position

    } else if (state_mode == StateMode::body_arm_leg_full){
        assert(state.size() == 14);
        _update_rpy();
        state[0] = rpy[0];
        state[1] = rpy[1];
        state[2] = rpy[2];
        #if !DEBUG
        state[3] = _stateEstimator->getResult().omegaWorld[0];  // angular velocity r of torso   //TODO: THIS MAY NOT RIGHT
        state[4] = _stateEstimator->getResult().omegaWorld[1];  // angular velocity p of torso
        state[5] = _stateEstimator->getResult().omegaWorld[2];  // angular velocity y of torso
        state[6] = _legController->datas[0].q[0];  // joint position
        state[7] = _legController->datas[0].q[1] - front_hip_offset;  // joint position
        state[8] = _legController->datas[1].q[0];  // joint position
        state[9] = _legController->datas[1].q[1] - front_hip_offset;  // joint position
        state[10] = _legController->datas[2].q[1] - back_hip_offset;
        state[11] = _legController->datas[2].q[2];
        state[12] = _legController->datas[3].q[1] - back_hip_offset;
        state[13] = _legController->datas[3].q[2];
        #else
        // state[3] = bullet_omegaWorld[0];
        // state[4] = bullet_omegaWorld[1];
        // state[5] = bullet_omegaWorld[2];
        // state[6] = bullet_q[0];  // joint position
        // state[7] = bullet_q[1] - front_hip_offset;  // joint position
        // state[8] = bullet_q[3];  // joint position
        // state[9] = bullet_q[4] - front_hip_offset;  // joint position
        // state[10] = bullet_q[7] - back_hip_offset;
        // state[11] = bullet_q[8];
        // state[12] = bullet_q[10] - back_hip_offset;
        // state[13] = bullet_q[11];
        state[0] = state_from_bullet[0];
        state[1] = state_from_bullet[1];
        state[2] = state_from_bullet[2];
        state[3] = state_from_bullet[3];
        state[4] = state_from_bullet[4];
        state[5] = state_from_bullet[5];
        state[6] = state_from_bullet[6];  // joint position
        state[7] = state_from_bullet[7] - front_hip_offset;  // joint position
        state[8] = state_from_bullet[8];  // joint position
        state[9] = state_from_bullet[9] - front_hip_offset;  // joint position
        state[10] = state_from_bullet[10] - back_hip_offset;
        state[11] = state_from_bullet[11];
        state[12] = state_from_bullet[12] - back_hip_offset;
        state[13] = state_from_bullet[13];



        #endif
    } else if (state_mode == StateMode::body_arm_leg_full_filtered) {

        assert(state.size() == 14);
        for(int i=0; i<6; i++)
            state[i] = body_state_sum[i] / body_state_buffer.size();


    }

    if (!fast_error_update){

        d_error_buffer.push_back(_stateEstimator->getResult().omegaBody[1]);
        d_error_sum += _stateEstimator->getResult().omegaBody[1];
        if (d_error_buffer.size() > error_buffer_size){
            d_error_sum -= d_error_buffer[0];
            d_error_buffer.erase(d_error_buffer.begin());
        } 
        d_error = d_error_sum / d_error_buffer.size();

        p_error_buffer.push_back(state[1] - pitch_ref);
        p_error_sum += state[1] - pitch_ref;
        if (p_error_buffer.size() > error_buffer_size){
            p_error_sum -= p_error_buffer[0];
            p_error_buffer.erase(p_error_buffer.begin());
        } 
        p_error = p_error_sum / p_error_buffer.size();

        // std::cout << "P ERROR: " << p_error << "  D ERROR: " << d_error << std::endl;
    }


    if(fabs(p_error) > 0.65){
        _done = true;
        std::cout << "[DONE] I AM DEAD !!!" << std::endl;
    }
    
}


void Recovering::_update_action(){

    // std::shuffle(action_test.begin(), action_test.end(), std::default_random_engine(233));
    // action = action_test;
    std::cout << "[AGENT] STATE: ";
    for(int i=0; i<s_dim; i++){
        std::cout << state[i];
        if (i != s_dim-1)
            std::cout << ", ";
    }
    std::cout << std::endl;
    

    std::memcpy(TF_TensorData(input_tensor), state.data(), std::min(state.size() * sizeof(float), TF_TensorByteSize(input_tensor)));
    // printf("DATA WERE GIVEN TO THE TENSOR !!! \n");

    // Run the Session
    TF_SessionRun(sess, 
                  nullptr, 
                  &input_op, &input_tensor, 1,  // Input tensors, input tensor values, number of inputs.
                  &out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
                  nullptr, 0,
                  nullptr, 
                  status);
    
    // if(TF_GetCode(status) == TF_OK)
    //   printf("Session is OK\n");
    // else
    //   printf("%s",TF_Message(status));

    auto data = static_cast<float*>(TF_TensorData(output_tensor));
    // std::cout << "ACTION: " << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << std::endl;
    for (int i=0; i<a_dim; i++){
        if (progressing)
            action[i] = data[i] * progressing_agent_factor;
        else
            action[i] = data[i] * agent_factor;
    }

    std::cout << "[AGENT] ACTION: ";
    for(int i=0; i<a_dim; i++){
        std::cout << action[i];
        if (i != a_dim-1)
            std::cout << ", ";
    }
    std::cout << std::endl;

    

}

void Recovering::_update_state_buffer(){

    _update_rpy();
    #if !DEBUG
    std::vector<float> curr_state = {rpy[0], rpy[1], rpy[2], _stateEstimator->getResult().omegaWorld[0], _stateEstimator->getResult().omegaWorld[1], _stateEstimator->getResult().omegaWorld[2]};
    #else
    std::vector<float> curr_state = {rpy[0], rpy[1], rpy[2], bullet_omegaWorld[0], bullet_omegaWorld[1], bullet_omegaWorld[2]};
    #endif
    
    body_state_buffer.push_back(curr_state);
    assert(body_state_sum.size() == curr_state.size());
    for(unsigned int i=0; i < body_state_sum.size(); i++)
        body_state_sum[i] += curr_state[i];

     if (body_state_buffer.size() > state_buffer_size){
        for(unsigned int i=0; i < body_state_sum.size(); i++)
            body_state_sum[i] -= body_state_buffer[0][i];

        body_state_buffer.erase(body_state_buffer.begin());
    } 

}


void Recovering::_update_basic_leg_actions(const int & curr_iter){

    period_half = PI/std::max(param_b, 0.00001f);
    // sub_t = curr_iter % (2*period_half);
    sub_t = fmod(curr_iter, (2*period_half));

    // std::cout << "[DEBUG] half period: " << period_half << "  sub_t: " << sub_t << std::endl;
    if (sub_t >= period_half && sub_t - 1 < period_half){
        param_a = std::accumulate(param_a_buffer.begin(), param_a_buffer.end(), 0.0) / (float)param_a_buffer.size();
        param_b = std::accumulate(param_b_buffer.begin(), param_b_buffer.end(), 0.0) / (float)param_b_buffer.size();
        param_a_buffer.clear();
        param_b_buffer.clear();
        delta_x = delta_x_buffered;
    }

    if (gait == Gait::line){
        float theta2_r_delta =  - param_a*(sin(param_b*curr_iter-PI/2)/2+0.5);
        float theta2_l_delta =  - param_a*(-sin(param_b*curr_iter-PI/2)/2+0.5);

        if (param_b*curr_iter < PI) {
            // std::cout << "[DEBUG] pos_impl: theta1 - " << theta2_r_delta << " , theta2 + " <<  theta2_r_delta << std::endl;
            // std::cout << "[DEBUG] pos_impl: " << theta1 - theta2_r_delta << " , " <<  theta2 + theta2_r_delta << std::endl;
            // std::cout << "[DEBUG] pos_impl[2][2] SO FAR: " << pos_impl[2][2] << std::endl;
            pos_impl[2] << 0, theta1 - theta2_r_delta, theta2 + theta2_r_delta;
            // pos_impl[2] << 0, _theta1_hat(theta2 + theta2_r_delta), theta2 + theta2_r_delta;
            pos_impl[3] << 0, theta1, theta2;
        } else {
            // pos_impl[2] << 0, _theta1_hat(theta2 - param_a*(sin(param_b*curr_iter-PI/2)/2+0.5)), theta2 - param_a*(sin(param_b*curr_iter-PI/2)/2+0.5);
            pos_impl[2] << 0, theta1 - theta2_r_delta, theta2 + theta2_r_delta;
            // pos_impl[2] << 0, _theta1_hat(theta2 + theta2_r_delta), theta2 + theta2_r_delta;
            pos_impl[3] << 0, theta1 - theta2_l_delta, theta2 + theta2_l_delta;
        }
        // pos_impl[2] << 0, _theta1_hat(theta2 - param_a*sin(param_b*curr_iter)), theta2 - param_a*sin(param_b*curr_iter);
        // pos_impl[3] << 0, theta1 - param_a*sin(param_b*curr_iter), theta2 + param_a*sin(param_b*curr_iter);
        // pos_impl[3] << 0, _theta1_hat(theta2 + param_a*sin(param_b*curr_iter)), theta2 + param_a*sin(param_b*curr_iter);

    } else if (gait == Gait::sine){
        if (sub_t < period_half){
            x_togo_r = x_original + (delta_x/period_half)*sub_t;
            y_togo_r = y_original + param_a * sin(PI/period_half*sub_t);

            if (sub_t != curr_iter){
                x_togo_l = x_original + delta_x - sub_t * delta_x/period_half;
                y_togo_l = y_original;
            } else {
                x_togo_l = x_original;
                y_togo_l = y_original;
            }

        } else {
            x_togo_r = x_original + delta_x - (sub_t-period_half) * delta_x/period_half;
            y_togo_r = y_original;
            x_togo_l = x_original + (delta_x/period_half)*(sub_t-period_half);
            y_togo_l = y_original + param_a * sin(PI/period_half*(sub_t-period_half));
        }    
    } else if (gait == Gait::rose){

        // assert(delta_x != 0);

        a_rose = std::max(delta_x, 0.00001f);
        k_rose = 4*param_a/a_rose;

        if (sub_t < period_half) {
            th = (PI/4)/period_half*(period_half-sub_t);
            x_togo_r = x_original + a_rose * cos(2*th) * cos(th);
            y_togo_r = y_original + k_rose * a_rose * cos(2*th) * sin(th);
            // std::cout << "[DEBUG] y_togo_r = " << y_original << " + " << k_rose  << " * " <<  a_rose << " * " << cos(2*th) << " * " << sin(th) << " = " << y_togo_r << std::endl;

            if (sub_t != curr_iter){
                x_togo_l = x_original + delta_x - sub_t * delta_x/period_half;
                y_togo_l = y_original;
            } else {
                x_togo_l = x_original;
                y_togo_l = y_original;
            }
        } else {
            th = (PI/4)/period_half*(period_half-(sub_t-period_half));
            x_togo_r = x_original + delta_x - (sub_t-period_half) * delta_x/period_half;
            y_togo_r = y_original;
            x_togo_l = x_original + a_rose * cos(2*th) * cos(th);
            y_togo_l = y_original + k_rose * a_rose * cos(2*th) * sin(th);
        }
    } else if (gait == Gait::triangle){

        if (curr_iter == 0){
            _FK(theta1, theta2, x_0, y_0);
            _FK(theta1-(-param_a), theta2+(-param_a), x_1, y_1);
            std::cout <<"[DEBUG] [TRIANGLE] CALCULATE FK OF (" << theta1-(-param_a)<< ", " << theta2+(-param_a) << ")  RESULTING Y: " << y_1 << std::endl;
            
        } else if (progressing) {
            _FK(theta1-(-param_a), theta2+(-param_a), x_1, y_1);
            // std::cout <<"[DEBUG] [TRIANGLE] CALCULATE FK OF (" << theta1-(-param_a)<< ", " << theta2+(-param_a) << ")  RESULTING Y: " << y_1 << std::endl;
        }

        x_2 = x_0 + delta_x;
        y_2 = y_0;
        
        if (sub_t < period_half){
            if (param_b*sub_t < PI/2){  // == sub_t < period_half/2:
                // print(f"{param_b*sub_t} --> {PI/2}")
                x_togo_r = x_0 + (sin(2*param_b*sub_t-PI/2)+1)/2*(x_1-x_0);
                y_togo_r = y_0 + (sin(2*param_b*sub_t-PI/2)+1)/2*(y_1-y_0);
            } else {
                // print(f"{param_b*sub_t} --> {PI}")
                x_togo_r = x_1 + (-sin(2*param_b*sub_t-PI/2)/2+0.5)*(x_2-x_1);
                y_togo_r = y_1 + (-sin(2*param_b*sub_t-PI/2)/2+0.5)*(y_2-y_1);
            }

            if (sub_t != curr_iter){
                x_togo_l = x_original + delta_x - sub_t * delta_x/period_half;
                y_togo_l = y_original;
            } else {
                x_togo_l = x_original;
                y_togo_l = y_original;
            }
        } else {
            x_togo_r = x_original + delta_x - (sub_t-period_half) * delta_x/period_half;
            y_togo_r = y_original;

            if (param_b*(sub_t-period_half) < PI/2){
                x_togo_l = x_0 + (sin(2*param_b*(sub_t-period_half)-PI/2)/2+0.5)*(x_1-x_0);
                y_togo_l = y_0 + (sin(2*param_b*(sub_t-period_half)-PI/2)/2+0.5)*(y_1-y_0);
            } else {
                x_togo_l = x_1 + (-sin(2*param_b*(sub_t-period_half)-PI/2)/2+0.5)*(x_2-x_1);
                y_togo_l = y_1 + (-sin(2*param_b*(sub_t-period_half)-PI/2)/2+0.5)*(y_2-y_1);
            }
        }

    }

    // std::cout << "AND  (" << x_togo_l << ", " << y_togo_l << ")" << std::endl;
    if (gait != Gait::line){
        _IK(x_togo_r, y_togo_r, pos_impl[2][1], pos_impl[2][2]);
        _IK(x_togo_l, y_togo_l, pos_impl[3][1], pos_impl[3][2]);
    }
    

    // std::cout << "SEND TO IK:  (" << x_togo_r << ", " << y_togo_r << ")   GOT:  (" << pos_impl[2][1] << ", " << pos_impl[2][2] << ") " << std::endl;

    #if DEBUG
    // debug_info_stream << "BASIC ACTION: [" <<  pos_impl[2][1] << ", " <<  pos_impl[2][2] << ", " <<  pos_impl[3][1] << ", "<<  pos_impl[3][2] << "] " << "\n";
    #endif

    
}

void Recovering::_process_leg_offsets(const int & curr_iter){


    float a(0.f);
    float b(1.f);
    Vec4<float> inter_leg_offsets;

    b = (float)(curr_iter%num_sub_steps)/(float)(num_sub_steps-1);
    a = 1.f - b;
    // compute setpoints
    inter_leg_offsets = a * leg_offsets_old + b * leg_offsets;

    pos_impl[2][1] -= inter_leg_offsets[0];
    pos_impl[2][2] += inter_leg_offsets[1];
    pos_impl[3][1] -= inter_leg_offsets[2];
    pos_impl[3][2] += inter_leg_offsets[3];

    // pos_impl[2][1] = (1-agent_factor)*pos_impl[2][1] + agent_factor*(pos_impl[2][1] - inter_leg_offsets[0]);
    // pos_impl[2][2] = (1-agent_factor)*pos_impl[2][2] + agent_factor*(pos_impl[2][2] + inter_leg_offsets[1]);
    // pos_impl[3][1] = (1-agent_factor)*pos_impl[3][1] + agent_factor*(pos_impl[3][1] - inter_leg_offsets[2]);
    // pos_impl[3][2] = (1-agent_factor)*pos_impl[3][2] + agent_factor*(pos_impl[3][2] + inter_leg_offsets[3]);

    #if DEBUG
    // debug_info_stream << "OFFSET: [" <<  inter_leg_offsets[0] << ", " <<  inter_leg_offsets[1] << ", " <<  inter_leg_offsets[2] << ", "<<  inter_leg_offsets[3] << "] " << "\n";
    #endif

}

void Recovering::_update_leg_offsets(){

    if (leg_action_mode == LegActionMode::parallel_offset) {
        leg_offsets[0] += action[4] * leg_offset_multiplier;
        leg_offsets[1] += action[4] * leg_offset_multiplier;
        leg_offsets[2] += action[5] * leg_offset_multiplier;
        leg_offsets[3] += action[5] * leg_offset_multiplier;
    } else if (leg_action_mode == LegActionMode::hips_offset) {
        leg_offsets[0] += action[4] * leg_offset_multiplier;
        leg_offsets[1] = 0;
        leg_offsets[2] += action[5] * leg_offset_multiplier;
        leg_offsets[3] = 0;
    } else if (leg_action_mode == LegActionMode::knees_offset) {
        leg_offsets[0] = 0;
        leg_offsets[1] += action[4] * leg_offset_multiplier;
        leg_offsets[2] = 0;
        leg_offsets[3] += action[5] * leg_offset_multiplier;
    } else if (leg_action_mode == LegActionMode::hips_knees_offset) {
        leg_offsets[0] += action[4] * leg_offset_multiplier;
        leg_offsets[1] += action[5] * leg_offset_multiplier;
        leg_offsets[2] += action[6] * leg_offset_multiplier;
        leg_offsets[3] += action[7] * leg_offset_multiplier;
    } else if (leg_action_mode == LegActionMode::none) {
        assert(leg_offsets[0] == 0 && leg_offsets[1] == 0 && leg_offsets[2] == 0);
    }

    for (int i=0; i<4; i++){
        leg_offsets[i] = _Box(leg_offsets[i], leg_offset_range[0], leg_offset_range[1]);
    }
}

void Recovering::_SettleDown(const int & curr_iter){

    for(size_t i(0); i < 4; ++i)
        initial_jpos[i] = this->_legController->datas[i].q;
    for(size_t i(0); i < 2; ++i)
        initial_jpos[i] = initial_jpos[i] - front_offset;
    for(size_t i(2); i < 4; ++i)
        initial_jpos[i] = initial_jpos[i] - back_offset;

    for(size_t leg(0); leg<4; ++leg)
        prepare_jpos[leg] = initial_jpos[leg];

    if (fabs(0 - prepare_jpos[0][0]) > 0.02)
        prepare_jpos[0][0] = initial_jpos[0][0] + (0 - initial_jpos[0][0])/fabs(0 - initial_jpos[0][0])*0.02;
    if (fabs(-PI/2 - prepare_jpos[0][1]) > 0.02)
        prepare_jpos[0][1] = initial_jpos[0][1] + (-PI/2 - initial_jpos[0][1])/fabs(-PI/2 - initial_jpos[0][1])*0.02;
    prepare_jpos[0][2] = _theta2_prime_hat(prepare_jpos[0][1]);
    
    if (fabs(0 - prepare_jpos[1][0]) > 0.02)
        prepare_jpos[1][0] = initial_jpos[1][0] + (0 - initial_jpos[1][0])/fabs(0 - initial_jpos[1][0])*0.02;
    if (fabs(-PI/2 - prepare_jpos[1][1]) > 0.02)
        prepare_jpos[1][1] = initial_jpos[1][1] + (-PI/2 - initial_jpos[1][1])/fabs(-PI/2 - initial_jpos[1][1])*0.02;
    prepare_jpos[1][2] = _theta2_prime_hat(prepare_jpos[1][1]);

    prepare_jpos[2] << -0.0f, theta1, theta2;
    prepare_jpos[3] << 0.0f, theta1, theta2;

    if (curr_iter % 50 == 0)
        std::cout << "SETTLING DOWN: " << prepare_jpos[0][0] << ", " << prepare_jpos[0][1] << ", " << prepare_jpos[0][2] << ", " << 
                                          prepare_jpos[1][0] << ", " << prepare_jpos[1][1] << ", " << prepare_jpos[1][2] << std::endl;



    for(size_t leg(0); leg<4; ++leg){
        _Step(curr_iter, 0, leg, initial_jpos[leg], prepare_jpos[leg]);
    }

    if(curr_iter >= 1200){  

        _phase = 7;
        _motion_start_iter = _state_iter + 1;
    }

    

}


void Recovering::_InverseRearLegsActions(const int & curr_iter){

    if (curr_iter==0){
      for(size_t i(0); i < 4; ++i) {
          initial_jpos[i] = this->_legController->datas[i].q;
      }
      for(size_t i(0); i < 2; ++i)
        initial_jpos[i] = initial_jpos[i] - front_offset;
      for(size_t i(2); i < 4; ++i)
        initial_jpos[i] = initial_jpos[i] - back_offset;

      stopping = false;

    //   theta1 = pre2_back_hip;  
    //   theta2 = pre2_back_knee;  // -1
    }

    if (theta2 > pre2_back_knee)
        theta2 -= 0.0005;
    if (k_ > 0.516)
        k_ -= 0.0003;
    // std::cout << "K: " << k_ << std::endl;
    // std::cout << "CALCULATING THE ACOS OF " << (l_/2 + b_*cos(PI - (-theta2) - gamma_) -c_*cos(gamma_)) <<  "/" << k_ << "/" << a_ << " = "<< acos((l_/2 + b_*cos(PI - (-theta2) - gamma_) -c_*cos(gamma_)) / k_ / a_)  << std::endl;

    theta1 = _theta1_hat(theta2); // -PI/2 + (-theta2) + gamma_ - acos((l_/2 + b_*cos(PI - (-theta2) - gamma_) -c_*cos(gamma_)) / k_ / a_);
   //  std::cout << "THETA1: " << theta1 << "    THETA2: " << theta2 << std::endl;
    // Eigen::Quaterniond _orientation(_stateEstimator->getResult().orientation[0], _stateEstimator->getResult().orientation[1], _stateEstimator->getResult().orientation[2], _stateEstimator->getResult().orientation[3]);
    _update_rpy();
    pitch = rpy[1];

    if (curr_iter % 50 == 0){
        std::cout << " THE PITCH (ALPHA) IS " << pitch << std::endl;
        std::cout << "   --  THE ORIENTATION IS " << _stateEstimator->getResult().orientation[0] << ", "<< _stateEstimator->getResult().orientation[1]<< ", "<< _stateEstimator->getResult().orientation[2]<< ", "<< _stateEstimator->getResult().orientation[3] << std::endl;
        std::cout << " THE POSITION OF THE BACK HIP (THETA1) IS " << _legController->datas[2].q[1] << ", " << _legController->datas[3].q[1]  << std::endl;
        std::cout << " THE POSITION OF THE BACK KNEE (THETA2) IS " << _legController->datas[2].q[2] << ", " << _legController->datas[3].q[2]  << std::endl;
    }
    // std::cout << "TH1: " << bullet_q[10] - back_hip_offset << ", TH2: " << bullet_q[11] << " ( k = " << k_ << ")  -->  PITCH: " << pitch << "    HEIGHT: " << _stateEstimator->getResult().position[2] << std::endl;
    // std::cout << "JPOS: [" << bullet_q[0] << ", " << bullet_q[1] << ", " << bullet_q[2] << ", " << bullet_q[3] << ", " << bullet_q[4] << ", "<< bullet_q[5] << ", "<< bullet_q[6] << ", "<< bullet_q[7] << ", "<< bullet_q[8] << ", "<< bullet_q[9] << ", "<< bullet_q[10] << ", " << bullet_q[11] << "]" << std::endl;
    prepare_jpos[2] << -0.0f, theta1 + back_hip_offset, theta2;
    prepare_jpos[3] << 0.0f, theta1 + back_hip_offset, theta2;

    for(size_t i(0); i<4; ++i){
      this->jointPDControl(i, prepare_jpos[i], zero_vec3);
    }

    prepare_jpos[0] << -0.0f, pre2_front_hip, pre2_front_knee;
    prepare_jpos[1] << 0.0f, pre2_front_hip, pre2_front_knee;

    for(size_t leg(0); leg<2; ++leg){
        _Step(curr_iter, 1000, 
        leg, initial_jpos[leg], prepare_jpos[leg]);
    }

    if(curr_iter >= 2000){
        _phase = 8;
        _motion_start_iter = _state_iter + 1;
    }


}


void Recovering::_InversePrepare_2(const int & curr_iter){

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i) {
            initial_jpos[i] = this->_legController->datas[i].q;
        }
    }

    for(size_t i(0); i<4; ++i){
      _SetJPosInterPts(curr_iter, 1200, i, 
      initial_jpos[i], stand_jpos[i]);
    }

    if(curr_iter >= 1200 + 100){
        _phase = 9;
        _motion_start_iter = _state_iter + 1;
    }
    
}


void Recovering::_InverseJustStandUp(const int & curr_iter){

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i) {
            initial_jpos[i] = this->_legController->datas[i].q;
        }

        prepare_jpos[0] << 0.0f, -0.75f, 1.5f;
        prepare_jpos[1] << 0.0f, -0.75f, 1.5f;
        prepare_jpos[2] << 0.0f, -0.62f, 1.3f;
        prepare_jpos[3] << 0.0f, -0.62f, 1.3f;
    } 

    for(size_t leg(0); leg<4; ++leg){
        _SetJPosInterPts(curr_iter, standup_ramp_iter, 
                         leg, initial_jpos[leg], prepare_jpos[leg]);
    }       

    if(curr_iter >= standup_ramp_iter+100){
        _phase = 11;
        _motion_start_iter = _state_iter+1;
    } 

}



void Recovering::_PushUp(const int & curr_iter){


    if (curr_iter == 0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

        prepare_jpos[0] << 0.0f, -1.0f, 2.0f;
        prepare_jpos[1] << 0.0f, -1.0f, 2.0f;
        prepare_jpos[2] << 0.0f, -PI/2, 0.0f;
        prepare_jpos[3] << 0.0f, -PI/2, 0.0f;

    } 
    if (curr_iter < 1000){
        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, 999, 
            leg, initial_jpos[leg], prepare_jpos[leg]);
        }

    } else {
        if ((curr_iter-1000) % 1000 == 0){
            for(size_t i(0); i < 4; ++i)
                initial_jpos[i] = prepare_jpos[i];
            if (prepare_jpos[0][2] == 2){
                prepare_jpos[0] << 0.0f, 0.0f, 0.0f;
                prepare_jpos[1] << 0.0f, 0.0f, 0.0f;

            } else {
                prepare_jpos[0] << 0.0f, -1.0f, 2.0f;
                prepare_jpos[1] << 0.0f, -1.0f, 2.0f;
            }
            
        }

        for(size_t leg(0); leg<4; ++leg){
            _Step((curr_iter-1000)%1000, 999, 
            leg, initial_jpos[leg], prepare_jpos[leg]);
        }
    }

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
        _phase = 1;
        _motion_start_iter = _state_iter+1;
    } 

}


void Recovering::_ClimbPre(const int & curr_iter){

    // std::cout << "[DEBUG] TRANFERING TO STANDING UP... (CURR_ITER " << curr_iter << ")" << std::endl;
    

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

        std::cout << "================== CALCULATING IK ====================" << std::endl;
        // INITIAL POSE IS NOT RIGHT IN SIMULATION 

        pre_climb_th1 = theta1 + 0.1;

        _arms_IK(0.1, 0, pre_climb_th1, theta2, table_global, climb_th1_p, climb_th2_p);


        std::cout << "    [INPUT] " << std::endl;
        std::cout << "    X: " << 0.1 << ", AB: " << 0 << ", TH1: " << pre_climb_th1 << ", TH2: " << theta2 << ", TABLE: " << table_global << std::endl;
        std::cout << "    [OUTPUT] " << std::endl;
        std::cout << "    TH1P: " << climb_th1_p << ", TH2P: " << climb_th2_p << std::endl;

        if (gamma_ - climb_th1 - theta2 + climb_th1_p + climb_th2_p - PI > 0)
            std::cout << "    FUCK!  THE ROBOT WILL USE ITS ELBOW TO CLIMB..." << std::endl;
        else
            std::cout << "    GOOD!  I GUESS THE ROBOT POSE WILL BE NORMAL..." << std::endl;
        
        std::cout << "======================================================" << std::endl;

    }

    // pos_impl[0] << -0.0f, -0.1f, _theta2_prime_hat(-0.1f);
    // pos_impl[1] << 0.0f, -0.1f, _theta2_prime_hat(-0.1f);
    pos_impl[0] << 0.0f, climb_th1_p, climb_th2_p;
    pos_impl[1] << 0.0f, climb_th1_p, climb_th2_p;
    pos_impl[2] << 0.0f, pre_climb_th1, theta2;
    pos_impl[3] << 0.0f, pre_climb_th1, theta2;

    if (_within_limits()){
        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 

    } else {
        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
    }

    if(curr_iter >= 2500+2500){
        _phase = 15; // phase 14 is skipped
        _motion_start_iter = _state_iter+1;
    } 

}


void Recovering::_ClimbPre1(const int & curr_iter){

    // THIS STEP IS SKIPPED !!!

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;
        _arms_IK(0.1, climb_th1_p, climb_th2_p);

        std::cout << "==================INITIAL POSE====================" << std::endl;
        // INITIAL POSE IS NOT RIGHT IN SIMULATION 
        std::cout << "initial_jpos[2][1]: " << initial_jpos[2][1] << "  theta1: " << theta1 << std::endl;
        std::cout << "initial_jpos[2][2]: " << initial_jpos[2][2] << "  theta2: " << theta2 << std::endl;
        std::cout << "initial_jpos[0][1]: " << initial_jpos[0][1] << "  PI/2: " << PI/2 << std::endl;

        std::cout << "=======================================================" << std::endl;
    }

    // pos_impl[0] << -0.0f, -0.1f, _theta2_prime_hat(-0.1f);
    // pos_impl[1] << 0.0f, -0.1f, _theta2_prime_hat(-0.1f);
    pos_impl[0] << -arm_ab, climb_th1_p, climb_th2_p;
    pos_impl[1] << arm_ab, climb_th1_p, climb_th2_p;
    pos_impl[2] << 0.0f, initial_jpos[2][1], initial_jpos[2][2];
    pos_impl[3] << 0.0f, initial_jpos[3][1], initial_jpos[3][2];

    if (_within_limits()){
        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 

    } else {
        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
    }

    if(curr_iter >= 2500+2500){
        _phase = 15;
        _motion_start_iter = _state_iter+1;
    } 

}

// void Recovering::_ClimbBAK(const int & curr_iter){

//     // std::cout << "[DEBUG] TRANFERING TO STANDING UP... (CURR_ITER " << curr_iter << ")" << std::endl;

//     if (curr_iter==0){
//         for(size_t i(0); i < 4; ++i)
//           initial_jpos[i] = this->_legController->datas[i].q;
//         for(size_t i(0); i < 2; ++i)
//             initial_jpos[i] = initial_jpos[i] - front_offset;
//         for(size_t i(2); i < 4; ++i)
//             initial_jpos[i] = initial_jpos[i] - back_offset;

//         climb_th1 = (initial_jpos[2][1] + initial_jpos[3][1]) / 2;
        

//         std::cout << "==================CLIMB====================" << std::endl;
//         // INITIAL POSE IS NOT RIGHT IN SIMULATION 
//         std::cout << "initial_jpos[2][1]: " << initial_jpos[2][1] << "  theta1: " << theta1 << std::endl;
//         std::cout << "initial_jpos[2][2]: " << initial_jpos[2][2] << "  theta2: " << theta2 << std::endl;
//         std::cout << "initial_jpos[0][1]: " << initial_jpos[0][1] << "  PI/2: " << PI/2 << std::endl;

//         std::cout << "============================================" << std::endl;
//     }


//     if (climb_x < climb_x_set) 
//         // initial climb_x: 0.1
//         climb_x += 0.0001;

//     if (arm_ab < arm_ab_set) 
//         arm_ab += 0.0005;

//     float foracos;

//     for (int i=0; i<80; i++){
//         foracos = (pow(climb_x, 2) + pow(_head_height(), 2) - pow(b_, 2) - pow(c_, 2)) / (2*b_*c_);
//         if (!(foracos < 0.98 && foracos > -0.98))
//             climb_th1 += 0.0006;
//         else
//             break;
//     }

//     if (!(foracos < 0.98 && foracos > -0.98))
//         std::cout << "SHIT... THE IK WILL NOT BE SOLVED FOR X = " << climb_x << " AND AB = " << arm_ab << ",  foracos: " << foracos << std::endl;

//     if (gamma_ - climb_th1 - initial_jpos[3][2] + climb_th1_p + climb_th2_p - PI > 0)
//         std::cout << "FUCK!  THE ROBOT IS USING ITS ELBOW TO CLIMB..." << std::endl;

//     _arms_IK(climb_x, climb_th1_p, climb_th2_p);

//     std::cout << "[DEBUG] TH1_P: " << climb_th1_p <<  " TH2_P: " << climb_th2_p  <<  " TH1: " << climb_th1 << " TH2: " << theta2 << std::endl; 

//     // pos_impl[0] << -0.0f, -0.1f, _theta2_prime_hat(-0.1f);
//     // pos_impl[1] << 0.0f, -0.1f, _theta2_prime_hat(-0.1f);
//     pos_impl[0] << -arm_ab, climb_th1_p, climb_th2_p;
//     pos_impl[1] << arm_ab, climb_th1_p, climb_th2_p;
//     pos_impl[2] << 0.0f, climb_th1, initial_jpos[2][2];
//     pos_impl[3] << 0.0f, climb_th1, initial_jpos[3][2];



//     if (_within_limits()){
//         for(size_t leg(0); leg<4; ++leg){
//             _Step(curr_iter, 0, 
//             leg, initial_jpos[leg], pos_impl[leg]);
//         } 

//     } else {
//         _phase = -911;
//         std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
//     }

    



//     if(curr_iter >= 3500){
//         _phase = 16;
//         _motion_start_iter = _state_iter+1;
//     } 

// }

void Recovering::_Climb(const int & curr_iter){

    // std::cout << "[DEBUG] TRANFERING TO STANDING UP... (CURR_ITER " << curr_iter << ")" << std::endl;

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

        climb_th1 = 0.1;
        

        std::cout << "================== CALCULATING IK ====================" << std::endl;
        // INITIAL POSE IS NOT RIGHT IN SIMULATION 

        _arms_IK(climb_x_set, arm_ab_set, climb_th1, theta2, table_global, climb_th1_p, climb_th2_p);


        std::cout << "    [INPUT] " << std::endl;
        std::cout << "    X: " << climb_x_set << ", AB: " << arm_ab_set << ", TH1: " << climb_th1 << ", TH2: " << theta2 << ", TABLE: " << table_global << std::endl;
        std::cout << "    [OUTPUT] " << std::endl;
        std::cout << "    TH1P: " << climb_th1_p << ", TH2P: " << climb_th2_p << std::endl;

        if (gamma_ - climb_th1 - theta2 + climb_th1_p + climb_th2_p - PI > 0)
            std::cout << "    FUCK!  THE ROBOT WILL USE ITS ELBOW TO CLIMB..." << std::endl;
        else
            std::cout << "    GOOD!  I GUESS THE ROBOT POSE WILL BE NORMAL..." << std::endl;
        
        std::cout << "================== CALCULATING AB ====================" << std::endl;

        float delta_h = ab_y*tan(leg_ab_support);
        float delta_h_l = delta_h;  //long one: keep the same as that in 0 roll case, since long the long one is not easy
        float delta_h_r = delta_h + sin(climb_roll)*torso_y;  // this is the value of length to be shortened
        head_height_curr = _head_height(climb_th1, theta2, arm_ab_set, table_global)*cos(arm_ab_set);
        butt_height_curr = _butt_height(theta2);
        delta_th_long =  _delta_th(butt_height_curr, delta_h_l);
        delta_th_shorten =  _delta_th(butt_height_curr, -delta_h_r);

        // IK version of lowering the right arm
        _arms_IK(climb_x_set, arm_ab_set + arm_ab_delta, climb_th1, theta2, table_global + sin(climb_roll)*torso_y, climb_th1_p_r, climb_th2_p_r);


        // delta_th =  _delta_th(butt_height_curr, delta_h);
        std::cout << "    THEORETICAL HEAD HEIGHT: " << head_height_curr << std::endl;
        std::cout << "    THEORETICAL BUTT HEIGHT: " << butt_height_curr << std::endl;
        std::cout << "    [INPUT] " << std::endl;
        std::cout << "    LEGS AB: " << leg_ab_support << ", DELTA H(L): " << delta_h_l << ", DELTA H(R): " << delta_h_r << std::endl;
        std::cout << "    [OUTPUT] " << std::endl;
        std::cout  << "   L-DELTA TH: " << delta_th_long << ", L-TH2: " << climb_th1 - delta_th_long  << ", L-TH1: " << theta2 + delta_th_long << std::endl;
        std::cout  << "   R-DELTA TH: " << delta_th_shorten << ", R-TH2: " << climb_th1 + delta_th_shorten  << ", R-TH1: " << theta2 - delta_th_shorten << std::endl;
        // LEFT LEG BECOMES HIGHER, RIGHT LEG BECOMES LOWER


        std::cout << "======================================================" << std::endl;

        // head_height_curr = (_head_height(climb_th1, theta2, arm_ab_set, table_global) + table_global)*cos(arm_ab_set) - table_global;
    }


    pos_impl[0] << -arm_ab_set - arm_ab_delta, climb_th1_p_r, climb_th2_p_r;
    pos_impl[1] << arm_ab_set, climb_th1_p, climb_th2_p;
    // pos_impl[2] << 0.0f, climb_th1, theta2;
    // pos_impl[3] << 0.0f, climb_th1, theta2;
    pos_impl[2] << leg_ab_support, climb_th1 + delta_th_shorten + 0.1, theta2 - delta_th_shorten;  // R: shorten
    pos_impl[3] << leg_ab_support, climb_th1 + delta_th_long + 0.1, theta2 - delta_th_long;  // L: long

    if (_within_limits()){
        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 

    } else {
        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
    }


    if(curr_iter >= 2500 + 2500){
        _phase = 17; // _Climb2 (test)
        _motion_start_iter = _state_iter+1;
    } 

}

// void Recovering::_Climb1(const int & curr_iter){

//     // lift the left hind leg, rotate the right hind leg inside, stretch the right front leg further

//     if (curr_iter==0){
//         for(size_t i(0); i < 4; ++i)
//           initial_jpos[i] = this->_legController->datas[i].q;
//         for(size_t i(0); i < 2; ++i)
//             initial_jpos[i] = initial_jpos[i] - front_offset;
//         for(size_t i(2); i < 4; ++i)
//             initial_jpos[i] = initial_jpos[i] - back_offset;

//     }


//     // pos_impl[0] << -arm_ab_set - 0.0, initial_jpos[0][1], initial_jpos[0][2];
//     // pos_impl[1] << arm_ab_set, initial_jpos[1][1], initial_jpos[1][2];
//     // pos_impl[2] << leg_ab_support, initial_jpos[2][1], initial_jpos[2][2];
//     // pos_impl[3] << leg_ab_support, initial_jpos[2][1] + 0.5, initial_jpos[2][2] - 0.5;

//     pos_impl[0] << -arm_ab_set - arm_ab_delta, climb_th1_p, climb_th2_p;
//     pos_impl[1] << arm_ab_set, climb_th1_p, climb_th2_p;
//     pos_impl[2] << leg_ab_support, climb_th1 + delta_th_shorten, theta2 - delta_th_shorten;  // R: shorten
//     pos_impl[3] << leg_ab_support, climb_th1 + delta_th_long + 1, theta2 - delta_th_long - 1;  // L: long

//     if (_within_limits()){
//         for(size_t leg(0); leg<4; ++leg){
//             _Step(curr_iter, 2000, 
//             leg, initial_jpos[leg], pos_impl[leg]);
//         } 

//     } else {
//         _phase = -911;
//         std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
//     }


//     if(curr_iter >= 2500 + 999999){
//         _phase = 17;
//         _motion_start_iter = _state_iter+1;
//     } 

// }

void Recovering::_Climb2(const int & curr_iter){

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

    }


    // pos_impl[0] << -arm_ab_set - 0.0, initial_jpos[0][1], initial_jpos[0][2];
    // pos_impl[1] << arm_ab_set, initial_jpos[1][1], initial_jpos[1][2];
    // pos_impl[2] << leg_ab_support, initial_jpos[2][1], initial_jpos[2][2];
    // pos_impl[3] << 1.3f, initial_jpos[2][1] + 0.5, initial_jpos[2][2] - 0.5;

    pos_impl[0] << -arm_ab_set - arm_ab_delta, climb_th1_p_r, climb_th2_p_r;
    pos_impl[1] << arm_ab_set, climb_th1_p, climb_th2_p;
    pos_impl[2] << leg_ab_support, climb_th1 + delta_th_shorten + 0.1, theta2 - delta_th_shorten;  // R: shorten
    pos_impl[3] << leg_ab_lift, climb_th1 + delta_th_long + 0.1 + 0.5 , theta2 - delta_th_long - 0.5;  // L: long

    if (_within_limits()){
        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 

    } else {
        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
    }


    if(curr_iter >= 2500){
        _phase = 18;
        _motion_start_iter = _state_iter+1;
    } 

}

void Recovering::_Climb3(const int & curr_iter){

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

        std::cout << "================== CALCULATING IK (HIND LEGS) ====================" << std::endl;

        float foot_up_x = 0.28;

        // DEBUG:
        leg_ab_touch = 1.0;
        ////////
        _update_rpy();
        pitch = rpy[1];
        _legs_IK(foot_up_x, leg_ab_touch, head_height_curr, pitch, climb_up_th1, climb_up_th2);


        std::cout << "    [INPUT] " << std::endl;
        std::cout << "    X: " << foot_up_x << ", AB: " << leg_ab_touch << ", HEAD HEIGHT: " << head_height_curr << ", PITCH: "<< pitch << std::endl;
        std::cout << "    [OUTPUT] " << std::endl;
        std::cout << "    TH1: " << climb_up_th1 << ", TH2: " << climb_up_th2 << std::endl;

        if (climb_up_th2 > 2.0)
            std::cout << "    OH... THE STICK WOULD BE SQUASHED!" << std::endl;

        // if (gamma_ - climb_th1 - theta2 + climb_th1_p + climb_th2_p - PI > 0)
        //     std::cout << "    FUCK!  THE ROBOT WILL USE ITS ELBOW TO CLIMB..." << std::endl;
        // else
        //     std::cout << "    GOOD!  I GUESS THE ROBOT POSE WILL BE NORMAL..." << std::endl;
        
        std::cout << "==================================================================" << std::endl;
        
    }


    // pos_impl[0] << -arm_ab_set - 0.0, initial_jpos[0][1], initial_jpos[0][2];
    // pos_impl[1] << arm_ab_set, initial_jpos[1][1], initial_jpos[1][2];
    pos_impl[0] << -arm_ab_set - arm_ab_delta, climb_th1_p_r, climb_th2_p_r;
    pos_impl[1] << arm_ab_set, climb_th1_p, climb_th2_p;
    pos_impl[2] << leg_ab_support, initial_jpos[2][1], initial_jpos[2][2];
    // pos_impl[3] << leg_ab_lift, initial_jpos[2][1] + 0.5 + 0.8, initial_jpos[2][2] - 0.5;
    // pos_impl[3] << leg_ab_lift, -0.1f, 2.0f;
    pos_impl[3] << leg_ab_lift, climb_up_th1, climb_up_th2;

    if (_within_limits()){
        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 

    } else {
        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
    }


    if(curr_iter >= 2500 + 1500){
        _phase = 19;
        _motion_start_iter = _state_iter+1;
    } 

}

void Recovering::_Climb4(const int & curr_iter){

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

    }

    pos_impl[0] << -arm_ab_set, climb_th1_p_r, climb_th2_p_r;
    pos_impl[1] << arm_ab_set, climb_th1_p, climb_th2_p;
    pos_impl[2] << 0, initial_jpos[2][1], initial_jpos[2][2];
    pos_impl[3] << leg_ab_touch, climb_up_th1, climb_up_th2;

    if (_within_limits()){
        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 

    } else {
        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
    }


    if(curr_iter >= 2500 + 1000){
        _phase = 21; // _Pull
        _motion_start_iter = _state_iter+1;
    } 

}

void Recovering::_Pull(const int & curr_iter){

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;


    }

    float delta_stretch = 1.3;
    pos_impl[0] << -arm_ab_set, climb_th1_p_r - delta_stretch, climb_th2_p_r;
    pos_impl[1] << arm_ab_set, climb_th1_p - delta_stretch, climb_th2_p;
    pos_impl[2] << 0, -PI/2, 0;
    pos_impl[3] << leg_ab_touch, climb_up_th1 - delta_stretch, climb_up_th2;

    if (_within_limits()){
        for(size_t leg(0); leg<2; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 
        for(size_t leg(2); leg<4; ++leg){
            _Step(curr_iter, 1000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 

    } else {
        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
    }


    if(curr_iter >= 2000){
        _phase = 22;
        _motion_start_iter = _state_iter+1;
    } 
    

    
}

void Recovering::_Pull1(const int & curr_iter){

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

    }

    float delta_stretch = 1.3;
    pos_impl[0] << -1.3, climb_th1_p_r - delta_stretch, climb_th2_p_r;
    pos_impl[1] << 1.3, climb_th1_p - delta_stretch, climb_th2_p;
    pos_impl[2] << -1.3, initial_jpos[2][1], 0;
    pos_impl[3] << 1.3, climb_up_th1 - delta_stretch, climb_up_th2;

    if (_within_limits()){
        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 
    } else {
        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
    }

    if(curr_iter >= 2500){
        _phase = 24; // shik Pull2
        _motion_start_iter = _state_iter+1;
    } 
    
}

void Recovering::_Pull2(const int & curr_iter){

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

    }

    float delta_stretch = 1.3;
    pos_impl[0] << -1.3, climb_th1_p_r - delta_stretch, climb_th2_p_r;
    pos_impl[1] << 1.3, climb_th1_p - delta_stretch, climb_th2_p;
    pos_impl[2] << -1.3, initial_jpos[2][1], 0;
    pos_impl[3] << 1.3, climb_up_th1 - delta_stretch, climb_up_th2;

    if (_within_limits()){
        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 
    } else {
        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
    }

    if(curr_iter >= 2000){
        _phase = 24;
        _motion_start_iter = _state_iter+1;
    } 
    
}

void Recovering::_Pull3(const int & curr_iter){

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

    }

    // float delta_stretch = 1.3;
    pos_impl[0] << -1.3, -1.0f, 2.3f;
    pos_impl[1] << 1.3, -1.0f, 2.3f;
    pos_impl[2] << -1.3, -0.5f, 2.0f;
    pos_impl[3] << 1.3, -0.5f, 2.0f;
    // pos_impl[3] << 1.3, -1.1f, climb_up_th2;

    if (_within_limits()){
        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 
    } else {
        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
    }

    if(curr_iter >= 2000){
        _phase = 25;
        _motion_start_iter = _state_iter+1;
    } 
    
}

void Recovering::_RecoverOnTable(const int & curr_iter){

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i)
          initial_jpos[i] = this->_legController->datas[i].q;
        for(size_t i(0); i < 2; ++i)
            initial_jpos[i] = initial_jpos[i] - front_offset;
        for(size_t i(2); i < 4; ++i)
            initial_jpos[i] = initial_jpos[i] - back_offset;

    }

    pos_impl[0] << -0.0f, -1.0f, 2.3f;
    pos_impl[1] <<  0.0f, -1.0f, 2.3f;
    // pos_impl[2] << 0, -1.4f, 2.0f;
    // pos_impl[3] << -0.0f, -1.4f, 2.0f;
    pos_impl[2] << 0, -0.5f, 2.0f;
    pos_impl[3] << -0.0f, -0.5f, 2.0f;

    if (_within_limits()){
        for(size_t leg(0); leg<4; ++leg){
            _Step(curr_iter, 2000, 
            leg, initial_jpos[leg], pos_impl[leg]);
        } 
    } else {
        _phase = -911;
        std::cout << "[DONE] SECURITY CHECK FAIL !!!!!!   ACTIONS GO BEYOND LIMITS !!!!!! " << "\n";
    }

    if(curr_iter >= 2500 + 999999){
        _phase = -17;
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

    if (_phase == 11 && (rc_mode == 11 || rc_mode == 13)) {
        _phase = 12;
        _motion_start_iter = _state_iter;

    } else if (_phase == 5 && (rc_mode == 11 || rc_mode == 13)){
        // delta_x_buffer.push_back(param_opt[8] + 0.1*rc_value);
        // delta_x_sum += param_opt[8] + 0.1*rc_value;

        // if (delta_x_buffer.size() > 100){
        //     delta_x_sum -= delta_x_buffer[0];
        //     delta_x_buffer.erase(delta_x_buffer.begin());
        // } 
        // delta_x_buffered = delta_x_sum / delta_x_buffer.size();

        // std::cout << "BUFFERED DELTA: " << delta_x_buffered << std::endl;

        // param_b_buffered = std::min(std::max(param_opt[0] + rc_value*0.03f, B_range[0]), B_range[1]); 
        // std::cout << "BUFFERED PARAM B: " << param_opt[0] << " + " << rc_value << " * 0.03 = " << param_b_buffered << std::endl;

        // NOTE THAT IN SOME CASES THIS WILL BE OVERWRITTEN 
    } else if (_phase == 5 && rc_mode == 12) {
        stopping = true;

    } else if (_phase >= 6 && _phase < 10 && (rc_mode == 11 || rc_mode == 13)) {
        _phase = -1;

    }

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
        else if (pos_impl[leg][2] < -2.7 || pos_impl[leg][2] > 2.7 || std::isnan(pos_impl[leg][2])){    
            std::cout << "[SECURITY] JOINT 2 OF LEG " << leg << " GOT A CRAZY ACTION: "<< pos_impl[leg][2] << "\n";
            return false;
        }
    }

    return true;
}

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