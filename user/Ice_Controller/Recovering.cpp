#include <Recovering.h>
// #include <pybind11/embed.h>
// #include <pybind11/stl.h>
// namespace py = pybind11;

Recovering::Recovering(LegController<float>* legController, StateEstimatorContainer<float>* stateEstimator){

    _legController = legController;
    _stateEstimator = stateEstimator;
}

Recovering::Recovering(){
    // goal configuration
    // Folding
    fold_jpos[0] << -0.0f, -1.4f, 2.7f;
    fold_jpos[1] << 0.0f, -1.4f, 2.7f;
    fold_jpos[2] << -0.0f, -1.4f, 2.7f;
    fold_jpos[3] << 0.0f, -1.4f, 2.7f;
    // Stand Up
    for(size_t i(0); i<4; ++i){
        //stand_jpos[i] << 0.f, -.9425f, 1.885f;
        stand_jpos[i] << 0.f, -.8f, 1.6f;
    }
    // Rolling
    rolling_jpos[0] << 1.5f, -1.6f, 2.77f;
    rolling_jpos[1] << 1.3f, -3.1f, 2.77f;
    rolling_jpos[2] << 1.5f, -1.6f, 2.77f;
    rolling_jpos[3] << 1.3f, -3.1f, 2.77f;

    f_ff << 0.f, 0.f, -25.f;
    zero_vec3 << 0.f, 0.f, -0.f;

    
    // agent = py::module::import("Agents").attr("cmaAgent")();
    // std::cout << "PYTHON OBJECT INITIALISATION TEST PASSED" << std::endl;
    // agent.attr("tell")(1.1f, true);
    // std::cout << "TELL MODULE TEST PASSED" << std::endl;
    // guess = agent.attr("ask")();
    // std::cout << "ASK MODULE TEST PASSED" << std::endl;
    
    // agent.attr("tell")(cost, firstIter);
    // guess = agent.attr("ask")();
    // std::cout << "TEST NO.1 " << std::endl;
    // py::initialize_interpreter();
    // py::module::import("sys").attr("argv").attr("append")("");
    // agent = py::module::import("Agents").attr("boAgent")();
    // std::cout << "TEST NO.2 " << std::endl;
    // py::object agent000 = py::module::import("Agents").attr("boAgent0")();
    // std::cout << "TEST NO.3 " << std::endl;
    // agent000.attr("tell")(1.1f, true);
    // std::cout << "TEST NO.4 " << std::endl;
    // py::list guess000 = agent000.attr("ask")();
    // std::cout << "TEST NO.5 " << std::endl;

}

void Recovering::runtest() {

    // std::cout << "TESTING!! PHASE: "<< _phase << "STEPS: " << _state_iter << std::endl;
    // _done = true;
    // testSmoothControl(_state_iter - _motion_start_iter);
    // testControlSignal();
    // _phase = 111;
    // std::cout << "A FUCKING NEW RUN IS STARTED! " << std::endl;
    body_height = _stateEstimator->getResult().position[2];
    std::cout << "HEIGHT: "<< body_height << std::endl;

    // if (!isSafe()){
    //   _phase = -1;
    //   std::cout << "SAFE CHECK FAIL! " << std::endl;
    // }

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
          _FrontLegsActions(_state_iter - _motion_start_iter);
          _RearLegsActions(_state_iter - _motion_start_iter);
          break;
      case 4:
          _train();
          break;
      case 5:
          _Finish();
          break;
    }

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

void Recovering::run() {

    // std::cout << "RECOVERING!! FLAG: " << _flag << "  STEPS: " << _state_iter << std::endl;
    // this->guess = agent.attr("ask")();

  switch(_flag){
    case StandUp:
      _StandUp(_state_iter - _motion_start_iter);
      break;
    case FoldLegs:
      _FoldLegs(_state_iter - _motion_start_iter);
      break;
    case RollOver:
      _RollOver(_state_iter - _motion_start_iter);
      break;
  }

 ++_state_iter;
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
    kdMat << 1, 0, 0, 0, 1, 0, 0, 0, 1;

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

void Recovering::_StandUp(const int & curr_iter){

  // std::cout << "_StandUp IS CALLED ! " << std::endl;


  body_height = this->_stateEstimator->getResult().position[2];
  
  bool something_wrong(false);

  if( _UpsideDown() || (body_height < 0.1 ) ) { 
    something_wrong = true;
  }

  if( (curr_iter > floor(standup_ramp_iter*0.7) ) && something_wrong){
    // If body height is too low because of some reason 
    // even after the stand up motion is almost over 
    // (Can happen when E-Stop is engaged in the middle of Other state)
    for(size_t i(0); i < 4; ++i) {
      initial_jpos[i] = this->_legController->datas[i].q;
    }
    _flag = FoldLegs;
    _motion_start_iter = _state_iter+1;

    printf("[Recovery Balance - Warning] body height is still too low (%f) or UpsideDown (%d); Folding legs \n", 
        body_height, _UpsideDown() );

  }else{
    for(size_t leg(0); leg<4; ++leg){
      _SetJPosInterPts(curr_iter, standup_ramp_iter, 
          leg, initial_jpos[leg], stand_jpos[leg]);
    }
  }
  // feed forward mass of robot.
  //for(int i = 0; i < 4; i++)
  //this->_data->_legController->commands[i].forceFeedForward = f_ff;
  //Vec4<T> se_contactState(0.,0.,0.,0.);
  Vec4<float> se_contactState(0.5,0.5,0.5,0.5);
  this->_stateEstimator->setContactPhase(se_contactState);

}

void Recovering::testControlSignal() {

    Vec3<float> qDes; Vec3<float> qdDes;

    kpMat << 80, 0, 0, 0, 80, 0, 0, 0, 80;
    kdMat << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    qDes << 0,0,0;
    qdDes << 0,0,0;

    for(int leg=0; leg<4; leg++){
      this->_legController->commands[leg].kpJoint = kpMat;
      this->_legController->commands[leg].kdJoint = kdMat;

      std::cout << "ACTIONS SENT TO _legController : " << qDes[0] << ", "<< qDes[1]<<", "<< qDes[2] <<std::endl;
      this->_legController->commands[leg].qDes = qDes;
      this->_legController->commands[leg].qdDes = qdDes;

    }
    
}

void Recovering::testSmoothControl(const int & curr_iter) {

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i) {
        initial_jpos[i] = this->_legController->datas[i].q; 
        }
      std::cout<< "STAND UP INITIALISATION DONE!" << curr_iter <<std::endl;
    }
    for(size_t i(0); i<4; ++i){
        //stand_jpos[i] << 0.f, -.9425f, 1.885f;
        stand_jpos[i] << 0.f, 0.f, 0.f;
        // initial_jpos[i] << 1.52302f, -0.79457f, 1.59995f; //1.52302, -0.79457, 1.59995
        initial_jpos[i] << 0.f, 0.f, 0.f;
    }
    Vec3<float> inter_pos;
    inter_pos<< 0,0,0;
    for(size_t leg(0); leg<4; ++leg){
      this->jointPDControl(leg, inter_pos, zero_vec3);
    }
    // for(size_t leg(0); leg<4; ++leg){

    //   float a(0.f);
    //   float b(1.f);

    //   if(curr_iter <= standup_ramp_iter) {
    //     b = (float)curr_iter/(float)standup_ramp_iter;
    //     a = 1.f - b;
    //   }
    //   // compute setpoints
    //   inter_pos = a * initial_jpos[leg] + b * stand_jpos[leg];
    //   inter_pos<< 0,0,0;


    //   this->jointPDControl(leg, inter_pos, zero_vec3);

    // }       
    
}

void Recovering::_JustStandUp(const int & curr_iter){

    // std::cout<< "STAND UP STEP: NO." << curr_iter <<std::endl;
    // std::cout<< "[1] STAND UP INI: " << initial_jpos[0][0] <<", "<< initial_jpos[0][1] <<", "<<  initial_jpos[0][2] << std::endl;
    // std::cout<< "[2] ]NOW: " << this->_legController->datas[0].q[0] <<", "<< this->_legController->datas[0].q[1] <<", "<<  this->_legController->datas[0].q[2] << std::endl;

    // std::cout<< "[3] STAND UP FIN: " << stand_jpos[0][0] <<", "<< stand_jpos[0][1] << ", "<< stand_jpos[0][2] << std::endl;

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i) {
        initial_jpos[i] = this->_legController->datas[i].q;
        }
      // std::cout<< "STAND UP INITIALISATION DONE!" << curr_iter <<std::endl;
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

    if(curr_iter >= standup_ramp_iter+300){
        std::cout<< "====================== STAND UP FINISHED! ======================" << curr_iter <<std::endl;
         _phase = 2;
        // _phase = 0; // JUST FOR TEST
        _motion_start_iter = _state_iter+1;
    } 

}

void Recovering::_FoldLegs(const int & curr_iter){
  if (curr_iter==0){
    for(size_t i(0); i < 4; ++i) {
      initial_jpos[i] = this->_legController->datas[i].q;
      
    }
    std::cout<< "====================== FOLD LEG BEGINS! ======================" << curr_iter <<std::endl;
  }

  // std::cout << "_FoldLegs IS CALLED ! " << std::endl;

  for(size_t i(0); i<4; ++i){
    _SetJPosInterPts(curr_iter, fold_ramp_iter, i, 
        initial_jpos[i], fold_jpos[i]);
  }
  if(curr_iter >= fold_ramp_iter + fold_settle_iter){
    if(_UpsideDown()){
      _flag = RollOver;
      for(size_t i(0); i<4; ++i) initial_jpos[i] = fold_jpos[i];
    }else{
      _flag = StandUp;
      for(size_t i(0); i<4; ++i) initial_jpos[i] = fold_jpos[i];
    }
    _phase = 1;
    _motion_start_iter = _state_iter + 1;
  }
}


void Recovering::_Prepare(const int & curr_iter){

    // std::cout << "PREPARE STEP:  " << curr_iter << std::endl;

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i) {
        initial_jpos[i] = this->_legController->datas[i].q;
        }
    }

    prepare_jpos[0] << -0.0f, -1.4f, 2.7f;
    prepare_jpos[1] << 0.0f, -1.4f, 2.7f;
    prepare_jpos[2] << -0.0f, -1.4f, 2.7f;
    prepare_jpos[3] << 0.0f, -1.4f, 2.7f;

    for(size_t i(0); i<4; ++i){
        _SetJPosInterPts(curr_iter, fold_ramp_iter, i, 
        initial_jpos[i], prepare_jpos[i]);
    }

    if(curr_iter >= fold_ramp_iter + 400){
        _phase = 3;
        _motion_start_iter = _state_iter + 1;
    }
}

void Recovering::_FrontLegsActions(const int & curr_iter){

    // std::cout << "TRY TO JUMP !!!!" << std::endl;

    if (curr_iter==0){
        for(size_t i(0); i < 4; ++i) {
        initial_jpos[i] = this->_legController->datas[i].q;
        }
    }

    int length = x_a*50*test_factor; //250
    for(size_t i(0); i<2; ++i){
        front_jpos[i] << 0.f, -.8*x_b, 1.6*x_c;
    }

    for(size_t leg(0); leg<2; ++leg){
      _Step(curr_iter, length, 
          leg, initial_jpos[leg], front_jpos[leg]);
    }


    Vec4<float> se_contactState(0.5,0.5,0.5,0.5);
    // this->_stateEstimator->setContactPhase(se_contactState);

    body_height = _stateEstimator->getResult().position[2];
    float v_curr =_stateEstimator->getResult().vBody.squaredNorm();
    if(body_height>max_body_height){
        max_body_height = body_height;
        top_velocity = v_curr;
    }
    // std::cout << "TESTING!!  max_body_height: " << max_body_height << "  top_velocity: "<< top_velocity << std::endl;
    if(curr_iter >= 1500){
        // _phase = 4; // STOP
        _motion_start_iter = _state_iter + 1;
    }

}


void Recovering::_RearLegsActions(const int & curr_iter){

    if (curr_iter==0){
      for(size_t i(0); i < 4; ++i) {
          initial_jpos[i] = this->_legController->datas[i].q;
      }
    }
    
    for(size_t i(0); i<2; ++i){
        // rear_jpos[i] << 0.f, -.8f, 1.6f;
        rear_jpos[i] << 0.f, -1.2f, 0.f;
    }

    int length = x_d*200*test_factor;
    if (curr_iter > x_e*100){
      length = x_f*150*test_factor; //250
    }

    for(size_t leg(2); leg<4; ++leg){
    _Step(curr_iter, length, 
        leg, initial_jpos[leg], rear_jpos[leg-2]);
    }

    Vec4<float> se_contactState(0.5,0.5,0.5,0.5);
    // this->_stateEstimator->setContactPhase(se_contactState);

}



void Recovering::_Step(
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

    Vec3<float> vec3;
    // vec3 << 0.0, -20.0, 20.0;
    vec3 << 0.0, 0.0, 0.0;

    // do control
    this->jointPDControl(leg, inter_pos, vec3);


}

void Recovering::_train(){
  // while (!_finishReset){
  //   _done = true;
  // }
  _done = true;
  
  /*
  // std::cout << "SCORE OF THIS EPISODE: " << -top_velocity << std::endl;
  // std::cout << "SEND THE SCORE ..." << std::endl;
  cost = top_velocity - max_body_height;
  agent.attr("tell")(-cost, firstIter);
  firstIter = false;

  // NOW BEGIN THE NEXT EPISODE
  max_body_height = 0;
  _phase = 0;  //set the flag of the mini FSM
  std::cout << "GET THE NEXT GUESS ..." << std::endl;
  guess = agent.attr("ask")();
  // fake_guess = agent.attr("shit")();
  // guess = fake_guess;

  for (int i=0; i<6; i++){
    X[i] = guess[i].cast<float>();
    std::cout << "X: " << X[i] << ",  ";
  }
  std::cout << std::endl;

  x_a=X[0]; x_b=X[1]; x_c=X[2]; x_d=X[3]; x_e=X[4]; x_f=X[5];
  */
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
  std::cout << "I was stopped! "  << std::endl; 
  // _phase = 0;
  _motion_start_iter = _state_iter + 1;
  // std::cout << "SHIT. WHAT IF I DO NOTHING ?? "  << std::endl; 
  // kpMat << 80, 0, 0, 0, 80, 0, 0, 0, 80;
  // kdMat << 1, 0, 0, 0, 1, 0, 0, 0, 1;

  // for(int leg=0; leg<4; leg++){
  //   this->_legController->commands[leg].kpJoint = kpMat;
  // }
  // std::cout << "Set Kp and Kd successfully! "  << std::endl; 
  // for (int leg = 0; leg < 4; leg++) {
  //   this->_legController->commands[leg].zero();
  // }
  // for (int leg=0; leg<4; leg++){
  //   for (int j=0; j<3; j++){
  //     std::cout << "[DEBUG] Set leg " <<leg<< " joint "<<j<<" to zero"  << std::endl; // 0
  //     this->_legController->commands[leg].tauFeedForward[0] = 0.0;
  //   }
  // }
}

void Recovering::_Finish(){
  std::cout << "I was finished! "  << std::endl; 
  _phase = 5;
  _motion_start_iter = _state_iter + 1;

}