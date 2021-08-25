#pragma once
#include <Recovering.h>
#include <Controllers/LegController.h>
#include "Controllers/StateEstimatorContainer.h"
#include "Controllers/DesiredStateCommand.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include "tensorflow/c/c_api.h"
#include "tf_utils_lite.hpp"
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <iomanip>
#include <initializer_list>
#include <fstream>
// #include <pybind11/embed.h>
// #include <pybind11/stl.h>
// namespace py = pybind11;
#define DEBUG false

#if DEBUG
// FOR DEBUGING: 
#include <lcm/lcm-cpp.hpp>
#include "state_estimator_lcmt.hpp"
#include "leg_control_data_lcmt.hpp"
#include <fstream>
#include <zmq.hpp>
#endif


// Normal robot states
#define K_PASSIVE 0
#define K_STAND_UP 1
#define K_BALANCE_STAND 3
#define K_LOCOMOTION 4
#define K_LOCOMOTION_TEST 5
#define K_RECOVERY_STAND 6
#define K_VISION 8
#define K_BACKFLIP 9
#define K_FRONTJUMP 11
#define K_FRONTFLIP 12
#define K_FRONTJUMP2 13


class Recovering {
 public:
    Recovering(LegController<float>* , StateEstimatorContainer<float>* );
    Recovering();
    void recover();
    void begin();
    void run();
    void runtest();
    void _Passive();

    LegController<float>* _legController;
    StateEstimatorContainer<float>* _stateEstimator;
    RobotControlParameters* _controlParameters;
    DesiredStateCommand<float>* _desiredStateCommand;


    bool _done = false;
    bool _finishReset = false;

    float body_height;
    float max_body_height;
    float top_velocity;

    // float front_factor = 0.857; //0.9; 
    // float back_factor = 1.4;

    int front_length = 71;
    int back_length1 = 136;
    int back_length2 = 306;
    int rear_ramp = 80;
    float hip_angle = -1.8;
    float knee_angle = 0;
    float step_size_case1 = 0.012; 
    float step_size_case2 = 0.012; 
    float step_size_case3 = 0.48; 

    float stand_front_hip = -0.2;
    float stand_front_knee = 1.6;
    float stand_back_hip = 0.2;
    float stand_back_knee = -0.272;

    float pre_front_hip = -1.0;
    float pre_front_knee = 2.4;
    float pre_back_hip = 0.7;
    float pre_back_knee = -0.272;

    float pre2_front_hip = -1.0;
    float pre2_front_knee = 2.4;
    float pre2_back_hip = 0.7;
    float pre2_back_knee = -0.272;

    float ad1 = 0.2;
    float ad2 = 1.2;

    // float pre3_front_hip = -1.0;
    // float pre3_front_knee = 2.4;
    // float pre3_back_hip = 0.7;
    // float pre3_back_knee = -0.272;

    float jump_front_hip = -0.5;
    float jump_front_knee = 1.2;
    bool pre1_enable = true;

    int bound_length = 120;

    float param_a = 0, param_b = 0;  // public for easiliy read from files
    std::vector<float> param_opt = std::vector<float>(9, 0);
    enum class Gait {sine, rose, triangle, line, none};
    Gait gait = Gait::triangle;  // changed to public varible for easily loading from files

    const float PI = 3.14159;

    void _update_action();

 private:
    // Keep track of the control iterations
    int iter = 0;
    int _motion_start_iter = 0;

    static constexpr int StandUp = 0;
    static constexpr int FoldLegs = 1;
    static constexpr int RollOver = 2;

    unsigned long long _state_iter=0;
    int _flag = FoldLegs;
    int _phase = 0;

    // JPos
    Vec3<float> fold_jpos[4];
    Vec3<float> stand_jpos[4];
    Vec3<float> rolling_jpos[4];
    Vec3<float> initial_jpos[4];
    Vec3<float> curr_jpos[4];
    Vec3<float> zero_vec3;

    Vec3<float> f_ff;

    Vec3<float> front_jpos[2];
    Vec3<float> rear_jpos[2];
    Vec3<float> rear_jpos_2[2];

    Vec3<float> prepare_jpos[4];

    void testControlSignal();
    void testSmoothControl(const int & curr_iter);

    bool isSafe();


    // iteration setup
    //const int rollover_ramp_iter = 300;
    //const int rollover_settle_iter = 300;

    //const int fold_ramp_iter = 1000;
    //const int fold_settle_iter = 1000;

    //const int standup_ramp_iter = 500;
    //const int standup_settle_iter = 500;

    // 0.5 kHz
    const int rollover_ramp_iter = 150;
    const int rollover_settle_iter = 150;

    //const int fold_ramp_iter = 500;
    //const int fold_settle_iter = 500;
    const int fold_ramp_iter = 1000;
    const int fold_settle_iter = 100;

    const int standup_ramp_iter = 1000;
    const int standup_settle_iter = 250;

    void _RollOver(const int & iter);
    void _StandUp(const int & iter);
    void _JustStandUp(const int & iter);
    void _FoldLegs(const int & iter);
    void _Prepare_2(const int & iter);
    void _Prepare_3(const int & iter);

    bool _UpsideDown();
    void _SetJPosInterPts(
    const size_t & curr_iter, size_t max_iter, int leg, 
    const Vec3<float> & ini, const Vec3<float> & fin);

    void jointPDControl(
    int leg, Vec3<float> qDes, Vec3<float> qdDes);
    void _FrontLegsActions(const int & curr_iter);
    void _RearLegsActions(const int & curr_iter);
    void _Prepare(const int & curr_iter);


    void _Step(
    const size_t & curr_iter, size_t max_iter, int leg, 
    const Vec3<float> & ini, const Vec3<float> & fin);

    void _MoveHands(const int & curr_iter);
    void _Finish();
    void _JustFoldLegs(const int & curr_iter);
    void _PushUp(const int & curr_iter);
    void _Bound(const int & curr_iter);
    
    void _LegsTest(const int & curr_iter);
    void _LegsTest2(const int & curr_iter);
    void _LegsTest3(const int & curr_iter);
    void _LegsTest4(const int & curr_iter);
    void _LegsTest5(const int & curr_iter);

    void _update_state();
    void _update_state_buffer();
    std::vector<float> state;
    std::vector<float> action;
    int a_dim = 0;  // this will set automatically by action_mode and leg_action_mode  none: 4, parameter/parallel_offset/hips_offset/knees_offset: 6, hips_knees_offset:8
    int s_dim = 0;  // this will set automatically by state_mode  body_arm: 10, body_arm_p: 12, body_arm_leg_full: 14

    const std::vector<float> stand_pos = {0.0, -0.795, 0.617, 0.0, -0.795, 0.617, 0.0, -0.94, -1, 0.0, -0.94, -1};
    std::vector<float> action_test = {0.6, -0.5, 0.7, 0.1};
    Vec3<float> pos_impl[4];

    void* _Test(void* args);
    void _Walk(const int & curr_iter);
    float _theta2_prime_hat(const float & th1_prime);
    float _theta1_hat(const float & th2);
    float _Box(const float & num, const float & min, const float & max);
    float _toPitch(const Eigen::Quaterniond& q);
    std::vector<float> _toRPY(const Eigen::Quaterniond& q);
    void _FK(const float & th1, const float & th2, float & x, float & y);
    void _IK(const float & x, const float & y, float & th1, float & th2);
    void _update_basic_leg_actions(const int & curr_iter);
    void _update_leg_offsets();
    void _process_leg_offsets(const int & curr_iter);
    void _update_rpy();
    void _process_remote_controller_signal(const int & curr_iter);
    void _SettleDown(const int & curr_iter);
    void _InverseRearLegsActions(const int & curr_iter);
    void _InversePrepare_2(const int & curr_iter);
    void _InverseJustStandUp(const int & curr_iter);
    bool _has_stopped();
    void _BoundToStand(const int & curr_iter);

    // Create the cartesian P gain matrix
    Mat3<float> kpMat;

    // Create the cartesian D gain matrix
    Mat3<float> kdMat;

    // float x_a=1, x_b=1, x_c=1, x_d=1, x_e=1, x_f=1;    //[0.963197,  1.09587,  1.0227,  0.909838,  1.09634,  1.08451]
    float x_a=0.963197, x_b=0.7, x_c=0.7, x_d=0.909838, x_e=1.09634, x_f=1.08451;    //[0.963197,  1.09587,  1.0227,  0.909838,  1.09634,  1.08451]
    float X[6];
    bool firstIter = true;
    float cost;

    double pitch, w_pitch, w_pitch_old, acc_pitch, pitch_error, pitch_eval;
    std::vector<double> w_pitch_buffer;
    double pitch_sum = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;
    double theta1, theta1_delta, theta2, theta1_act, theta2_act, theta1_delta_old, tau, theta1_prime, theta2_prime;
    float adaptive_hip;
    bool pre_adaptive_hip = false;

    bool p_control = false;
    bool bend_arms = false;
    bool hip_lock = false;
    int bend_start_iter = 0;
    double k_ = 0.516;

    const double gamma_ = 0.97213, l_ = 0.184, a_ = 0.395, b_ = 0.215, c_ = 0.215;
    #if !DEBUG
    const float k_final = 0.72; //0.69; // If the CoM is too forward, increase this value ~ A good choice in theory: 0.61
    #else
    const float k_final = 0.8;
    #endif

    const float front_hip_offset = 0.11, back_hip_offset = -0.155;
    Vec3<float> front_offset;
    Vec3<float> back_offset;

    const int num_sub_steps = 10;

    TF_Graph* graph;
    TF_Status* status;
    TF_SessionOptions* SessionOpts;
    TF_Buffer* RunOpts;
    TF_Session* sess;
    TF_Tensor* input_tensor;
    TF_Tensor* output_tensor;
    TF_Output input_op;
    TF_Output out_op;

    float ad_right_min, ad_right_max, ad_left_min, ad_left_max;

    std::vector<float> rot_mat = vector<float>(16, 0);
    std::vector<float> rpy;
    int bound_phase = 1, ini_bound_length=100;

    float p_error=0, d_error=0, p_error_yaw=0, d_error_yaw=0, abs_p_error=0, abs_d_error=0;
    std::vector<float> p_error_buffer, d_error_buffer, p_error_yaw_buffer, d_error_yaw_buffer;
    // For progressing control
    std::vector<float> abs_p_error_buffer, abs_d_error_buffer; 
    const unsigned int progressing_buffer_size = 1000;
    float progressing_param_a_multiplier = 0;
    ////
    float p_error_sum=0, d_error_sum=0, p_error_yaw_sum=0, d_error_yaw_sum=0, abs_p_error_sum=0, abs_d_error_sum=0;
    enum class ActionMode {partial, whole, residual, open_loop};
    enum class LegActionMode {none, parameter, parallel_offset, hips_offset, knees_offset, hips_knees_offset};
    enum class StateMode {h_body_arm, h_body_arm_p, body_arm, body_arm_p, body_arm_leg_full, body_arm_leg_full_filtered, body_arm_leg_full_p};
    int agent_ver = 3;
    float pitch_ref = 0;
    float old_sin_value = 0;
    
    float param_c_buffered = 0;
    float param_d_buffered = 0;

    std::vector<float> param_a_buffer;
    std::vector<float> param_b_buffer;
    float param_a_buffered = 0;  // elements pushed back to the buffer every timestep
    float param_b_buffered = 0;

    float period_half = 0, sub_t = 0;
    float x_togo_r = 0, x_togo_l = 0, y_togo_r = 0, y_togo_l = 0, x_original = 0, y_original = 0;
    float a_rose = 0, k_rose = 0, th = 0; // for rose gait
    float x_0 = 0, y_0 = 0, x_1 = 0, y_1 = 0, x_2 = 0, y_2 = 0; // for triangle gait
    float Kp_pitch = 0, Kd_pitch = 0, Kp_yaw = 0, Kd_yaw = 0, delta_x = 0;  // these values are given by vector param_opt
    std::vector<float> delta_x_buffer;
    float delta_x_buffered = 0, delta_x_sum = 0;

    Vec4<float> leg_offsets;
    Vec4<float> leg_offsets_old;

    bool stopping = false;
    int rc_mode = 0;

    float progressing_agent_factor = 0;
    std::vector<std::vector<float>> body_state_buffer;
    std::vector<float> body_state_sum = std::vector<float>(6, 0);

    // reward calculator
    float reward = 0.0;
    float episode_return = 0.0;
    float limit_cost = 0.0;
    float action_cost = 0.0;
    bool reach_limit = false;
    void _update_reward();

    bool agent_enable = false; // determined by whether the model exists

    // #if !DEBUG
    // Environment parameters
    const bool walk_enable = true;
    const float agent_factor = 1.0;
    const ActionMode action_mode = ActionMode::residual;
    const LegActionMode leg_action_mode = LegActionMode::hips_offset;
    const StateMode state_mode = StateMode::body_arm_leg_full;
    // const Gait gait = Gait::triangle;  // changed to public varible for easily loading from files
    const bool arm_pd_control = true;
    const bool fast_error_update = true;
    const float action_multiplier = 0.01;
    const float residual_multiplier = 0.2; // used for parameterised residual training, please check the code before using his mode 
    const float leg_offset_multiplier = 0.01; // 0.1;
    const std::vector<float> A_range = {0.0, 99};
    const std::vector<float> B_range = {0.01, 0.1};
    const std::vector<float> leg_offset_range = {-0.4, 0.4};
    const bool progressing = true;

    const unsigned int error_buffer_size = 15;
    const float arm_pd_multiplier = 1;
    const unsigned int state_buffer_size = 10;

   //  #else
   //  // Environment parameters
   //  const bool walk_enable = true;
   //  const bool agent_enable = false;
   //  const float agent_factor = 1;
   //  const ActionMode action_mode = ActionMode::residual;
   //  const LegActionMode leg_action_mode = LegActionMode::hips_offset;
   //  const StateMode state_mode = StateMode::body_arm_leg_full;
   //  const Gait gait = Gait::triangle;
   //  const bool arm_pd_control = true;
   //  const bool fast_error_update = true;
   //  const float action_multiplier = 0.01;
   //  const float residual_multiplier = 0.2; // used for parameterised residual training, please check the code before using his mode 
   //  const float leg_offset_multiplier = 0.01; // 0.1;
   //  const std::vector<float> A_range = {0.0, 99};
   //  const std::vector<float> B_range = {0.01, 0.1};
   //  const std::vector<float> leg_offset_range = {-0.4, 0.4};
   //  const bool progressing = false;

   //  const unsigned int error_buffer_size = 15;
   //  const float arm_pd_multiplier = 0.0;

   //  #endif

    // Debugging
    #if DEBUG
    lcm::LCM lcm;
    void receive_state_from_bullet(const lcm::ReceiveBuffer* rbuf, const std::string& chan, const state_estimator_lcmt* msg);
    void receive_leg_state_from_bullet(const lcm::ReceiveBuffer* rbuf, const std::string& chan, const leg_control_data_lcmt* msg);

    std::vector<float> bullet_omegaWorld = std::vector<float>(3, 0);
    std::vector<float> bullet_q = std::vector<float>(12, 0);
    std::vector<float> bullet_orientation = std::vector<float>(4, 0);

    std::ofstream debug_file;
    std::vector<std::string> debug_info;
    std::ostringstream debug_info_stream;

    const std::string socket_data{"World"};
    zmq::message_t request;
    // initialize the zmq context with a single IO thread
    zmq::context_t context{1};
    // construct a REP (reply) socket and bind to interface
    zmq::socket_t socket{context, zmq::socket_type::rep};
    std::vector<float> state_from_bullet;
    void wait_for_simulation_state();
    void send_action_to_simulator();
    #endif




};