#pragma once
#include <Recovering.h>
#include <Controllers/LegController.h>
#include "Controllers/StateEstimatorContainer.h"
// #include <pybind11/embed.h>
// #include <pybind11/stl.h>
// namespace py = pybind11;
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
    bool _done = false;
    bool _finishReset = false;

    float body_height;
    float max_body_height;
    float top_velocity;

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
    Vec3<float> zero_vec3;

    Vec3<float> f_ff;

    Vec3<float> front_jpos[2];
    Vec3<float> rear_jpos[2];
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
    const int fold_ramp_iter = 500;
    const int fold_settle_iter = 700;

    const int standup_ramp_iter = 500;
    const int standup_settle_iter = 250;

    void _RollOver(const int & iter);
    void _StandUp(const int & iter);
    void _JustStandUp(const int & iter);
    void _FoldLegs(const int & iter);

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

    void _train();
    void _Finish();
    


    // Create the cartesian P gain matrix
    Mat3<float> kpMat;

    // Create the cartesian D gain matrix
    Mat3<float> kdMat;

    // float x_a=1, x_b=1, x_c=1, x_d=1, x_e=1, x_f=1;    //[0.963197,  1.09587,  1.0227,  0.909838,  1.09634,  1.08451]
    float x_a=0.95, x_b=1.09587, x_c=1.0227, x_d=0.909838, x_e=1.09634, x_f=1.08451;    //[0.963197,  1.09587,  1.0227,  0.909838,  1.09634,  1.08451]
    float X[6];
    bool firstIter = true;
    float cost;
    int test_factor = 1.0;
   // Create the BO agent from python
   // py::scoped_interpreter guard{};
    
   // py::object agent;
   // py::list guess;

};