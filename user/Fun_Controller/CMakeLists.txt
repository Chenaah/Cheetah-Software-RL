include_directories("./")
include_directories("./Controllers")
include_directories("../../robot/include")
include_directories("../../common/include/")
include_directories("../../common/FootstepPlanner")
include_directories("../../third-party/")
include_directories(SYSTEM "../../third-party/qpOASES/include")
include_directories("../../third-party/ParamHandler")

include_directories("../../lcm-types/cpp")
include_directories("/usr/local/include/lcm/")   # lcm includes
include_directories("/usr/local/include/eigen3")   # lcm includes
#include_directories("/usr/local/include/librealsense2") 
include_directories(${PROJECT_SOURCE_DIR})


file(GLOB_RECURSE sources
"./*.cpp"
"FSM_States/*.cpp" 
"Controllers/BalanceController/*.cpp" 
"Controllers/convexMPC/*.cpp")

add_subdirectory(Controllers/WBC)
add_subdirectory(Controllers/WBC_Ctrl)
add_subdirectory(Controllers/VisionMPC)

#    set(LIB_DIR "/usr/local/lib")
#    link_directories(${LIB_DIR})
#    link_libraries(librealsense2)

add_executable(fun ${sources} MIT_Controller.cpp main.cpp Controllers/BackFlip/FrontFlipUpCtrl.h Controllers/BackFlip/FrontFlipUpCtrl.cpp FSM_States/FSM_State_FrontJump2.h FSM_States/FSM_State_FrontJump2.cpp)
target_link_libraries(fun robot biomimetics)
target_link_libraries(fun dynacore_param_handler qpOASES)
target_link_libraries(fun Goldfarb_Optimizer osqp)
target_link_libraries(fun WBC_Ctrl)
target_link_libraries(fun VisionMPC)
#target_link_libraries(mit_ctrl realsense2)
include_directories(${CMAKE_BINARY_DIR})
