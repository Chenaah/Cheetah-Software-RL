include_directories("./")
include_directories("../Fun_Controller/")
include_directories("../Fun_Controller/Controllers")
include_directories("../Fun_Controller/Controllers/BackFlip")
include_directories("../../robot/include")
include_directories("../../common/include/")
include_directories("../../common/FootstepPlanner")
include_directories("../../third-party/")
include_directories(SYSTEM "../../third-party/qpOASES/include")
include_directories("../../third-party/ParamHandler")

include_directories("../../lcm-types/cpp")
include_directories("/usr/local/include/lcm/")   # lcm includes
include_directories("/usr/local/include/eigen3")   # lcm includes
include_directories(${CMAKE_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR})
# list(APPEND CMAKE_PREFIX_PATH "../../third-party/libtorch")
# find_package(Torch REQUIRED)
# set(ONNX_PATH "../../third-party/onnxruntime")
set(ONNX_PATH "/home/chen/fun_v0.3/third-party/onnxruntime")
include_directories("${ONNX_PATH}/include")

file(GLOB_RECURSE sources
"./*.cpp"
# "../Fun_Controller/FSM_States/*.cpp" 
# "../Fun_Controller/Controllers/BackFlip/*.cpp" 
# "../Fun_Controller/Controllers/BalanceController/*.cpp" 
# "../Fun_Controller/Controllers/convexMPC/*.cpp"
)

# set(PYTHON_EXECUTABLE "/home/user/miniconda3/bin/python")
# set(PYTHON_LIBRARY "/home/user/miniconda3/lib/libpython3.7m.so")

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# find_package(pybind11 REQUIRED)
# add_subdirectory(pybind11)

add_executable(zero ${sources} main.cpp Training_Controller.cpp DataManager.cpp BackFlipTesting.cpp Recovering.cpp SafetyCheckerPro.cpp)

# target_link_libraries(zero "${TORCH_LIBRARIES}")
target_link_libraries(zero robot biomimetics)
target_link_libraries(zero dynacore_param_handler qpOASES)
target_link_libraries(zero Goldfarb_Optimizer osqp)
# target_link_libraries(zero tensorflow)
# target_link_libraries(zero torch)
target_link_libraries(zero "${ONNX_PATH}/lib/libonnxruntime.so")
target_link_libraries(zero zmq)
# target_link_libraries(zero WBC_Ctrl)
# target_link_libraries(zero VisionMPC)

