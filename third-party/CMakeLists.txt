add_subdirectory(Goldfarb_Optimizer)
add_subdirectory(ParamHandler)
add_subdirectory(inih)
add_subdirectory(osqp)
add_subdirectory(JCQP)
add_subdirectory(qpOASES)
add_subdirectory(lord_imu)
add_subdirectory(SOEM)

# set(PYTHON_EXECUTABLE "/home/user/miniconda3/bin/python")
# set(PYTHON_LIBRARY "/home/user/miniconda3/lib/libpython3.7m.so")
# add_subdirectory(pybind11)


if(CMAKE_SYSTEM_NAME MATCHES Linux)
add_subdirectory(vectornav)
endif()
