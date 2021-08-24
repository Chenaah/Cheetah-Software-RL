#include <main_helper.h>
#include "Training_Controller.hpp"
// #include <pybind11/embed.h>

// namespace py = pybind11;

int main(int argc, char** argv) {
  std::cout << "I AM A MAIN !!! " << std::endl;
  // py::initialize_interpreter();
  // (void) argc; (void) argv;
  // std::cout << "TEST NO.1 " << std::endl;
  // py::initialize_interpreter();
  // std::cout << "TEST NO.2 " << std::endl;
  // py::object agent000 = py::module::import("Agents").attr("boAgent0")();
  // std::cout << "TEST NO.3 " << std::endl;
  // agent000.attr("tell")(1.1f, true);
  // std::cout << "TEST NO.4 " << std::endl;
  // py::list guess000 = agent000.attr("ask")();
  // std::cout << "TEST NO.5 " << std::endl;


  main_helper(argc, argv, new Training_Controller());
  return 0;
}
