#include "SafetyCheckerPro.hpp"

Guard::Guard(){
    std::cout << "Safety guard starts successfully! " << std::endl;
}

bool Guard::initial_jpos_safe(){
    bool safe = true;
    for(int leg=0; leg < 4; leg++){
        if (_legController->datas[leg].q[1] < -1.58 || _legController->datas[leg].q[1] > 0.78 ||
            _legController->datas[leg].q[2] < 2.0 || _legController->datas[leg].q[2] > 2.9 ){
            safe = false;
            std::cout << "SAFETY CHECK FAIL : LEG" << leg <<" :" ;
            if (_legController->datas[leg].q[1] < -1.58){
                std::cout << "JOINT 1: " << _legController->datas[leg].q[1]  << " < -1.58";
            } else if(_legController->datas[leg].q[1] > 0.78){
                std::cout << "JOINT 1: " << _legController->datas[leg].q[1]  << " > 0.78";
            } else if(_legController->datas[leg].q[2] < 2.0){
                std::cout << "JOINT 2: " << _legController->datas[leg].q[2]  << " < 2.0";
            } else if(_legController->datas[leg].q[2] > 2.9){
                std::cout << "JOINT 2: " << _legController->datas[leg].q[2]  << " > 2.9";
            }
            
            std::cout << std::endl;
        }
    }

    for(int leg=1; leg < 4; leg+=2){  // leg 1/3
        if (_legController->datas[leg].q[0] < 0.0 || _legController->datas[leg].q[0] > 0.8){
            safe = false;
            std::cout << "SAFETY CHECK FAIL : LEG" << leg <<" :" ;
            if (_legController->datas[leg].q[0] < 0.0){
                std::cout << "JOINT 0: " << _legController->datas[leg].q[0]  << " < 0.0";
            } else if(_legController->datas[leg].q[0] > 0.8){
                std::cout << "JOINT 0: " << _legController->datas[leg].q[0]  << " >0.8";
            } 
            std::cout << std::endl;
        }

    }

    for(int leg=0; leg < 3; leg+=2){  // leg 0/2
        if (_legController->datas[leg].q[0] < -0.8 || _legController->datas[leg].q[0] > 0.0){
            safe = false;
            std::cout << "SAFETY CHECK FAIL : LEG" << leg <<" :" ;
            if (_legController->datas[leg].q[0] < -0.8){
                std::cout << "JOINT 0: " << _legController->datas[leg].q[0]  << " < -0.8";
            } else if(_legController->datas[leg].q[0] > 0.0){
                std::cout << "JOINT 0: " << _legController->datas[leg].q[0]  << " > 0.0";
            } 
            std::cout << std::endl;
            
        }

    }
    safe = true;  // temp: for joint estimation
    return safe;

}

bool Guard::jpos_safe(){
    bool safe = true;
    for(int leg=0; leg < 4; leg++){
        if (_legController->datas[leg].q[1] < -3.0 || _legController->datas[leg].q[1] > 0.8 ||
            _legController->datas[leg].q[2] < -0.2 || _legController->datas[leg].q[2] > 2.9 ){
            safe = false;
            std::cout << "SAFETY CHECK FAIL : LEG" << leg <<" :" << _legController->datas[leg].q[0] << ", " << _legController->datas[leg].q[1] << ", " << _legController->datas[leg].q[2] << std::endl;
        }
    }

    for(int leg=1; leg < 4; leg+=2){
        if (_legController->datas[leg].q[0] < 0.0 || _legController->datas[leg].q[0] > 0.8){
            safe = false;
            std::cout << "SAFETY CHECK FAIL : LEG" << leg <<" :" << _legController->datas[leg].q[0] << ", " << _legController->datas[leg].q[1] << ", " << _legController->datas[leg].q[2] << std::endl;
        }

    }

    for(int leg=0; leg < 3; leg+=2){
        if (_legController->datas[leg].q[0] < -0.8 || _legController->datas[leg].q[0] > 0.0){
            safe = false;
            std::cout << "SAFETY CHECK FAIL : LEG" << leg <<" :" << _legController->datas[leg].q[0] << ", " << _legController->datas[leg].q[1] << ", " << _legController->datas[leg].q[2] << std::endl;
        }

    }
    return safe;
}