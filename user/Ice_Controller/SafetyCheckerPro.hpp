#pragma once
#include <Controllers/LegController.h>

class Guard{
    public:
        Guard();

        bool initial_jpos_safe();
        bool jpos_safe();

        bool global_safe = false;

        LegController<float>* _legController;

        
};