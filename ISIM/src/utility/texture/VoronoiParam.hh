#pragma once
#include "Param.hh"

class VoronoiParam : public Param{
public:
    VoronoiParam();
    void run(Size &size, std::string saveAs);
};
