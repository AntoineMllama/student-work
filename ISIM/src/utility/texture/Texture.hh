#pragma once

#include <string>
#include "Size.hh"
#include "Param.hh"

class Texture{

public:

    Texture(Size* size, std::string type, std::string saveAS);
    ~Texture();

    Size* size;
    std::string saveAS;
    std::string type;
    Param* param;
};