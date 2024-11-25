#include "Texture.hh"
#include "Param.hh"

Texture::Texture(Size* size, std::string type, std::string saveAS)
: size(size)
, type(type)
, saveAS(saveAS)
{}

Texture::~Texture() {
    delete size;
    delete param;
}