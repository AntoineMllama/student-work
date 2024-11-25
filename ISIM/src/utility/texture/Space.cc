#include "Space.hh"

Space::Space(sf::Color color , int len, Texture* texture)
:color(color)
,len(len)
,texture(texture)
{}

Space::~Space() {
    delete texture;
}
