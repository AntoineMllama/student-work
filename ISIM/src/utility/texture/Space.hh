#pragma once

#include <SFML/Graphics.hpp>

#include "Texture.hh"

class Space {

    public:
        Space(sf::Color color, int len, Texture* texture);
        ~Space();

        sf::Color color;
        int len;
        Texture* texture;
};
