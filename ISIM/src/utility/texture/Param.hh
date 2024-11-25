#pragma once

#include "Size.hh"
#include <SFML/Graphics.hpp>

class Param {
    public:
        sf::Image image;

        virtual ~Param() = default;
        virtual void run(Size& size, std::string saveAs) = 0;
};
