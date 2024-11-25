#pragma once

#include "Param.hh"
#include "Space.hh"
#include "Size.hh"
#include "Texture.hh"

class BrickParam : public Param {

    public:
        BrickParam(sf::Color color, Size* size, Space* space, Texture* texture_brick);
        BrickParam(Size* size, Space* space, Texture* texture_brick);
        ~BrickParam();

        void run(Size& size, std::string saveAs);

        sf::Color color;
        Size* size;
        Space* space;
        Texture* texture_brick;
};
