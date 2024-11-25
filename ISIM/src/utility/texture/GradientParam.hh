#pragma once

#include "Param.hh"

class GradientParam : public Param {

    public:
        GradientParam(std::vector<sf::Color> color, int angle);

        void run(Size& size, std::string saveAs);

        std::vector<sf::Color> color;
        int angle;
};