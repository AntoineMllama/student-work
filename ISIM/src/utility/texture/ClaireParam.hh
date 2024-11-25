#pragma once
#include "Param.hh"
#include "Space.hh"
#include "texture/Texture.hh"


class ClaireParam : public Param {

    public:
        ClaireParam(int a, int b, sf::Color up, Texture* textureUP, sf::Color down, Texture* textureDOWN, sf::Color left, Texture* textureLEFT, sf::Color right, Texture* textureRIGHT, Space* space);
        ~ClaireParam();

        int a;
        int b;
        sf::Color up;
        Texture* textureUP;
        sf::Color down;
        Texture* textureDOWN;
        sf::Color left;
        Texture* textureLEFT;
        sf::Color right;
        Texture* textureRIGHT;
        Space* space;

        void run(Size& size, std::string saveAs);
};
