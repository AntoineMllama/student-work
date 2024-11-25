#pragma once

#include "Param.hh"

class PerlinParam : public Param {

    public:
        PerlinParam(std::string type);

        void run(Size& size, std::string saveAs);

        std::string type;

        sf::Color bois3_inf;
        sf::Color bois3_mid;
        sf::Color bois3_up;

        double borne_inf;
        double borne_sup;
private:
    void bois1(Size& size);
    void bois2(Size& size);
    void marbre1(Size& size);
    void marbre2(Size& size);
    void ciel(Size& size);
    void ocean(Size& size);
    void bois3(Size& size);
};