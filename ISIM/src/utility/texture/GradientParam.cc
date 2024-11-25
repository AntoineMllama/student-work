#include "GradientParam.hh"
#include <iostream>
#include <cmath>

GradientParam::GradientParam(std::vector<sf::Color> color, int angle)
:color(color)
,angle(angle)
{}

void GradientParam::run(Size& size, std::string saveAs) {
    
    int width = size.width;
    int height = size.height;
    sf::Image image;

    image.create(width, height);

    float radAngle = angle * 3.14159265 / 180.0;
    float cosA = std::cos(radAngle);
    float sinA = std::sin(radAngle);

    unsigned int numColors = this->color.size();
    if (numColors < 2) {
        std::cerr << "Il faut au moins deux couleurs pour créer un gradient." << std::endl;
        exit(-1);
    }

    float halfWidth = width / 2.0f;
    float halfHeight = height / 2.0f;

    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            float dx = x - halfWidth;
            float dy = y - halfHeight;

            float distance = (dx * cosA + dy * sinA + halfWidth * std::sqrt(2.0f)) / (width * std::sqrt(2.0f));
            distance = std::max(0.0f, std::min(1.0f, distance));

            float scaledDistance = distance * (numColors - 1);
            unsigned int colorIndex = static_cast<unsigned int>(scaledDistance);
            float factor = scaledDistance - colorIndex;

            sf::Color color;
            if (colorIndex < numColors - 1) {
                sf::Uint8 red = this->color.at(colorIndex).r + factor * (this->color.at(colorIndex + 1).r - this->color.at(colorIndex).r);
                sf::Uint8 green = this->color.at(colorIndex).g + factor * (this->color.at(colorIndex + 1).g - this->color.at(colorIndex).g);
                sf::Uint8 blue = this->color.at(colorIndex).b + factor * (this->color.at(colorIndex + 1).b - this->color.at(colorIndex).b);
                sf::Uint8 a = this->color.at(colorIndex).a + factor * (this->color.at(colorIndex + 1).a - this->color.at(colorIndex).a);
                color = sf::Color(red, green, blue, a);
            } else {
                color = this->color.back();
            }

            image.setPixel(x, y, color);
        }
    }
    this->image = image;
    this->image = image;
    if (!saveAs.empty()) {
        if (!image.saveToFile(saveAs)) {
            std::cout << "SaveAs Fail with : " + saveAs + '\n';
        }
    }
}