#include "BrickParam.hh"

#include <iostream>

BrickParam::BrickParam(sf::Color color, Size* size, Space* space, Texture* texture_brick)
:color(color)
,size(size)
,space(space)
,texture_brick(texture_brick)
{}

BrickParam::BrickParam(Size* size, Space* space, Texture* texture_brick)
:size(size)
,space(space)
,texture_brick(texture_brick)
{}




BrickParam::~BrickParam() {
    delete space;
    delete size;
    delete texture_brick;

}

void BrickParam::run(Size& size, std::string saveAs) {

    sf::Image image;
    sf::Image apply;
    bool sup_full = false;

    int width = size.width;
    int height = size.height;
    int brickWidth = this->size->width;
    int brickHeight = this->size->height;
    int spaceLen = this->space->len;

    if (this->texture_brick != nullptr) {
        this->texture_brick->param->run(*this->texture_brick->size, this->texture_brick->saveAS);
        apply = this->texture_brick->param->image;
        sup_full = apply.getSize().x == width && apply.getSize().y == height;
    }
    if (this->space->texture != nullptr) {
        this->space->texture->param->run(*this->space->texture->size, this->space->texture->saveAS);
        image = this->space->texture->param->image;
    }
    else {image.create(width, height, this->space->color);}

    for (unsigned int y = 0; y < height; y += brickHeight + spaceLen) {
        if ((y / (brickHeight + spaceLen)) % 2 == 1) {
            // demi-brique au lignes impaires
            for (unsigned int i = 0; i < brickHeight; ++i) {
                for (unsigned int j = 0; j < brickWidth / 2; ++j) {
                    if (j < width && y + i < height) {
                        if (sup_full) {
                            image.setPixel(j, y + i, apply.getPixel(j, y + 1));
                        }
                        else if (this->texture_brick != nullptr) {
                            image.setPixel(j, y + i, apply.getPixel(j, i));
                        }
                        else {
                            image.setPixel(j, y + i, this->color);
                        }
                    }
                }
            }
        }

        for (unsigned int x = 0; x < width; x += brickWidth + spaceLen) {
            unsigned int xOffset = (y / (brickHeight + spaceLen)) % 2 == 1 ? (brickWidth / 2 + spaceLen) : 0;
            unsigned int xBrickStart = x + xOffset;
            unsigned int yBrickStart = y;

            for (unsigned int i = 0; i < brickHeight; ++i) {
                for (unsigned int j = 0; j < brickWidth; ++j) {
                    if (xBrickStart + j < width && yBrickStart + i < height) {
                        if (sup_full) {
                            image.setPixel(xBrickStart + j, yBrickStart + i, apply.getPixel(xBrickStart + j, yBrickStart + i));
                        }
                        else if (this->texture_brick != nullptr) {
                            image.setPixel(xBrickStart + j, yBrickStart + i, apply.getPixel(j, i));
                        }
                        else {
                            image.setPixel(xBrickStart + j, yBrickStart + i, this->color);
                        }
                    }
                }
            }
        }
    }
    this->image = image;
    if (!saveAs.empty()) {
        if (!image.saveToFile(saveAs)) {
            std::cout << "SaveAs Fail with : " + saveAs + '\n';
        }
    }
}