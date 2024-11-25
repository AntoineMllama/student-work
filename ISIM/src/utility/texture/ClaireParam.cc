#include "ClaireParam.hh"
#include <cmath>
#include <iostream>

ClaireParam::ClaireParam(int a, int b, sf::Color up, Texture* textureUP, sf::Color down, Texture* textureDOWN, sf::Color left, Texture* textureLEFT, sf::Color right, Texture* textureRIGHT, Space* space)
:a(a)
,b(b)
,up(up)
,textureUP(textureUP)
,down(down)
,textureDOWN(textureDOWN)
,left(left)
,textureLEFT(textureLEFT)
,right(right)
,textureRIGHT(textureRIGHT)
,space(space)
{}

ClaireParam::~ClaireParam() {
    delete textureUP;
    delete textureDOWN;
    delete textureRIGHT;
    delete textureLEFT;
}


sf::ConvexShape createCairoPentagon(int a, int b, int x, int y) {
    sf::ConvexShape pentagon;
    pentagon.setPointCount(5);

    pentagon.setPoint(0, sf::Vector2f(x, a + b + y));
    pentagon.setPoint(1, sf::Vector2f(x + b, b + y));
    pentagon.setPoint(2, sf::Vector2f(x + (b - a), y));
    pentagon.setPoint(3, sf::Vector2f(x - (b - a), y));
    pentagon.setPoint(4, sf::Vector2f(x - b, b + y));

    return pentagon;
}

sf::ConvexShape createCairoPentagonInv(int a, int b, int x, int y) {
    sf::ConvexShape pentagon;
    pentagon.setPointCount(5);

    pentagon.setPoint(0, sf::Vector2f(a + b + y, x));
    pentagon.setPoint(1, sf::Vector2f(b + y, x + b));
    pentagon.setPoint(2, sf::Vector2f(y, x + (b - a)));
    pentagon.setPoint(3, sf::Vector2f( y, x - (b - a)));
    pentagon.setPoint(4, sf::Vector2f(b + y, x - b));

    return pentagon;
}

void ClaireParam::run(Size &size, std::string saveAs) {
    sf::RenderTexture tex;
    tex.create(size.width, size.height);

    if (this->space->texture != nullptr) {
        this->space->texture->param->run(*this->space->texture->size, this->space->texture->saveAS);

        sf::Texture back;
        back.loadFromImage(this->space->texture->param->image);
        sf::Sprite sprite(back);
        tex.draw(sprite);
    }
    else {
        tex.clear(this->space->color);
    }

    sf::ConvexShape pentagon;

    bool texUP = false;
    sf::Texture textureUP;
    bool texDOWN = false;
    sf::Texture textureDOWN;
    bool texLEFT= false;
    sf::Texture textureLEFT;
    bool texRIGHT = false;
    sf::Texture textureRIGHT;

    if (this->textureUP != nullptr) {
        auto tan = std::tan(45 * 3.14159265 / 180.0);
        this->textureUP->size->height = (tan * this->b) + this->b;
        this->textureUP->size->width = 2 * this->b;
        this->textureUP->param->run(*this->textureUP->size, this->textureUP->saveAS);
        textureUP.loadFromImage(this->textureUP->param->image);
        texUP = true;
    }
    if (this->textureDOWN != nullptr) {
        auto tan = std::tan(45 * 3.14159265 / 180.0);
        this->textureDOWN->size->height = (tan * this->b) + this->b;
        this->textureDOWN->size->width = 2 * this->b;
        this->textureDOWN->param->run(*this->textureDOWN->size, this->textureDOWN->saveAS);
        textureDOWN.loadFromImage(this->textureDOWN->param->image);
        texDOWN = true;
    }
    if (this->textureLEFT != nullptr) {
        auto tan = std::tan(45 * 3.14159265 / 180.0);
        this->textureLEFT->size->height = (tan * this->b) + this->b;
        this->textureLEFT->size->width = 2 * this->b;
        this->textureLEFT->param->run(*this->textureLEFT->size, this->textureLEFT->saveAS);
        textureLEFT.loadFromImage(this->textureLEFT->param->image);
        texLEFT = true;
    }
    if (this->textureRIGHT != nullptr) {
        auto tan = std::tan(45 * 3.14159265 / 180.0);
        this->textureRIGHT->size->height = (tan * this->b) + this->b;
        this->textureRIGHT->size->width = 2 * this->b;
        this->textureRIGHT->param->run(*this->textureRIGHT->size, this->textureRIGHT->saveAS);
        textureRIGHT.loadFromImage(this->textureRIGHT->param->image);
        texRIGHT = true;
    }

    bool decal = true;

    int posY = size.height + this->b;
    while (posY > -size.height) {
        bool inv = false;
        int posX = -3*this->b;
        if (decal) {
            posX = -this->b + this->space->len;
        }
        while (posX < size.width + 2*this->b) {
            if (!inv) {
                pentagon  = createCairoPentagon(this->a, this->b, posX, posY+this->space->len/2);
                if (texUP) {
                    pentagon.setTexture(&textureUP);
                } else {
                    pentagon.setFillColor(this->up);
                }
                tex.draw(pentagon);
                pentagon  = createCairoPentagon(-this->a, -this->b, posX, posY-this->space->len/2);
                if (texDOWN) {
                    pentagon.setTexture(&textureDOWN);
                } else {
                    pentagon.setFillColor(this->down);
                }
                tex.draw(pentagon);
                inv = true;
            }
            else {
                pentagon = createCairoPentagonInv(this->a, this->b, posY, posX  + this->space->len/2);
                if (texLEFT) {
                    pentagon.setTexture(&textureRIGHT);
                } else {
                    pentagon.setFillColor(this->right);
                }
                tex.draw(pentagon);
                pentagon  = createCairoPentagonInv(-this->a, -this->b, posY, posX - this->space->len/2);
                if (texRIGHT) {
                    pentagon.setTexture(&textureLEFT);
                } else {
                    pentagon.setFillColor(this->left);
                }
                tex.draw(pentagon);
                inv = false;
            }
            posX += 2 * this->b + this->space->len;
        }
        decal = !decal;
        posY -= 2* this->b + this->space->len;
    }
    sf::Image image = tex.getTexture().copyToImage();

    this->image = image;
    if (!saveAs.empty()) {
        if (!image.saveToFile(saveAs)) {
            std::cout << "SaveAs Fail with : " + saveAs + '\n';
        }
    }
}

