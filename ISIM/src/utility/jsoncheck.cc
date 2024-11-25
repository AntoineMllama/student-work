#include "jsoncheck.hh"
#include <iostream>
#include "texture/GradientParam.hh"
#include "texture/BrickParam.hh"
#include "texture/PerlinParam.hh"
#include "texture/Size.hh"
#include <string>
#include <SFML/Graphics.hpp>
#include <texture/VoronoiParam.hh>

#include "texture/ClaireParam.hh"


/// Zone Message Erreur
void jsonerr(std::string fields, std::string type) {
    std::cerr << fields << " not found or not " << type << '\n';
    exit(-1);
}


void colorerr(std::string name) {
    std::cerr << "Field: " << name << ", is not valid. Expected hexacode (#xxxxxx)\n";
    exit(-2);
}
///


/// Zone Couleur
bool hexacheck(const std::string& s) {
    for (char c : s) {
        if (!std::isxdigit(c)) {
            return true;
        }
    }
    return false;
}


sf::Color string_to_color(std::string str) {
    if (str.size() != 7 && str.size() != 9) {colorerr("color");}
    std::string str_r = str.substr(1,2);
    std::string str_g = str.substr(3,2);
    std::string str_b = str.substr(5,2);
    std::string str_a = "FF";

    if (str.size() == 9) {
        str_a = str.substr(7,2);
    }

    if (hexacheck(str_r) || hexacheck(str_g) || hexacheck(str_b) || hexacheck(str_a)) {
        colorerr("color");
    }

    sf::Uint8 r = std::stoi(str_r, 0, 16);
    sf::Uint8 g = std::stoi(str_g, 0, 16);
    sf::Uint8 b = std::stoi(str_b, 0, 16);
    sf::Uint8 a = std::stoi(str_a, 0, 16);

    return sf::Color(r, g, b, a);
}
///

/// Zone Parseur
Size* parseSize(const rapidjson::Value& dim) {
    if (!(dim.HasMember("width") && dim["width"].IsInt())) {
        jsonerr("width", "an Int");
    }
    if (!(dim.HasMember("height") && dim["height"].IsInt())) {
        jsonerr("height", "an Int");
    }
    int height = dim["height"].GetInt();
    int width = dim["width"].GetInt();
    return new Size(width, height);
}

Space* parseSpace(const rapidjson::Value& space) {
    sf::Color space_color = sf::Color::Transparent;
    if (space.HasMember("color") && space["color"].IsString()) {
        space_color = string_to_color(space["color"].GetString());
    }
    if (!(space.HasMember("len") && space["len"].IsInt())) {
        jsonerr("len", "an Int");
    }
    int len = space["len"].GetInt();

    Texture* tex = nullptr;
    if (space.HasMember("parametre") && space["parametre"].IsObject()) {
        const rapidjson::Value& paramsBis = space["parametre"];
        tex = jsoncheck::init(paramsBis);
        tex->param = jsoncheck::param(paramsBis, tex->type);
    }
    return new Space(space_color, len, tex);
}


Param* gradientInit(const rapidjson::Value& params) {
    if (!(params.HasMember("color") && params["color"].IsArray())) {
            jsonerr("color", "an Array");
    }
    if (!(params.HasMember("angle") && params["angle"].IsInt())) {
            jsonerr("angle", "an Int");
    }

    std::vector<sf::Color> arrayColor;
    const rapidjson::Value& colorArray = params["color"];
    for (auto i = 0; i < colorArray.Size(); ++i) {
        if (colorArray[i].IsString()) {
            arrayColor.push_back(string_to_color(colorArray[i].GetString()));
        }
    }
    int angle = params["angle"].GetInt();
    return new GradientParam(arrayColor, angle);
}


Param* brickInit(const rapidjson::Value& params) {
    if (!(params.HasMember("brick") && params["brick"].IsObject())) {
            jsonerr("brick", "an Object");
    }
    const rapidjson::Value& brick = params["brick"];

    if (!(params.HasMember("space") && params["space"].IsObject())) {
        jsonerr("space", "an Object");
    }
    const rapidjson::Value& space_param = params["space"];

    Space* space = parseSpace(space_param);

    if (!(brick.HasMember("size") && brick["size"].IsObject())) {
        jsonerr("size", "an Object");
    }
    const rapidjson::Value& dim = brick["size"];
    Size* size = parseSize(dim);

    if (brick.HasMember("parametre") && brick["parametre"].IsObject()) {
        const rapidjson::Value& paramsBis = brick["parametre"];
        Texture* newtexture = jsoncheck::init(paramsBis);
        newtexture->param = jsoncheck::param(paramsBis, newtexture->type);
        return new BrickParam(size, space, newtexture);
    }

    //else

    sf::Color color = sf::Color::Transparent;
    if (brick.HasMember("color") && brick["color"].IsString()) {
        color = string_to_color(brick["color"].GetString());
    }
    return new BrickParam(color, size, space, nullptr);
}


Param* claireInit(const rapidjson::Value& params) {
    if (!(params.HasMember("a") && params["a"].IsInt())) {
        jsonerr("a", "an Int");
    }
    if (!(params.HasMember("b") && params["b"].IsInt())) {
        jsonerr("b", "an Int");
    }
    int a = params["a"].GetInt();
    int b = params["b"].GetInt();

    if (!(params.HasMember("space") && params["space"].IsObject())) {
        jsonerr("space", "an Object");
    }
    const rapidjson::Value& space_param = params["space"];
    Space* space = parseSpace(space_param);

    sf::Color up;
    sf::Color down;
    sf::Color left;
    sf::Color right;
    Texture* textureUP = nullptr;
    Texture* textureDOWN = nullptr;
    Texture* textureLEFT = nullptr;
    Texture* textureRIGHT = nullptr;
    if (params.HasMember("up")) {
        if (params["up"].IsString()) {
            up = string_to_color(params["up"].GetString());
        }
        else if (params["up"].IsObject()) {
            const rapidjson::Value& paramsBisUP = params["up"];
            textureUP = jsoncheck::init(paramsBisUP);
            textureUP->param = jsoncheck::param(paramsBisUP, textureUP->type);
        }
    }
    if (params.HasMember("down")) {
        if (params["down"].IsString()) {
            down = string_to_color(params["down"].GetString());
        }
        else if (params["down"].IsObject()) {
            const rapidjson::Value& paramsBisDOWN = params["down"];
            textureDOWN = jsoncheck::init(paramsBisDOWN);
            textureDOWN->param = jsoncheck::param(paramsBisDOWN, textureDOWN->type);
        }
    }
    if (params.HasMember("left")) {
        if (params["left"].IsString()) {
            left = string_to_color(params["left"].GetString());
        }
        else if (params["left"].IsObject()) {
            const rapidjson::Value& paramsBisLEFT = params["left"];
            textureLEFT = jsoncheck::init(paramsBisLEFT);
            textureLEFT->param = jsoncheck::param(paramsBisLEFT, textureLEFT->type);
        }
    }
    if (params.HasMember("right")) {
        if (params["right"].IsString()) {
            right = string_to_color(params["right"].GetString());
        }
        else if (params["right"].IsObject()) {
            const rapidjson::Value& paramsBisRIGHT = params["right"];
            textureRIGHT = jsoncheck::init(paramsBisRIGHT);
            textureRIGHT->param = jsoncheck::param(paramsBisRIGHT, textureRIGHT->type);
        }
    }
    return new ClaireParam(a, b, up, textureUP, down, textureDOWN, left, textureLEFT, right, textureRIGHT, space);
}

Param* voronoiInit(const rapidjson::Value& params) {
    return new VoronoiParam();
}

Param* perlininit(const rapidjson::Value& params, std::string type) {
    auto perlin_param = new PerlinParam(type);
    if (type == "bois3") {
        if (!(params.HasMember("inf") && params["inf"].IsString())) {
            jsonerr("inf", "a String");
        }
        if (!(params.HasMember("mid") && params["mid"].IsString())) {
            jsonerr("mid", "a String");
        }
        if (!(params.HasMember("up") && params["up"].IsString())) {
            jsonerr("up", "a String");
        }

        perlin_param->bois3_inf = string_to_color(params["inf"].GetString());
        perlin_param->bois3_mid = string_to_color(params["mid"].GetString());
        perlin_param->bois3_up = string_to_color(params["up"].GetString());

        if (!(params.HasMember("borne_up") && params["borne_up"].IsDouble())) {
            jsonerr("borne_up", "a Double (float)");
        }
        if (!(params.HasMember("borne_inf") && params["borne_inf"].IsDouble())) {
            jsonerr("borne_inf", "a Double (float)");
        }
        double up = params["borne_up"].GetDouble();
        double inf = params["borne_inf"].GetDouble();

        if (up < inf) {
            const auto tmp = inf;
            inf = up;
            up = tmp;
        }

        perlin_param->borne_inf = inf;
        perlin_param->borne_sup = up;
    }

    return perlin_param;
}
///



namespace jsoncheck {


    Texture* init(const rapidjson::Value& doc) {
        if (!(doc.HasMember("size") && doc["size"].IsObject())) {
            jsonerr("size", "an Object");
        }
        const rapidjson::Value& dim = doc["size"];
        Size* size = parseSize(dim);
        if (!(doc.HasMember("type") && doc["type"].IsString())) {
            jsonerr("type", "a String");
        }
        std::string type = doc["type"].GetString();
        std::string saveAS;
        if (doc.HasMember("saveAs") && doc["saveAs"].IsString()) {
            saveAS = doc["saveAs"].GetString();
        }
        return new Texture(size, type, saveAS);
    }


    Param* param(const rapidjson::Value& doc, const std::string type) {

        if (!(doc.HasMember("parametre") && doc["parametre"].IsObject())) {
            jsonerr("parametre", "an Object");
        }
        const rapidjson::Value& params = doc["parametre"];

        Param* param;
        if (type == "gradient"){
            param = gradientInit(params);
        }
        else if (type == "brick"){
            param = brickInit(params);
        }
        else if (type == "claire") {
            param = claireInit(params);
        }
        else if (type == "voronoi") {
            param = voronoiInit(params);
        }
        else {
            param = perlininit(params, type);
        }
        return param;
    }
}