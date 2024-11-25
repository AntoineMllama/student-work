#pragma once

#include "rapidjson/document.h"
#include "texture/Texture.hh"

namespace jsoncheck {

    Texture* init(const rapidjson::Value& doc);
    Param* param(const rapidjson::Value& doc, std::string type);
}