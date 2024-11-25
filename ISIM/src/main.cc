#include "rapidjson/document.h" 
#include "rapidjson/filereadstream.h" 
#include <iostream>
#include <vector>

#include "utility/jsoncheck.hh"
#include "utility/texture/Texture.hh"

int main(int argc, char *argv[]){
    if (argc != 2) {
        std::cerr << "Nombre d'argument invalide: ./ISIM72 [input].JSON\n";
        return 1;
    }

    FILE* fp = fopen(argv[1], "r");

    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    rapidjson::Document doc;

    doc.ParseStream(is);
    fclose(fp);

    if (!doc.IsArray()) {
        std::cerr << "JSON must be an Array" << std::endl;
        return -1;
    }

    for (const auto& item : doc.GetArray()) {
        if (item.IsObject()) {
            Texture* texture = jsoncheck::init(item);
            texture->param = jsoncheck::param(item, texture->type);
            texture->param->run(*texture->size, texture->saveAS);
            delete texture;
        }
    }
}
