#include "VoronoiParam.hh"

#include <iostream>
#include <unistd.h>
#include <SFML/Graphics.hpp>
#include <experimental/random>

int last_x;
int last_y;
bool first_ligne;

class CoordPoly {
public:
    CoordPoly()= default;
    CoordPoly(int x, int y): x(x), y(y){}
    int x;
    int y;
};

bool compareByX(const CoordPoly& a, const CoordPoly& b) {
    return a.x < b.x;
}

std::vector<std::vector<CoordPoly>> prev_ligne;
std::vector<CoordPoly> prevCoords;

VoronoiParam::VoronoiParam() {
}


sf::ConvexShape randomPolygon(int centerX, int centerY, int minRadius, int maxRadius, int must_x, int must_y) {
    sf::ConvexShape polygon;
    int nbrPoint = std::experimental::randint(3, 6);
    polygon.setPointCount(nbrPoint);

    float angleIcrem = 2 * M_PI / nbrPoint;

    std::vector<CoordPoly> coord_polys;

    //Calcul des points
    for (float i = 0; i < nbrPoint; ++i) {
        float angle = i * angleIcrem + std::experimental::randint(-15, 15) * M_PI / 180;
        int radius = std::experimental::randint(minRadius, maxRadius);
        int x = centerX + radius * std::cos(angle);
        int y = centerY + radius * std::sin(angle);
        coord_polys.push_back(CoordPoly(x, y));
        angle += angleIcrem;
    }

    //Jointure des points must_xy
    auto maxX = std::max_element(coord_polys.begin(), coord_polys.end(), [](const CoordPoly& a, const CoordPoly& b) {
        return a.x < b.x;
    });

    if (!(must_x <= 0 || must_y < 0)) {
        auto closest_point = std::min_element(coord_polys.begin(), coord_polys.end(), [must_x, must_y](const CoordPoly& a, const CoordPoly& b) {
            float distA = std::sqrt((a.x - must_x) * (a.x - must_x) + (a.y - must_y) * (a.y - must_y));
            float distB = std::sqrt((b.x - must_x) * (b.x - must_x) + (b.y - must_y) * (b.y - must_y));
            return distA < distB;
        });

        closest_point->x = must_x;
        closest_point->y = must_y;
    }

    //Jointure des points les plus proches sur la meme ligne afin de cole les deux polygon
    float min_dist = MAXFLOAT;
    CoordPoly close_prev;
    CoordPoly close_actu;

    for (CoordPoly prev_coord : prevCoords) {
        for (CoordPoly actu_coord : coord_polys) {
            if ((prev_coord.x == must_x && prev_coord.y == must_y) || (actu_coord.x == must_x && actu_coord.y == must_y)) {continue;}
            float dist = std::sqrt((prev_coord.x - actu_coord.x) * (prev_coord.x - actu_coord.x) + (prev_coord.y - actu_coord.y) * (prev_coord.y - actu_coord.y));
            if (dist < min_dist) {
                min_dist = dist;
                close_prev = prev_coord;
                close_actu = actu_coord;
            }
        }
    }

    //Idem mais pour la collonne
    CoordPoly close_prev_col;
    CoordPoly close_actu_col;
    if (!first_ligne) {

        std::vector<CoordPoly> polygonne_inferieur;
        //Recheche du polygon inferieur
        for (std::vector<CoordPoly> infPoly : prev_ligne) {
            if (centerX <= infPoly.back().x) {
                polygonne_inferieur = infPoly;
                break;
            }
        }

        min_dist = MAXFLOAT;
        for (CoordPoly prev_coord : polygonne_inferieur) {
            for (CoordPoly actu_coord : coord_polys) {
                if ((prev_coord.x == must_x && prev_coord.y == must_y) || (actu_coord.x == must_x && actu_coord.y == must_y)
                    ||( prev_coord.x == close_prev.x &&  prev_coord.y == close_prev.y) || (actu_coord.x == close_prev.x && actu_coord.y == close_prev.y)) {continue;}
                float dist = std::sqrt((prev_coord.x - actu_coord.x) * (prev_coord.x - actu_coord.x) + (prev_coord.y - actu_coord.y) * (prev_coord.y - actu_coord.y));
                if (dist < min_dist) {
                    min_dist = dist;
                    close_prev_col = prev_coord;
                    close_actu_col = actu_coord;
                }
            }
        }
    }

    last_x = maxX->x;
    last_y = maxX->y;

    prevCoords.clear();

    //Creation final du polygon
    for (int i = 0; i < nbrPoint; ++i) {
        bool change = false;
        int x = coord_polys.at(i).x;
        int y = coord_polys.at(i).y;

        if (x == close_actu.x && y == close_actu.y) {
            x = close_prev.x;
            y = close_prev.y;
            change = true;
        }
        if (!change && x == close_actu_col.x && y == close_actu_col.y) {
            x = close_prev_col.x;
            y = close_prev_col.y;
            change = true;
        }
        if (!change && !first_ligne) {
            //x -= std::abs(close_actu_col.x - close_prev_col.x);
            //y += std::abs(close_actu_col.y - close_prev_col.y);
        }
        polygon.setPoint(i, sf::Vector2f(x, y));
        prevCoords.push_back(CoordPoly(x, y));
    }


    //Ajout du polygon dans la ligne memoire
    std::sort(prevCoords.begin(), prevCoords.end(), compareByX);
    prev_ligne.push_back(prevCoords);


    return polygon;
}


void VoronoiParam::run(Size &size, std::string saveAs) {

    sf::RenderTexture tex;
    sf::ConvexShape polygon;
    tex.create(size.width, size.height);

    std::vector<std::vector<CoordPoly>> ligne_buffer;

    last_x = 0 -size.width * 0.2;

    int minRadius = 100;
    int maxRadius = 150;
    first_ligne = true;

    int pos_Y = size.height;

    prevCoords.clear();

    int count = 0;

while (pos_Y >= 0) {
    while (last_x < size.width * 1.2) {
        polygon = randomPolygon(last_x + 100, pos_Y, minRadius, maxRadius, last_x, last_y);
        sf::Uint8 r = std::experimental::randint(0, 255);
        sf::Uint8 g = std::experimental::randint(0, 255);
        sf::Uint8 b = std::experimental::randint(0, 255);
        polygon.setFillColor(sf::Color(r,g,b));
        tex.draw(polygon);
        tex.display();
        std::sort(prevCoords.begin(), prevCoords.end(), compareByX);
        ligne_buffer.push_back(prevCoords);
    }

    //trouver le y max;
    for (std::vector<CoordPoly> polygon_prev_ligne : prev_ligne) {
        for (CoordPoly point : polygon_prev_ligne) {
            if (point.y < pos_Y) {
                pos_Y = point.y;
            }
        }
    }
    count++;
    last_y = pos_Y;
    pos_Y -= maxRadius;
    if (count >= 2) {
        pos_Y = -1;
    }
    first_ligne = false;
    last_x = 0 -size.width * 0.2;
    prevCoords.clear();
    // prev_ligne.clear();
    // for (std::vector<CoordPoly> buffer : ligne_buffer) {
    //     prev_ligne.push_back(buffer);
    // }
}

    this->image = tex.getTexture().copyToImage();

    if (!saveAs.empty()) {
        if (!image.saveToFile(saveAs)) {
            std::cout << "SaveAs Fail with : " + saveAs + '\n';
        }
    }
    std::cout << "Done";
}


/*
 * [*]Stocker les coord pour chaque poly dans un vec de vec
 * [*]par diplothomie chercher le polygone inferieur (les coords seront a trier par ordre des x
 * [*]Faire une liaison must have avec l'inferieur
 * [*]chercher le point le plus proche avec l'inferieur
 * [*]Changer les cords pour coller avec l'inf
 * [ ]retirer a tout les point du nouveau poly la distance x et y avec le point le plus proche pour ajuster la taille du poly
 */



