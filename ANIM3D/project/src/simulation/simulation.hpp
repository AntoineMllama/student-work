#pragma once

#include "cgp/cgp.hpp"

struct particle_structure
{
    static cgp::numarray<cgp::vec3> norm_spawn;
    static cgp::numarray<cgp::vec3> points_spawn;

    static cgp::numarray<cgp::vec3> norm_elevator;
    static cgp::numarray<cgp::vec3> points_elevator;

    static cgp::numarray<cgp::vec3> norm_out;
    static cgp::numarray<cgp::vec3> points_out;

    static cgp::numarray<cgp::vec3> cylindre_up;
    static cgp::numarray<cgp::vec3> cylindre_down;
    static float cylindre_centre_radius;
    static float cylindre_all_radius;

    struct status
    {
        enum particle_location
        {
            SPAWN,
            ELEVATOR,
            ELEVATOR_OUT,
            CHUTE,
            END,
        };

        enum particle_basket
        {
            BASKET0,
            BASKET1,
            BASKET2,
            BASKET3,
            BASKET4,
        };

        particle_location pl;
        particle_basket pb;
    };

    cgp::vec3 p; // Position
    cgp::vec3 v; // Speed

    cgp::vec3 c; // Color
    float r;     // Radius
    float m;     // mass

    status status;
};


void simulate(std::vector<particle_structure>& particles, float dt);
