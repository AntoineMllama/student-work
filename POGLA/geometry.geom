#version 450

layout (points) in;
layout (triangle_strip, max_vertices = 256) out; // max de ma machine de travail

in vec3 v_color[];
out vec3 g_color;

const float PI = 3.14;

uniform float anim_time;


float random(float seed) {
    return fract(sin(seed) * 42.0*72.0);
}


float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}


vec4 tige(vec4 pos, float height, float weight, int substep, vec4 wind, vec3 color) {

    float step = height / substep;

    //pos.w += -2.0;

    vec4 prev2 = vec4(0.0);
    vec4 prev4 = vec4(0.0);

    for (int i = substep; i > 0; i--) {
        g_color = color; // #2A7025

        float factor = float(substep - i) / float(substep);
        vec4 windOffset = wind * factor;

        gl_Position = pos;
        EmitVertex();

        prev2 = pos + vec4(0.0, step, 0.0, 0.0) + windOffset;
        gl_Position = prev2;
        EmitVertex();

        if (i != substep)
            gl_Position = prev4;
        else
            gl_Position = pos + vec4(weight, 0.0, 0.0, 0.0) + windOffset;
        EmitVertex();

        gl_Position = pos + vec4(weight,  step, 0.0, 0.0) + windOffset;
        prev4 = gl_Position;
        EmitVertex();
        EndPrimitive();

        pos += vec4(0.0, step, 0.0, 0.0) + windOffset;
    }

    return ((prev4 - prev2) / 2) + prev2;
}


void circle(vec4 center, int segments, float radius) {
    g_color = vec3(0.937, 0.752, 0.0);

    gl_Position = center;
    EmitVertex();

    for (int i = 0; i <= segments; i++) {
        float angle = 2.0 * PI * float(i) / float(segments);
        vec3 offset = vec3(cos(angle), sin(angle), 0.0) * radius;

        gl_Position = center + vec4(offset, 0.1);
        EmitVertex();

        gl_Position = center;
        EmitVertex();
    }
    EndPrimitive();
}


void petals(vec4 center, int nbr_petal, float radius) {
    float offset_angle_petal = 0.209;

    for (int i = 0; i <= nbr_petal; i++) {
        float angle = 2.0 * PI * float(i) / float(nbr_petal);
        vec3 offset_plus = vec3(cos(angle + offset_angle_petal), sin(angle + offset_angle_petal), 0.0) * radius;
        vec3 offset_moins = vec3(cos(angle - offset_angle_petal), sin(angle - offset_angle_petal), 0.0) * radius;

        g_color = vec3(1.0, 0.980, 0.941);

        gl_Position = center;
        EmitVertex();

        gl_Position = center + vec4(offset_plus, 0.1);
        EmitVertex();

        gl_Position = center + vec4(offset_moins, 0.1);
        EmitVertex();

        EndPrimitive();
    }
}

void flower(vec4 wind, float height, float weight) {
    vec4 center = tige(gl_in[0].gl_Position, height, weight, 4, wind, vec3(0.164, 0.539, 0.145));
    circle(center, 8, 0.4);
    petals(center, 7, 0.8);
}

void grass(vec4 wind) {
    for (float i = 0.0; i < 10.0; i++) {
        float angle = rand(vec2(i, 0.1)) * 2.0 * PI;
        float radius = rand(vec2(i, 0.2)) * 0.5;
        vec4 basePosition = gl_in[0].gl_Position + vec4(cos(angle) * radius, sin(angle) * radius, 0.0, 0.0);


        float randomHeight = 1.5 + rand(vec2(i, 0.3)) * 1.0;
        float randomWidth = 0.05 + rand(vec2(i, 0.4)) * 0.1;

        tige(basePosition, randomHeight, randomWidth, 5, wind, vec3(0.164, 0.439, 0.145));
    }
}


void main() {
    float sin_x = sin(2 * PI * anim_time * 0.01) * sin(2 * PI * anim_time * 0.01)  * 0.5;
    vec4 wind = vec4(sin_x, 0.0, 0.0, 0.0);

    grass(wind);

    float r = rand(gl_in[0].gl_Position.xy);
    if (r > 0.85)
            flower(wind, 3.0, 0.2);

}