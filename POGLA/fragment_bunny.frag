#version 450

in vec3 v_color;

layout(location=0) out vec4 output_color;

void main() {
  output_color = vec4(v_color, 1.0);
}
