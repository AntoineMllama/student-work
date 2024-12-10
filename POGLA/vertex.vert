#version 450

in vec3 position;
in vec3 normalSmooth;

out vec3 v_color;
out vec3 v_normal;


vec3 light_color = vec3(1.0, 1.0, 1.0);
vec3 light_position = vec3(-10.0, -10.0, -50.0);

vec3 object_color = vec3(0.0, 1.0, 1.0);

mat4 model_view_matrix = mat4(
			      0.57735, -0.33333, 0.57735, 0.00000,
			      0.00000, 0.66667, 0.57735, 0.00000,
			      -0.57735, -0.33333, 0.57735, 0.00000,
			      0.00000, 0.00000, -17, 1.00000);
mat4 projection_matrix = mat4(
			      15.00000, 0.00000, 0.00000, 0.00000,
			      0.00000, 15.00000, 0.00000, 0.00000,
			      0.00000, 0.00000, -1.00020, -1.00000,
			      0.00000, 0.00000, -10.00100, 0.00000);

void main() {

  gl_Position = projection_matrix * model_view_matrix * vec4(position, 1.0);

  float c = clamp(dot(normalize(position - light_position), normalSmooth), 0.0, 1.0);
  v_color = object_color * c;
  v_normal = normalSmooth;

}
