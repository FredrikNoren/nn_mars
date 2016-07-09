#version 150

in vec2 vertex_texcoord;
out vec4 out_color;
uniform sampler2D normalmap;
uniform float min_value;
uniform float max_value;
uniform vec4 ambient;
uniform vec4 diffuse;

void main() {
  //out_color = texture(normalmap, vertex_texcoord);
  vec3 normal = texture(normalmap, vertex_texcoord).xyz*2 - 1;
  vec3 light_direction = normalize(vec3(1, 1, 0.8));
  float light = clamp(dot(light_direction, normal), 0, 1);
  out_color = vec4(mix(ambient, diffuse, light).rgb, 1);
}
