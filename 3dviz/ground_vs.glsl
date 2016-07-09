#version 150

in vec3 position;
uniform mat4 world_transform;
uniform mat4 view_projection;
in vec2 texcoord;

out vec2 vertex_texcoord;
uniform sampler2D heightmap;
out vec3 vertex_world_pos;
out float vertex_eye_z;
uniform float min_value;
uniform float max_value;

void main() {
  vertex_texcoord = vec2(texcoord.x, 1 - texcoord.y);
  float raw_height = texture(heightmap, vertex_texcoord).r;
  if (raw_height < min_value) raw_height = min_value;
  float height = 5 * (raw_height - min_value) / (max_value - min_value);
  vertex_world_pos = (world_transform * vec4(position.x, position.y, height, 1.0)).xyz;
  vec4 pos = view_projection * vec4(vertex_world_pos, 1.0);
  vertex_eye_z = pos.z;

  gl_Position = pos;
}
