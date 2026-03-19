
#version 450

layout(location = 0) in vec2 position;
layout(push_constant) uniform Zoom {
    float zoom;
} zoom;
layout(location = 0) out vec2 tex_coords;

float standart_zoom = 1.8;
float step = 0.1;
float slowdown = 0.1;

float zooming() {
    float delta = step * zoom.zoom / (1.0 + abs(zoom.zoom) * slowdown);
    return standart_zoom * (1.0 + delta);
}

void main() {
  gl_Position = vec4(position, 0.0, 1.0);

  tex_coords = position / zooming() + 0.5;
}
