#version 430 core

uniform sampler2D img;
uniform vec2 scale;

in vec2 texCoord;
out vec4 FragColor;

void main()
{
	vec2 tex_offset = 2.0 / textureSize(img, 0);
	
	float u_px = texture(img, texCoord + vec2(tex_offset.x, 0)).r;
	float u_py = texture(img, texCoord + vec2(0, tex_offset.y)).r;
	float u_nx = texture(img, texCoord - vec2(tex_offset.x, 0)).r;
	float u_ny = texture(img, texCoord - vec2(0, tex_offset.y)).r;
	
	vec2 h = 2 * tex_offset * scale;
	vec3 dx = vec3(h.x, 0, u_px - u_nx);
	vec3 dy = vec3(0, h.y, u_py - u_ny);
	vec3 n = normalize(cross(dx, dy));
	
	FragColor = vec4(n, 1);
	//FragColor = texture(img, texCoord);
}