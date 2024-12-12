#version 430 core

const int N = 128;

uniform float R;

in vec2 pos;
out vec4 FragColor;
layout(std140) uniform v_block
{
	vec4 vertices[N];
};

void main()
{
	if (dot(pos, pos) >= 1) {
		FragColor = vec4(0);
		return;
	}

	float s = 2 / (1 - dot(pos, pos));

	vec2 v;
	float d = 1000;
	for (int i = 0; i < N; i++) {
		v = vertices[i].xy;
		d = min(d, s * dot(v - pos, v - pos) / (1 - dot(v, v)));

		v = vertices[i].zw;
		d = min(d, s * dot(v - pos, v - pos) / (1 - dot(v, v)));
	}

	d = acosh(1 + d);
	d = min(d, abs(R - 2 * atanh(length(pos))));

	FragColor = vec4(d);
}