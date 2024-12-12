#version 430 core

const int N = 128;
uniform int R;

in vec2 pos;
out vec4 FragColor;
layout(std140) uniform v_block
{
	vec4 vertices[N];
};

void main()
{
	vec2 v;
	float d = 1000;
	for (int i = 0; i < N; i++) {
		v = vertices[i].xy - pos;
		d = min(d, dot(v, v));

		v = vertices[i].zw - pos;
		d = min(d, dot(v, v));
	}

	d = min(sqrt(d), abs(1 - length(pos)));

	FragColor = vec4(d);
}