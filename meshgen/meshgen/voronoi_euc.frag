#version 430 core

const int N = 512;
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

	int j = 0;
	bool isMinimum;
	float _d;
	for (int i = 0; i < N; i++) {
		v = vertices[i].xy - pos;
		_d = dot(v, v);
		isMinimum = _d < d;
		d = isMinimum ? _d : d;
		j = isMinimum ? 2 * i : j;

		v = vertices[i].zw - pos;
		_d = dot(v, v);
		isMinimum = _d < d;
		d = isMinimum ? _d : d;
		j = isMinimum ? 2 * i + 1 : j;
	}

	//d = min(sqrt(d), abs(1 - length(pos)));
	d = sqrt(d);

	FragColor = vec4(d, j, 0, 0);
}