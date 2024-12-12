#version 430 core

layout (location = 0) in vec2 aPos;

out vec2 pos;

uniform vec2 scale;


void main()
{
	gl_Position = vec4(aPos, 0, 1);
	//texCoord = aPos;
	pos = vec2(aPos.x * scale.x, aPos.y * scale.y);
}