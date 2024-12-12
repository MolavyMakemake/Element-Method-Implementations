#version 430 core

layout (location = 0) in vec2 aPos;

out vec2 texCoord;

uniform vec2 scale;


void main()
{
	gl_Position = vec4(aPos, 0, 1);
	texCoord = vec2((aPos + 1) / 2);
}