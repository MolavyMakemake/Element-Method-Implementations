#pragma once
#include <glad/glad.h>

class ComputeShader {
public:
	GLuint ID;

	ComputeShader(const char* computePath);
	void Activate();

	GLuint Loc(const char* name);
};