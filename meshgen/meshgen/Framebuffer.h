#pragma once

#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>

#include <string>
#include "Shader.h"


struct Texture {
	unsigned int id;
	std::string type;
	std::string path;
};

class Framebuffer {
public:
	Framebuffer(int width, int height);
	
	void draw();
	void blip();

	void activate();
	void bind();
	void unbind();

	void resize(int width, int height);

	void Delete();

	GLuint texture;

private:
	int width, height;

	GLuint VAO, VBO, framebuffer;
};

const float vertices[] = {
	-1.0f, -1.0f,
	-1.0f,  3.0f,
	 3.0f, -1.0f
};