#pragma once

#include<glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include<string>
#include<vector>

#include"Shader.h"


class Mesh {
public:
	std::vector<float> vertices;
	std::vector<GLuint> indices;

	Mesh() {}
	Mesh(
		const std::vector<float>& vertices,
		const std::vector<GLuint>& indices);

	inline void Update(std::vector<double>& vertices, std::vector<size_t>& indices) {
		this->vertices.clear();
		this->indices.clear();

		for (size_t t : indices)
			this->indices.push_back(t);

		for (double v : vertices)
			this->vertices.push_back(v);

		Delete();
		Generate();
	}

	void Generate();
	void Delete();

	void Draw();

private:
	GLuint VAO, VBO, EBO;
};