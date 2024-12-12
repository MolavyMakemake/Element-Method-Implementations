#include "ComputeShader.h"
#include "Shader.h"


ComputeShader::ComputeShader(const char* computeFile) {
	std::string computeCode = get_file_contents(computeFile);

	const char* computeSource = computeCode.c_str();


	GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(computeShader, 1, &computeSource, NULL);
	glCompileShader(computeShader);


	ID = glCreateProgram();
	glAttachShader(ID, computeShader);
	glLinkProgram(ID);

	glDeleteShader(computeShader);
}

void ComputeShader::Activate() {
	glUseProgram(ID);
}

GLuint ComputeShader::Loc(const char* name) {
	return glGetUniformLocation(ID, name);
}