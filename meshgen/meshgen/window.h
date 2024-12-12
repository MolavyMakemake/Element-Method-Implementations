#pragma once

#include"imgui.h"
#include"imgui_impl_glfw.h"
#include"imgui_impl_opengl3.h"
#include <glm/glm.hpp>

class Window {
public:
	Window(int width, int height, const char* title, int fps = 60);
	void Run();

	glm::ivec2 size;

private:
	GLFWwindow* glfwWindow = nullptr;
};