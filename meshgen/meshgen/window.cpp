#include "Window.h"

#include<iostream>

#include<glad/glad.h>
#include<GLFW/glfw3.h>

#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>

#include <vector>
#include "Shader.h"
#include "Framebuffer.h"
#include "ComputeShader.h"


Window::Window(int width, int height, const char* title, int fps) {

    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    // Create window
    glfwWindow = glfwCreateWindow(width, height, title, NULL, NULL);
    if (glfwWindow == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(glfwWindow);

    // Input events 
    //glfwSetKeyCallback(glfwWindow, Input::onKeyAction);
    //glfwSetCursorPosCallback(glfwWindow, Input::onMouseMove);
    //glfwSetMouseButtonCallback(glfwWindow, Input::onMouseAction);
    //glfwSetScrollCallback(glfwWindow, Input::onScroll);

    // Init glad
    gladLoadGL();

    glViewport(0, 0, width, height);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(glfwWindow, true);
    ImGui_ImplOpenGL3_Init("#version 430");

    size.x = width;
    size.y = height;
}

void distribute_euc(int N, float R, glm::vec2* vertices) {
    const float arg = glm::pi<float>() * (3 - glm::sqrt(5));

    for (int i = 0; i < N; i++) {
        float t = arg * i;
        float r = glm::sqrt((float)i / N) * R;

        vertices[i] = glm::vec2(r * glm::cos(t), r * glm::sin(t));
    }
}

void distribute_hyp(int N, float R, glm::vec2* vertices) {
    const float arg = glm::pi<float>() * (3 - glm::sqrt(5));

    for (int i = 0; i < N; i++) {
        float t = arg * i;
        float x = glm::sqrt((float)i / N) * glm::sinh(R / 2);
        float r = x / glm::sqrt(1 + x * x);

        vertices[i] = glm::vec2(r * glm::cos(t), r * glm::sin(t));
    }
}

void Window::Run() {
    Shader shader("shader.vert", "shader.frag");
    Shader voronoi("voronoi.vert", "voronoi_euc.frag");

    Framebuffer framebuffer(size.x, size.y);
    glm::vec2 scale((float)size.x / size.y, 1);

    bool redraw = true;

    const int N = 256;
    const float R = 1.0f;
    glm::vec2 vertices[N];
    distribute_euc(N, R, vertices);

    for (auto v : vertices) {
        std::cout << v.x << ", " << v.y << std::endl;
    }

    GLuint UBO;
    glGenBuffers(1, &UBO);
    glBindBuffer(GL_UNIFORM_BUFFER, UBO);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, UBO);
    glUniformBlockBinding(voronoi.ID, voronoi.Loc("v_block"), 0);

    glDisable(GL_DEPTH_TEST);
    while (!glfwWindowShouldClose(glfwWindow)) {

        glm::ivec2 _size{};
        glfwGetWindowSize(glfwWindow, &_size.x, &_size.y);
        if (size != _size) {
            size = _size;
            glViewport(0, 0, size.x, size.y);

            framebuffer.resize(size.x, size.y);

            scale.x = glm::max((float)size.x / size.y, 1.f);
            scale.y = glm::max((float)size.y / size.x, 1.f);
        }

        if (redraw) {
            framebuffer.bind();
            voronoi.Activate();
            glUniform2f(voronoi.Loc("scale"), scale.x, scale.y);
            glUniform1f(voronoi.Loc("R"), R);
            framebuffer.draw();
            framebuffer.unbind();
        }

        // Render
        // ------

        shader.Activate();
        glUniform2f(shader.Loc("scale"), scale.x, scale.y);
        framebuffer.blip();

        // GUI
        // ---

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Settings");
        redraw = ImGui::Button("Render");
        //ImGui::SliderInt("Iterations per frame", &iterationsPerFrame, 1, 100);
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(glfwWindow);
        glfwPollEvents();
    }
}