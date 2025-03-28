#include "Window.h"

#include <iostream>
#include <iomanip>

#include<glad/glad.h>
#include<GLFW/glfw3.h>

#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>

#include <vector>
#include "Shader.h"
#include "Framebuffer.h"
#include "ComputeShader.h"

#include "generate.h"
#include "Mesh.h"

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

void Window::Run() {
    Shader shader("shader.vert", "shader.frag");
    Shader voronoi("voronoi.vert", "voronoi_euc.frag");

    Framebuffer framebuffer(size.x, size.y);
    glm::vec2 scale((float)size.y / size.x, 1);

    bool redraw = true;
    bool redist = true;

    int N = 256;
    int N_bdry = 130;
    int N_iterations = 0;
    int integral_resolution = 10;

    float R = 3.f;
    bool hyperbolic = true;

    triangulation_t triangulation;
    analytics_t analytics;

    Mesh mesh;
    Shader mesh_shader("wireframe.vert", "wireframe.frag");

    bool auto_distribute = false;

    glDisable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    while (!glfwWindowShouldClose(glfwWindow)) {

        glm::ivec2 _size{};
        glfwGetWindowSize(glfwWindow, &_size.x, &_size.y);
        if (size != _size) {
            size = _size;
            glViewport(0, 0, size.x, size.y);

            framebuffer.resize(size.x, size.y);

            scale.x = glm::min((float)size.y / size.x, 1.f);
            scale.y = glm::min((float)size.x / size.y, 1.f);
        }

        if (redist) {
            N = std::max<int>(N, 8);
            N_bdry = std::min<int>(N * N, N_bdry);
            R = std::max<double>(1e-1, R);
            if (hyperbolic) {
                triangulation = square(N, N_bdry, R, METRIC_POINCARE, N_iterations, integral_resolution);
                analytics = analytics_hyp(triangulation);
            }
            else {
                triangulation = square(N, N_bdry, R, METRIC_EUCLIDIAN, N_iterations, integral_resolution);
                analytics = analytics_euc(triangulation);
            }

            //for (int i = 0; i < N; i++)
            //    vertices[i] = glm::vec2(triangulation.vertices[2 * i], triangulation.vertices[2 * i + 1]);

            mesh.Update(triangulation.vertices, triangulation.triangles);

            redraw = true;
            redist = false;
        }
        //if (redraw) {
        //    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(vertices), vertices);
        //
        //    framebuffer.bind();
        //    voronoi.Activate();
        //    glUniform2f(voronoi.Loc("scale"), scale.x, scale.y);
        //    glUniform1f(voronoi.Loc("R"), R);
        //    framebuffer.draw();
        //    framebuffer.unbind();
        //
        //    redraw = false;
        //}

        // Render
        // ------

        //shader.Activate();
        //glUniform2f(shader.Loc("scale"), scale.x, scale.y);
        //framebuffer.blip();

        glClear(GL_COLOR_BUFFER_BIT);
        mesh_shader.Activate();
        glUniform2f(mesh_shader.Loc("scale"), scale.x, scale.y);
        mesh.Draw();

        // GUI
        // ---

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Settings");
        redraw = ImGui::Button("Render");

        bool _redist = false;
        _redist |= ImGui::Checkbox("Auto distribute", &auto_distribute);
        redist |= ImGui::Button("Distribute");

        _redist |= ImGui::InputInt("# points", &N);
        _redist |= ImGui::InputInt("# boundary points", &N_bdry);
        _redist |= ImGui::SliderInt("# iterations", &N_iterations, 0, 100);
        _redist |= ImGui::SliderInt("integral resolution", &integral_resolution, 1, 100);
        _redist |= ImGui::InputFloat("radius", &R);
        _redist |= ImGui::Checkbox("Hyperbolic", &hyperbolic);

        redist |= _redist && auto_distribute;

        ImGui::Separator();
        ImGui::Text("QUALITY: %f", analytics.angle_min);
        ImGui::Text("mean: %f, sd: %f", analytics.angle_mean, analytics.angle_sd);

        ImGui::Text("\nSIZE: %f", analytics.size_max);
        ImGui::Text("mean: %f, sd: %f", analytics.size_mean, analytics.size_sd);

        ImGui::Separator();
        if (ImGui::Button("Compute optimal")) {
            if (hyperbolic) {
                triangulation = disk_hyp(N, R, 100, 100);
                analytics = analytics_hyp(triangulation);
            }
            else {
                triangulation = disk_euc(N, R, 100, 100);
                analytics = analytics_euc(triangulation);
            }

            //for (int i = 0; i < N; i++)
            //    vertices[i] = glm::vec2(triangulation.vertices[2 * i], triangulation.vertices[2 * i + 1]);

            N_bdry = triangulation.N_boundary;

            mesh.Update(triangulation.vertices, triangulation.triangles);

            redraw = true;
        }

        if (ImGui::Button("Save")) {
            std::ofstream file;
            
            std::string path = "../output/triangulation_square";
            path += hyperbolic ? "_hyp_" : "_euc_";
            path += std::to_string(triangulation.N_vertices);
            path += "(" + std::to_string(triangulation.N_boundary) + ")";

            file.open(path + ".txt");
            
            file << "vertices = [\n" << std::fixed << std::setprecision(16);
            for (double v : triangulation.vertices)
                file << v << ",\n";
            file << "]\n\n";

            file << "triangles = [\n";
            for (int I = 0; I < triangulation.triangles.size(); I += 3) {
                size_t i0 = triangulation.triangles[I + 0];
                size_t i1 = triangulation.triangles[I + 1];
                size_t i2 = triangulation.triangles[I + 2];
                
                file << i0 << ", " << i1 << ", " << i2 << ",\n";
            }
            file << "]";

            file.close();
        }

        //ImGui::SliderInt("Iterations per frame", &iterationsPerFrame, 1, 100);
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(glfwWindow);
        glfwPollEvents();
    }
}