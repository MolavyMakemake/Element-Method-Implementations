#include "generate.h"

namespace {

    std::vector<double> fibonacci(int N_vertices, int N_boundary, double radius) {
        std::vector<double> vertices;
        vertices.reserve(2 * N_vertices);
        
        N_boundary = std::min(N_vertices, N_boundary);
        double arg = glm::pi<double>() * (3 - glm::sqrt(5));

        double S = glm::sqrt(N_vertices / (N_vertices - .5f * N_boundary)) * radius;
        double t = 0;
        for (int i = 0; i < N_vertices - N_boundary; i++) {
            double r = glm::sqrt((double)i / N_vertices) * S;

            vertices.push_back(r * glm::cos(t));
            vertices.push_back(r * glm::sin(t));
            t += arg;
        }

        arg = 2 * glm::pi<double>() / N_boundary;
        for (int i = N_vertices - N_boundary; i < N_vertices; i++) {
            vertices.push_back(radius * glm::cos(t));
            vertices.push_back(radius * glm::sin(t));
            t += arg;
        }

        return vertices;
    }

    double magnitude(double x0, double y0, double x, double y) {
        double dx = x0 - x;
        double dy = y0 - y;
        return dx * dx + dy * dy;
    }

    size_t voronoi_index(std::vector<double>& vertices, double x, double y) {
        size_t j = 0;
        double dst = 1e10;

        for (size_t i = 0; i < vertices.size() / 2; i++) {
     
            double _dst = magnitude(vertices[2 * i + 0], vertices[2 * i + 1], x, y);

            j = _dst < dst ? i : j;
            dst = std::min<double>(dst, _dst);
        }

        return j;
    }
    
    // Decent approximation and way faster
    size_t sloppy_voronoi_index(std::vector<double>& vertices, size_t i0, size_t i1, size_t i2, double x, double y) {

        double dst = magnitude(vertices[2 * i0 + 0], vertices[2 * i0 + 1], x, y);
        double _dst = magnitude(vertices[2 * i1 + 0], vertices[2 * i1 + 1], x, y);

        size_t j = _dst < dst ? i1 : i0;
        dst = std::min<double>(dst, _dst);

        return magnitude(vertices[2 * i2 + 0], vertices[2 * i2 + 1], x, y) < dst ? i2 : j;
    }

    void iterate(triangulation_t& triangulation, double radius, Integrator& integrator) {
        std::vector<double> vertices;
        std::vector<double> area;

        vertices.assign(2 * triangulation.N_vertices, 0);
        area.assign(triangulation.N_vertices, 0);

        for (size_t I = 0; I < triangulation.triangles.size(); I += 3) {
            size_t i0 = triangulation.triangles[I + 0];
            size_t i1 = triangulation.triangles[I + 1];
            size_t i2 = triangulation.triangles[I + 2];

            std::vector<double> samples;
            std::vector<double> weights;
            integrator.sample_euc(&triangulation.vertices, i0, i1, i2, &samples, &weights);
            for (size_t j = 0; j < integrator.N_vertices; j++) {
                double x = samples[2 * j + 0];
                double y = samples[2 * j + 1];

                size_t k = sloppy_voronoi_index(triangulation.vertices, i0, i1, i2, x, y);

                vertices[2 * k + 0] += x * weights[j];
                vertices[2 * k + 1] += y * weights[j];

                area[k] += weights[j];
            }
        }

        for (size_t i = 0; i < triangulation.N_vertices - triangulation.N_boundary; i++) {
            vertices[2 * i + 0] /= area[i];
            vertices[2 * i + 1] /= area[i];
        }

        for (size_t i = triangulation.N_vertices - triangulation.N_boundary; 
            i < triangulation.N_vertices; i++) {

            vertices[2 * i + 0] = triangulation.vertices[2 * i + 0];
            vertices[2 * i + 1] = triangulation.vertices[2 * i + 1];
        }

        delaunator::Delaunator d(vertices);
        triangulation.vertices = vertices;
        triangulation.triangles = d.triangles;
    }

}

triangulation_t disk_euc(int N_vertices, int N_boundary, double radius, int N_iterations, int integral_resolution) {
    triangulation_t triangulation;
    triangulation.vertices = fibonacci(N_vertices, N_boundary, radius);
    triangulation.N_vertices = N_vertices;
    triangulation.N_boundary = N_boundary;
    
    delaunator::Delaunator d(triangulation.vertices);
    triangulation.triangles = d.triangles;

    Integrator integrator(integral_resolution);
    for (int i = 0; i < N_iterations; i++) {
        iterate(triangulation, radius, integrator);
    }

    return triangulation;
}

namespace {
    void triangle_quality(std::vector<double>& vertices, size_t i0, size_t i1, size_t i2, double* angle, double* size) {
        double x0 = vertices[2 * i0 + 0];
        double x1 = vertices[2 * i0 + 1];
        double y0 = vertices[2 * i1 + 0];
        double y1 = vertices[2 * i1 + 1];
        double z0 = vertices[2 * i2 + 0];
        double z1 = vertices[2 * i2 + 1];

        double u0 = y0 - x0;
        double u1 = y1 - x1;
        double v0 = z0 - y0;
        double v1 = z1 - y1;
        double w0 = x0 - z0;
        double w1 = x1 - z1;

        double lu = glm::sqrt(u0 * u0 + u1 * u1);
        double lv = glm::sqrt(v0 * v0 + v1 * v1);
        double lw = glm::sqrt(w0 * w0 + w1 * w1);

        double a_uv = (u0 * v0 + u1 * v1) / (lu * lv);
        double a_vw = (v0 * w0 + v1 * w1) / (lv * lw);
        double a_wu = (w0 * u0 + w1 * u1) / (lw * lu);

        *angle = glm::acos(-std::min<double>(a_uv, std::min<double>(a_vw, a_wu)));
        *size = std::max<double>(lu, std::max<double>(lv, lw));
    }

    void compute_analytics(analytics_t* analytics) {
        size_t N = std::min<size_t>(analytics->angles.size(), analytics->sizes.size());

        analytics->angle_mean = 0;
        analytics->angle_sd = 0;
        analytics->angle_min = 1e10;

        analytics->size_mean = 0;
        analytics->size_sd = 0;
        analytics->size_max = 0;

        for (size_t i = 0; i < N; i++) {
            double angle = analytics->angles[i];
            double size = analytics->sizes[i];

            analytics->angle_mean += angle;
            analytics->angle_sd += angle * angle;
            analytics->angle_min = std::min<double>(analytics->angle_min, angle);

            analytics->size_mean += size;
            analytics->size_sd += size * size;
            analytics->size_max = std::max<double>(analytics->size_max, size);
        }

        analytics->angle_mean /= N;
        analytics->size_mean /= N;

        analytics->angle_sd = glm::sqrt(analytics->angle_sd / N - analytics->angle_mean * analytics->angle_mean);
        analytics->size_sd = glm::sqrt(analytics->size_sd / N - analytics->size_mean * analytics->size_mean);
    }
}

analytics_t analytics_euc(triangulation_t& triangulation) {
    analytics_t analytics;

    analytics.angles.reserve(triangulation.triangles.size() / 3);
    analytics.sizes.reserve(triangulation.triangles.size() / 3);

    for (size_t i = 0; i < triangulation.triangles.size(); i += 3) {

        size_t i0 = triangulation.triangles[i + 0];
        size_t i1 = triangulation.triangles[i + 1];
        size_t i2 = triangulation.triangles[i + 2];

        double angle;
        double size;
        triangle_quality(triangulation.vertices, i0, i1, i2, &angle, &size);

        analytics.angles.push_back(angle);
        analytics.sizes.push_back(size);
    }

    compute_analytics(&analytics);
    return analytics;
}

triangulation_t disk_euc(int N_vertices, double radius, int N_iterations, int integral_resolution) {
    clock_t start_clock = std::clock();

    int N = N_vertices / 2; // Only check half of possible # boundary points    
    int k = (int)glm::log2((double)N_vertices - 1);

    std::vector<double> quality(N);
    std::vector<bool> mask(N);
    mask.assign(N, true);

    int N_boundary;
    for (int it = 1;; it++) {
        int _N_iterations = 70 * it / k;
        int _integral_resolution = std::max<int>(30 * it / k, 5);

        std::cout << 100 * it / k << "; ir: " << _integral_resolution << ", it: " << _N_iterations << std::endl;

        // Compute triangulations
        std::vector<int> index;

        for (int i = 0; i < N; i++) {
            if (!mask[i])
                continue;

            triangulation_t triangulation = disk_euc(N_vertices, i, radius, _N_iterations, _integral_resolution);
            analytics_t analytics = analytics_euc(triangulation);

            quality[i] = analytics.angle_min;
            index.push_back(i);
        }

        // Cull half
        std::sort(index.begin(), index.end(), [quality](int l, int r) { return quality[l] < quality[r]; });
        
        if (index.size() == 2) {
            N_boundary = index[1];
            break;
        }

        for (int i = 0; i < index.size() / 2; i++)
            mask[index[i]] = false;
    }

    std::cout << "computing final triangulation..." << std::endl;
    triangulation_t _out = disk_euc(N_vertices, N_boundary, radius, N_iterations, integral_resolution);
    std::cout << "finished in " << (std::clock() - start_clock) / CLOCKS_PER_SEC << " seconds\n\n";

    return _out;
}