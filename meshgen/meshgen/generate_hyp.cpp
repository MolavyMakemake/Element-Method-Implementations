#include "generate.h"

namespace {

    std::vector<double> fibonacci(int N_vertices, int N_boundary, double radius) {
        std::vector<double> vertices;
        vertices.reserve(2 * N_vertices);

        N_boundary = std::min(N_vertices, N_boundary);
        double arg = glm::pi<double>() * (3 - glm::sqrt(5));

        double S = glm::sqrt(N_vertices / (N_vertices - .5f * N_boundary)) * glm::sinh(radius / 2);
        double t = 0;
        for (int i = 0; i < N_vertices - N_boundary; i++) {
            double x = glm::sqrt((float)i / N_vertices) * S;
            double r = x / glm::sqrt(1 + x * x);

            vertices.push_back(r * glm::cos(t));
            vertices.push_back(r * glm::sin(t));
            t += arg;
        }

        arg = 2 * glm::pi<double>() / N_boundary;
        double r = glm::tanh(radius / 2);
        for (int i = N_vertices - N_boundary; i < N_vertices; i++) {
            vertices.push_back(r * glm::cos(t));
            vertices.push_back(r * glm::sin(t));
            t += arg;
        }

        return vertices;
    }

    std::vector<double> square(int N_vertices, int N_boundary, double radius) {
        std::vector<double> vertices;

        N_boundary = std::min(N_vertices, N_boundary);
        vertices.reserve(2 * (N_vertices * N_vertices + 4 * N_boundary));

        double r = glm::tanh(radius);

        double d = 2.0 * r / (N_vertices + 1.0);
        for (int i = 0; i < N_vertices * N_vertices; i++) {
            int x = i % N_vertices;
            int y = i / N_vertices;
            vertices.push_back(-r + d * (x + 1.0));
            vertices.push_back(-r + d * (y + 1.0));
        }

        d = 2.0 * r / N_boundary;
        for (int i = 0; i < N_boundary; i++) {
            vertices.push_back(r);
            vertices.push_back(-r + d * i);

            vertices.push_back(r - d * i);
            vertices.push_back(r);

            vertices.push_back(-r);
            vertices.push_back(r - d * i);

            vertices.push_back(-r + d * i);
            vertices.push_back(-r);
        }

        for (int i = 0; i < vertices.size(); i += 2) {
            double x0 = vertices[i + 0];
            double x1 = vertices[i + 1];
            double t = 1.0 + glm::sqrt(1.0 - x0 * x0 - x1 * x1);

            vertices[i + 0] = x0 / t;
            vertices[i + 1] = x1 / t;
        }

        return vertices;
    }

    // Requires x, y to be fixed for correct comparisons
    double voronoi_magnitude(double x0, double y0, double x, double y) {
        double dx = x0 - x;
        double dy = y0 - y;
        return (dx * dx + dy * dy) / (1 - x0 * x0 - y0 * y0);
    }

    size_t voronoi_index(std::vector<double>& vertices, double x, double y) {
        size_t j = 0;
        double dst = 1e10;

        for (size_t i = 0; i < vertices.size() / 2; i++) {

            double _dst = voronoi_magnitude(vertices[2 * i + 0], vertices[2 * i + 1], x, y);

            j = _dst < dst ? i : j;
            dst = std::min<double>(dst, _dst);
        }

        return j;
    }

    // Decent approximation and way faster
    size_t sloppy_voronoi_index(std::vector<double>& vertices, size_t i0, size_t i1, size_t i2, double x, double y) {

        double dst = voronoi_magnitude(vertices[2 * i0 + 0], vertices[2 * i0 + 1], x, y);
        double _dst = voronoi_magnitude(vertices[2 * i1 + 0], vertices[2 * i1 + 1], x, y);

        size_t j = _dst < dst ? i1 : i0;
        dst = std::min<double>(dst, _dst);

        return voronoi_magnitude(vertices[2 * i2 + 0], vertices[2 * i2 + 1], x, y) < dst ? i2 : j;
    }

    void iterate(triangulation_t& triangulation, double radius, Integrator& integrator, bool retriangulate=true) {
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
            integrator.sample_hyp(&triangulation.vertices, i0, i1, i2, &samples, &weights);
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
            triangulation.vertices[2 * i + 0] = vertices[2 * i + 0] / area[i];
            triangulation.vertices[2 * i + 1] = vertices[2 * i + 1] / area[i];
        }

        if (retriangulate) {
            delaunator::Delaunator d(triangulation.vertices);
            triangulation.triangles = d.triangles;
        }
    }

}

triangulation_t disk_hyp(int N_vertices, int N_boundary, double radius, int N_iterations, int integral_resolution) {
    triangulation_t triangulation;
    triangulation.vertices = fibonacci(N_vertices, N_boundary, radius);
    triangulation.N_vertices = N_vertices;
    triangulation.N_boundary = N_boundary;

    delaunator::Delaunator d(triangulation.vertices);
    triangulation.triangles = d.triangles;

    Integrator integrator(integral_resolution);
    for (int i = 0; i < N_iterations; i++) {
        iterate(triangulation, radius, integrator, true);
    }

    return triangulation;
}

void cull_triangulation(triangulation_t& triangulation) {
    size_t i = 0;
    size_t N = triangulation.N_vertices - triangulation.N_boundary;

    for (std::vector<size_t>::iterator it = triangulation.triangles.begin();
        it != triangulation.triangles.end();) {

        if (*(it + 0) >= N &&
            *(it + 1) >= N &&
            *(it + 2) >= N)
        {
            it = triangulation.triangles.erase(it, it + 3);
        }
        else
            it += 3;
    }
}

triangulation_t square_hyp(int N_vertices, int N_boundary, double radius, int N_iterations, int integral_resolution) {
    triangulation_t triangulation;
    triangulation.vertices = square(N_vertices, N_boundary, radius);
    triangulation.N_vertices = N_vertices * N_vertices + 4 * N_boundary;
    triangulation.N_boundary = 4 * N_boundary;

    delaunator::Delaunator d(triangulation.vertices);
    triangulation.triangles = d.triangles;
    cull_triangulation(triangulation);

    Integrator integrator(integral_resolution);
    for (int i = 0; i < N_iterations; i++) {
        iterate(triangulation, radius, integrator);
        cull_triangulation(triangulation);
    }

    return triangulation;
}

namespace {
    double dot_klein(double u0, double v0, double u1, double v1, double x, double y) {
        double k = (1.0 - x * x - y * y);
        double z = ((1 - y * y) * u0 + x * y * v0) / (k * k);
        double w = ((1 - x * x) * v0 + x * y * u0) / (k * k);

        return z * u1 + w * v1;
    }

    double cos_a(double u0, double v0, double u1, double v1, double x, double y) {
        double k = (1.0 - x * x - y * y);

        double z0 = ((1 - y * y) * u0 + x * y * v0) / (k * k);
        double w0 = ((1 - x * x) * v0 + x * y * u0) / (k * k);
        double z1 = ((1 - y * y) * u1 + x * y * v1) / (k * k);
        double w1 = ((1 - x * x) * v1 + x * y * u1) / (k * k);

        return (z0 * u1 + w0 * v1) / glm::sqrt((z0 * u0 + w0 * v0) * (z1 * u1 + w1 * v1));
    }

    void triangle_quality(std::vector<double>& vertices, size_t i0, size_t i1, size_t i2, double* angle, double* size) {
        double x0 = vertices[2 * i0 + 0];
        double y0 = vertices[2 * i0 + 1];
        double x1 = vertices[2 * i1 + 0];
        double y1 = vertices[2 * i1 + 1];
        double x2 = vertices[2 * i2 + 0];
        double y2 = vertices[2 * i2 + 1];

        double u0 = x1 - x0;
        double v0 = y1 - y0;
        double u1 = x2 - x1;
        double v1 = y2 - y1;
        double u2 = x0 - x2;
        double v2 = y0 - y2;

        double s0 = 2 * (u0 * u0 + v0 * v0) / ((1.0 - x0 * x0 - y0 * y0) * (1.0 - x1 * x1 - y1 * y1));
        double s1 = 2 * (u1 * u1 + v1 * v1) / ((1.0 - x1 * x1 - y1 * y1) * (1.0 - x2 * x2 - y2 * y2));
        double s2 = 2 * (u2 * u2 + v2 * v2) / ((1.0 - x2 * x2 - y2 * y2) * (1.0 - x0 * x0 - y0 * y0));

        *size = glm::acosh(1.0 + std::max<double>(s0, std::max<double>(s1, s2)));

        // map points to Klein model to compute angle
        double t0 = 2.0 / (1.0 + x0 * x0 + y0 * y0);
        double t1 = 2.0 / (1.0 + x1 * x1 + y1 * y1);
        double t2 = 2.0 / (1.0 + x2 * x2 + y2 * y2);

        x0 *= t0; y0 *= t0;
        x1 *= t1; y1 *= t1;
        x2 *= t2; y2 *= t2;

        u0 = x1 - x0;
        v0 = y1 - y0;
        u1 = x2 - x1;
        v1 = y2 - y1;
        u2 = x0 - x2;
        v2 = y0 - y2;

        double a0 = cos_a(u0, v0, u1, v1, x1, y1);
        double a1 = cos_a(u1, v1, u2, v2, x2, y2);
        double a2 = cos_a(u2, v2, u0, v0, x0, y0);

        *angle = glm::acos(-std::min<double>(a0, std::min<double>(a1, a2)));
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

analytics_t analytics_hyp(triangulation_t& triangulation) {
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

triangulation_t disk_hyp(int N_vertices, double radius, int N_iterations, int integral_resolution) {

    int N = N_vertices / 2;    
    int k = (int)glm::log2((double)N_vertices - 1);

    std::vector<double> quality(N);
    std::vector<bool> mask(N);
    mask.assign(N, true);

    for (int i = 0; i < 5; i++)
        mask[i] = false;

    int N_boundary;
    for (int it = 0;; it++) {
        int _N_iterations = 30 * (it + 1) / (k + 1);
        int _integral_resolution = 60 * (it + 1) / (k + 1);

        std::cout << 100 * it / k << "; ir: " << _integral_resolution << ", it: " << _N_iterations << std::endl;

        // Compute triangulations
        std::vector<int> index;

        for (int i = 0; i < N; i++) {
            if (!mask[i])
                continue;

            triangulation_t triangulation = disk_hyp(N_vertices, i, radius, _N_iterations, _integral_resolution);
            analytics_t analytics = analytics_hyp(triangulation);

            quality[i] = analytics.size_max;
            index.push_back(i);
        }

        // Cull half
        std::sort(index.begin(), index.end(), [quality](int l, int r) { return quality[l] > quality[r]; });

        if (index.size() == 2) {
            N_boundary = index[1];
            break;
        }

        for (int i = 0; i < index.size() / 2; i++)
            mask[index[i]] = false;
    }

    return disk_hyp(N_vertices, N_boundary, radius, N_iterations, integral_resolution);
}