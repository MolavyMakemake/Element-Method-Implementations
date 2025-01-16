#pragma once

#include <vector>
#include "integrate.h"

#include <delaunator.hpp>
#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>

struct triangulation_t {
	std::vector<double> vertices;
	std::vector<size_t> triangles;

	size_t N_vertices;
	size_t N_boundary;
};

struct analytics_t {
	std::vector<double> angles;
	std::vector<double> sizes;

	double angle_mean;
	double angle_sd;
	double angle_min;

	double size_mean;
	double size_sd;
	double size_max;
};

analytics_t analytics_euc(triangulation_t& triangulation);
analytics_t analytics_hyp(triangulation_t& triangulation);

triangulation_t disk_euc(int N_vertices, int N_boundary, double radius, int N_iterations, int integral_resolution);
triangulation_t disk_hyp(int N_vertices, int N_boundary, double radius, int N_iterations, int integral_resolution);

triangulation_t square_euc(int N_vertices, int N_boundary, double radius, int N_iterations, int integral_resolution);
triangulation_t square_hyp(int N_vertices, int N_boundary, double radius, int N_iterations, int integral_resolution);

inline triangulation_t disk(int N_vertices, int N_boundary, double radius, METRIC_ metric, int N_iterations, int integral_resolution) {
	switch (metric) {
	case METRIC_EUCLIDIAN:
		return disk_euc(N_vertices, N_boundary, radius, N_iterations, integral_resolution);

	case METRIC_POINCARE:
		return disk_hyp(N_vertices, N_boundary, radius, N_iterations, integral_resolution);

	default:
		throw 0;
	}
}

inline triangulation_t square(int N_vertices, int N_boundary, double radius, METRIC_ metric, int N_iterations, int integral_resolution) {
	switch (metric) {
	case METRIC_EUCLIDIAN:
		return square_euc(N_vertices, N_boundary, radius, N_iterations, integral_resolution);

	case METRIC_POINCARE:
		return square_hyp(N_vertices, N_boundary, radius, N_iterations, integral_resolution);

	default:
		throw 0;
	}
}

triangulation_t disk_euc(int N_vertices, double radius, int N_iterations, int integral_resolution);
triangulation_t disk_hyp(int N_vertices, double radius, int N_iterations, int integral_resolution);

inline triangulation_t disk(int N_vertices, double radius, int N_iterations, int integral_resolution, METRIC_ metric) {
	switch (metric) {
	case METRIC_EUCLIDIAN:
		return disk_euc(N_vertices, radius, N_iterations, integral_resolution);

	case METRIC_POINCARE:
		return disk_hyp(N_vertices, radius, N_iterations, integral_resolution);

	default:
		throw 0;
	}
}