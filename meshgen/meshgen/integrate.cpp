#include "integrate.h"
#include <glm/glm.hpp>

Integrator::Integrator(size_t resolution) {
	N_vertices = resolution * (resolution + 1) / 2;

	_vertices.reserve(2 * N_vertices);
	_weights.assign(N_vertices, .0);

	double h = 1.0 / (resolution - 1.0);
	double y = .0;

	size_t x = 0;
	size_t w = resolution;
	double s = h * h / 6.0;
	for (size_t i = 0; i < N_vertices - 1;) {
		_vertices.push_back(h * x);
		_vertices.push_back(y);
		
		if (x < w - 2) {
			_weights[i] += s;
			_weights[i + 1] += 2 * s;
			_weights[i + w] += 2 * s;
			_weights[i + w + 1] += s;

			i += 1;
			x += 1;
		}
		else {
			_vertices.push_back(h * (x + 1));
			_vertices.push_back(y);

			_weights[i] += s;
			_weights[i + 1] += s;
			_weights[i + w] += s;

			x = 0;
			w -= 1;
			y += h;
			i += 2;
		}
	}

	_vertices.push_back(0.0);
	_vertices.push_back(1.0);
}

void Integrator::sample_euc(double x0, double y0, double x1, double y1, double x2, double y2, 
	std::vector<double>* samples, std::vector<double>* weights) {
	
	samples->reserve(2 * N_vertices);
	weights->reserve(N_vertices);
	
	double u0 = x1 - x0;
	double v0 = y1 - y0;

	double u1 = x2 - x0;
	double v1 = y2 - y0;

	double s = glm::abs(u0 * v1 - u1 * v0);

	for (size_t i = 0; i < N_vertices; i++) {
		double z = _vertices[2 * i + 0];
		double w = _vertices[2 * i + 1];

		samples->push_back(x0 + u0 * z + u1 * w);
		samples->push_back(y0 + v0 * z + v1 * w);
		weights->push_back(_weights[i] * s);
	}
}

void Integrator::sample_hyp(double x0, double y0, double x1, double y1, double x2, double y2, 
	std::vector<double>* samples, std::vector<double>* weights) {
	
	samples->reserve(2 * N_vertices);
	weights->reserve(N_vertices);
	
	double t0 = 2.0 / (1.0 + x0 * x0 + y0 * y0);
	double t1 = 2.0 / (1.0 + x1 * x1 + y1 * y1);
	double t2 = 2.0 / (1.0 + x2 * x2 + y2 * y2);

	x0 *= t0; y0 *= t0;
	x1 *= t1; y1 *= t1;
	x2 *= t2; y2 *= t2;
	
	double u0 = x1 - x0;
	double v0 = y1 - y0;

	double u1 = x2 - x0;
	double v1 = y2 - y0;

	double s = glm::abs(u0 * v1 - u1 * v0);

	for (size_t i = 0; i < N_vertices; i++) {
		double z = _vertices[2 * i + 0];
		double w = _vertices[2 * i + 1];

		double x = x0 + u0 * z + u1 * w;
		double y = y0 + v0 * z + v1 * w;

		double T = 1.0 - x * x - y * y;
		double t = glm::sqrt(T);
		weights->push_back(_weights[i] * s / (t * T));

		samples->push_back(x / (1 + t));
		samples->push_back(y / (1 + t));
	}
}