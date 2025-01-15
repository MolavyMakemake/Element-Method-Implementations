#pragma once
#include <vector>

enum METRIC_ {
	METRIC_EUCLIDIAN = 0,
	METRIC_POINCARE,
};

class Integrator {
public:
	Integrator(size_t resolution);

	size_t N_vertices;

	void sample_euc(double x0, double y0, double x1, double y1, double x2, double y2, 
		std::vector<double>* samples, std::vector<double>* weights);
	void sample_hyp(double x0, double y0, double x1, double y1, double x2, double y2, 
		std::vector<double>* samples, std::vector<double>* weights);

	inline void sample_euc(std::vector<double>* vertices, size_t i0, size_t i1, size_t i2, 
		std::vector<double>* samples, std::vector<double>* weights) {
		return sample_euc(
			(*vertices)[2 * i0 + 0], (*vertices)[2 * i0 + 1],
			(*vertices)[2 * i1 + 0], (*vertices)[2 * i1 + 1],
			(*vertices)[2 * i2 + 0], (*vertices)[2 * i2 + 1],
			samples, weights);
	}

	inline void sample_hyp(std::vector<double>* vertices, size_t i0, size_t i1, size_t i2, 
		std::vector<double>* samples, std::vector<double>* weights) {
		return sample_hyp(
			(*vertices)[2 * i0 + 0], (*vertices)[2 * i0 + 1],
			(*vertices)[2 * i1 + 0], (*vertices)[2 * i1 + 1],
			(*vertices)[2 * i2 + 0], (*vertices)[2 * i2 + 1],
			samples, weights);
	}

private:
	std::vector<double> _vertices;
	std::vector<double> _weights;
};