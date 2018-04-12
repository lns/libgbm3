#pragma once

#include <algorithm>
#include <cmath>

using std::max;

/**
 * For more objectives, see http://www.saedsayad.com/docs/gbm2.pdf
 */
template <typename T>
class Objective {
public:
	virtual void Loss(const T* y, int yinc, const T* f, int finc, size_t n, 
			T* res, int resinc) const = 0;
	virtual void FirstOrder(const T* y, int yinc, const T* f, int finc, size_t n,
			T* res, int resinc) const = 0;
	virtual void SecondOrder(const T* y, int yinc, const T* f, int finc, size_t n,
			T* res, int resinc) const = 0;
	virtual ~Objective() {} // why this cannot be 0?
};

template <typename T>
class LogLoss : public Objective<T> {
public:
	void Loss(const T* y, int yinc, const T* f, int finc, size_t n,
			T* res, int resinc) const override {
		#pragma omp parallel for
		for(size_t i=0;i<n;i++)
			res[i*resinc] = log(1+exp((1-2*y[i*yinc])*f[i*finc]));
	}
	void FirstOrder(const T* y, int yinc, const T* f, int finc, size_t n,
			T* res, int resinc) const override {
		#pragma omp parallel for
		for(size_t i=0;i<n;i++)
			res[i*resinc] = 1/(1+exp(-f[i*finc])) - y[i*yinc];
	}
	void SecondOrder(const T* y, int yinc, const T* f, int finc, size_t n,
			T* res, int resinc) const override {
		#pragma omp parallel for
		for(size_t i=0;i<n;i++) {
			T p = 1/(1+exp(-f[i*finc]));
			res[i*resinc] = max(p*(1-p),1e-16);
		}
	}
	~LogLoss() {}
};

template <typename T>
class L2Loss : public Objective<T> {
public:
	void Loss(const T* y, int yinc, const T* f, int finc, size_t n,
			T* res, int resinc) const override {
		#pragma omp parallel for
		for(size_t i=0;i<n;i++)
			res[i*resinc] = (y[i*yinc]-f[i*finc])*(y[i*yinc]-f[i*finc]);
	}
	void FirstOrder(const T* y, int yinc, const T* f, int finc, size_t n,
			T* res, int resinc) const override {
		#pragma omp parallel for
		for(size_t i=0;i<n;i++)
			res[i*resinc] = 2*(y[i*yinc]-f[i*finc]);
	}
	void SecondOrder(const T* y, int yinc, const T* f, int finc, size_t n,
			T* res, int resinc) const override {
		#pragma omp parallel for
		for(size_t i=0;i<n;i++)
			res[i*resinc] = 2;
	}
	~L2Loss() {}
};

