#pragma once

#include <sstream>
#include <string>
#include <cmath>
#include "qstdlib.hpp"

/**
 * Soft thresholding
 */
template<typename T>
inline T soft_thres(T x, T thres) {
	return (x>thres?(x-thres):(x<-thres?(x+thres):0));
}

/**
 * Definition 1:
 *
 * Soft Thresholding solution of l1-reg loss
 *
 * L(x) := (x+c)^2 + d*fabs(x)
 * x_min = soft_thres(-c, 0.5*d);
 * L_min = L(0) - x_min^2
 */

/**
 * Definition 2:
 *
 * Regularized Loss is
 * L(x) L= 0.5*a*x^2 + b*x + 0.5*l2reg*x^2 + l1reg*fabs(x)
 * 
 * Solution is:
 * 	x_min = soft_thres(-b, l1reg)/(a+l2reg)
 *
 * (The implementation is inspired from Tong Zhang's code.)
 */
// Regularized Loss
template<typename T>
inline T reg_loss(T a, T b, T l2reg, T l1reg, T x) {
	return(0.5*(a+l2reg)*x*x + b*x + l1reg*fabs(x));
}
// Argmin of Regularized Loss
template<typename T>
inline T argmin_reg_loss(T a, T b, T l2reg, T l1reg) {
	return(soft_thres(-b, l1reg)/(a+l2reg));
}
// Minimum of Regularized Loss
template<typename T>
inline T min_reg_loss(T a, T b, T l2reg, T l1reg) {
	double x_min = argmin_reg_loss(a,b,l2reg,l1reg);
	return(reg_loss(a,b,l2reg,l1reg,x_min));
}

/**
 * String to Any Type
 */
template<typename F>
inline F parse_cstr(const char * bgn, const char * end);

template<>
inline std::string parse_cstr(const char * bgn, const char * end) {
	return std::string(bgn,end);
}

template<>
inline unsigned parse_cstr(const char * bgn, const char * end) {
	const char * q;
	unsigned ret = qstrtoul(bgn, &q);
	if(q!=end)
		qlog_warning("Parsing error: '%s'\n",std::string(bgn,end).c_str());
	return ret;
}

template<>
inline unsigned long parse_cstr(const char * bgn, const char * end) {
	const char * q;
	unsigned long ret = qstrtoul(bgn, &q);
	if(q!=end)
		qlog_warning("Parsing error: '%s'\n",std::string(bgn,end).c_str());
	return ret;
}

#if 0
template<typename F>
F parse_cstr(const char * bgn, const char * end) {
	F ret;
	std::istringstream(std::string(bgn,end)) >> ret;
	return ret;
}
#endif

