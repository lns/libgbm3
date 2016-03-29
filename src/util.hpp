#pragma once

#include <sstream>
#include <string>
#include "qstdlib.hpp"

/**
 * Soft thresholding
 */
template<typename T>
inline T soft_thres(T x, T thres) {
	return (x>thres?(x-thres):(x<-thres?(x+thres):0));
}

/**
 * Soft Thresholding solution of l1-reg loss
 *
 * L(x) := (x+c)^2 + d*fabs(x)
 * x_min = soft_thres(-c, 0.5*d);
 * L_min = L(0) - x_min^2
 */

/**
 * Regularized Loss is
 *
 * 	L(x) := 0.5*a*x^2 + b*x + 0.5*l2reg*(x0+x)^2 - 0.5*l2reg*(x0)^2 
 * 		+ l1reg*fabs(x0+x) - l1reg*fabs(x0)
 *
 * Solution is:
 * 	x_min = soft_thres(a*x0-b, l1reg)/(a+l2reg)-x0
 * 	L_min = L(0) - 0.5*(a+l2reg)*(x_min+x0)^2
 */

// Argmin of Regularized Loss
template<typename T>
inline T argmin_reg_loss(T a, T b, T l2reg, T l1reg, T x0) {
	return(soft_thres(a*x0-b, l1reg)/(a+l2reg)-x0);
}

// Minimum of Regularized Loss
template<typename T>
inline T min_reg_loss(T a, T b, T l2reg, T l1reg, T x0) {
	T x_min = soft_thres(a*x0-b, l1reg)/(a+l2reg);
	return(-0.5*(a+l2reg)*x_min*x_min);
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

