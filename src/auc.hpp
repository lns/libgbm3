#pragma once

#include <vector>
#include <algorithm>
#include <limits>

/**
 * Calculate AUC.
 * Return NA if all truth are the same
 * @len length of array. Max length is 2^31
 */
template<typename T, typename F>
double calc_auc(const T* truth, const F* prob, size_t len) {

	class Comp {
		const F* x_;
		public:
		Comp(const F* x): x_(x) {}
		bool operator()(int a, int b) const {
			return x_[a-1]<x_[b-1];
		}
	};

	std::vector<int> order(len);
	for(size_t i=0;i<len;i++)
		order[i] = i+1;
	std::sort(order.begin(), order.end(), Comp(prob));
	std::vector<int> rank_x2(len);
	int index = 0;
	while(index<len) {
		int i = index;
		while(++i<len and prob[order[i]-1]==prob[order[index]-1])
			;
		for(int k=index;k<i;k++)
			rank_x2[order[k]-1] = index+i;
		index = i;
	}
	int n1=0;
	for(int i=0;i<len;i++)
		n1 += (truth[i]>0?1:0);
	int n0=len-n1;
	if(n1==0 or n0==0) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	long u=0;
	for(int i=0;i<len;i++)
		u += (truth[i]>0?rank_x2[i]:0);
	u -= n1*(n1+1);
	double res = u/(2.0f*n1*n0);
	return res;
}

