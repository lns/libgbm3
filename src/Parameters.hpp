#pragma once

class Parameters {
public:
	double eta; // stepsize
	double l1reg;
	double l2reg;
	double inner_thres; // stop threshold in inner loop
	double inner_precs; // stop precision in inner loop
	double outer_thres; // stop threshold in outer loop
	double outer_precs; // stop precision in outer loop
	float cut_thres; // threshold for difference of feature value for a cut
	double min_node_weight;
};

