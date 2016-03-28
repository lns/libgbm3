#pragma once

class Parameters {
public:
	double eta; // stepsize
	double l1reg;
	double l2reg;
	double inner_thres; // stop threshold in inner loop
	double inner_precs; // stop precision in inner loop
	double min_node_weight;
};

