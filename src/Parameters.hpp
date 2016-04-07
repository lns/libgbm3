#pragma once

class Parameters {
public:
	double eta; // stepsize
	double l1reg;
	double l2reg;
	double update_thres; // stop threshold in update loop
	double update_precs; // stop precision in update loop
	// TODO: add inner stop criteria in refine()
	double inner_thres; // stop threshold in inner loop
	double inner_precs; // stop precision in inner loop
	double outer_thres; // stop threshold in outer loop
	double outer_precs; // stop precision in outer loop
	double proximal_l2; // l2 coefficient for proximal gradient
	float cut_thres; // threshold for difference of feature value for a cut
	double min_node_weight;
	unsigned max_tree_node; // max number of nodes in a tree
	unsigned max_trees; // max number of trees
	unsigned max_leaves; // max number of total leaves
	unsigned max_depth; // max depth of leaf node
	unsigned max_inner_iter; // max iteration of inner loop (refine())
	unsigned search_recent_tree; // number of recent tree to search
};

