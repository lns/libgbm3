#pragma once

#include "qopt.hpp"

class Parameters {
public:
	std::string train_file_path;
	std::string test_file_path;
	std::string in_model_path;
	std::string out_model_path;
	std::string out_conf_path;

	std::string objective; // Only LosLoss is supported.
	double l1reg;
	double l2reg;
	double proximal_l2; // l2 coefficient for proximal gradient
	double col_sampling; // column sampling ratio (by node)
	double eta; // stepsize
	double min_node_weight; // minimum node weight
	double relative_tol; // stopping criterion
	float cut_thres; // threshold for difference of feature value for a cut
	unsigned max_trees; // max number of trees
	unsigned max_leaves; // max number of total leaves
	unsigned max_depth; // max depth of leaf node
	unsigned max_tree_node; // max number of nodes in a tree
	unsigned max_inner_iter; // max iteration of inner loop (refine())
	unsigned search_recent_tree; // number of recent tree to search
	bool check_loss; // make sure loss is decreasing with extra computation

	qlib::OptParser<Parameters> * opr; // the OptParser

	// Register in OptParser
	void operator()(qlib::OptParser<Parameters>& x, qlib::OptAction _) {
		// optional option
		#define O(var,descr,val) x(_,var,"-"#var,descr,static_cast<decltype(var)>(val))
		// required option
		#define X(var,descr) x(_,var,"-"#var,descr)
		X(train_file_path,"Path of train file");
		O(test_file_path,"Path of test file","");
		O(in_model_path,"Path of input model","");
		O(out_model_path,"Path of output model","model.conf");
		O(out_conf_path,"Path of output conf","");
		O(objective,"Objective ('LogLoss' or 'L2Loss')","LogLoss");
		O(l1reg,"L1 regularization coefficient",1e-6);
		O(l2reg,"L2 regularization coefficient",1e-6);
		O(proximal_l2,"L2 Proximal coefficient for Newton-method",0.25);
		O(col_sampling,"Column sampling ratio (by node)",1.0);
		O(eta,"stepsize for newton update",1.0);
		O(min_node_weight,"Minimum weights(samples) for a node",1.0);
		O(relative_tol,"Relative tolerance for stopping criterion",1e-6);
		O(cut_thres,"Threshold for difference of feature value for a cut",1e-6);
		O(max_trees,"Max number of trees",100);
		O(max_leaves,"Max leaf nodes in the forest",400);
		O(max_depth,"Max depth of leaf node",2);
		O(max_tree_node,"Max nodes in a tree (max 127)",127);
		O(max_inner_iter,"Max number of iteration per round (fully-corrective)",10);
		O(search_recent_tree,"Number of recent tree to search for a split",1);
		O(check_loss,"Check loss while optimizing (for tuning)",false);
		#undef O
		#undef X
	}
};

