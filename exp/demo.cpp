#include "DataSheet.hpp"
#include "GBM.hpp"
#include "auc.hpp"
#include <string>

int main(int argc, char* argv[]) {
	typedef std::string FeaType;
	Parameters param;
	param.eta = 1.0f;
	param.l1reg = 1e-5f;
	param.l2reg = 1e-5f;
	param.proximal_l2 = 1e-2f; // for LR, this should no larger than 0.25f
	// Max number of nodes in a tree
	param.max_tree_node = 10;
	// Number of recent tree to search
	param.search_recent_tree = 1;
	// Max depth of node
	param.max_depth = 4;
	// Max number of trees
	param.max_trees = 200;
	// Max number of leaves
	param.max_leaves = 800;
	// Stop criteria:
	// inner: gain < relative_tolerance * recent_loss
	// outer: loss reduction < relative_tolerance * recent_loss
	param.relative_tol = -1e6; // set to -Inf to disable this.
	//
	param.max_inner_iter = 10;
	// Check loss increasing when update tree prediction
	param.check_loss = false;
	param.min_node_weight = 10.0f;
	param.cut_thres = 1e-6;
	//
	const char * filename = "/dev/stdin";
	if(argc>1)
		filename = argv[1];
	if(true) { // Training
		GBM<FeaType> gbm(param);
		gbm._obj = new LogLoss<double>();
		//gbm.read_data_from_libsvm("../data/fsg/test.10k.txt");
		gbm.read_data_from_libsvm(filename);
		//gbm.read_data_from_libsvm("heart_scale.txt");
		gbm.assign_weights(false); // balance positive/negitive sample
		gbm.boost();
		gbm.save_to_file("model.txt");
		// teardown
		delete gbm._obj;
	}
	//
	if(false) { // Testing
		DataSheet<FeaType> ds;
		std::vector<float> y;
		//ds.from_libsvm("../data/fsg/test.10k.txt",y);
		ds.from_libsvm("../data/SCS/train.svm",y);

		Forest<FeaType> forest;
		forest.parse(fopen("model.txt","r"));

		std::vector<float> prob(y.size());
		qlog_info("[%s] Start predicting..\n",qlib::timestr().c_str());
		for(size_t i=0;i<ds.size();i++)
			prob[i] = 1/(1+std::exp(-forest.predict(ds[i])));
		qlog_info("[%s] Done predicting.\n",qlib::timestr().c_str());
		printf("AUC: %lf\n",calc_auc(&y[0], &prob[0], y.size()));
	}
	return 0;
}

