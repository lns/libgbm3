#include "DataSheet.hpp"
#include "GBM.hpp"
#include "auc.hpp"

int main(int argc, char* argv[]) {
	typedef uint32_t FeaType;
	Parameters param;
	param.eta = 1.0f;
	param.l1reg = 1.0f;
	param.l2reg = 1.0f;
	// Max number of nodes in a tree
	param.max_tree_node = 24;
	// Number of recent tree to search
	param.search_recent_tree = 1;
	// Max depth of node
	param.max_depth = 5;
	// Max number of trees
	param.max_trees = 200;
	// Max number of leaves
	param.max_leaves = 1000;
	// Stopping criteria: fabs(diff) < max(threshold, precision * fabs(value))
	param.update_thres = 1e-5;
	param.update_precs = 1e-3;
	/* TODO
	// Inner stopping criteria: loss reduction < max(threshold, precision * loss)
	param.inner_thres = 1e-5;
	param.inner_precs = 1e-3;
	*/
	// outer is on loss reduction
	// Outer stopping criteria: new split's loss reduction < max(threshold, precision * loss)
	param.outer_thres = 1e-7;
	param.outer_precs = 1e-4;
	//
	param.max_inner_iter = 0;
	param.min_node_weight = 10.0f;
	param.cut_thres = 1e-6;
	//
	if(true) { // Training
		GBM<FeaType> gbm(param);
		gbm._obj = new LogLoss<double>();
		//gbm.read_data_from_libsvm("../data/fsg/test.10k.txt");
		gbm.read_data_from_libsvm("../data/SCS/train.svm");
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
		qlog_info("[%s] Start predicting..\n",qstrtime());
		for(size_t i=0;i<ds.size();i++)
			prob[i] = 1/(1+std::exp(-forest.predict(ds[i])));
		qlog_info("[%s] Done predicting.\n",qstrtime());
		printf("AUC: %lf\n",calc_auc(&y[0], &prob[0], y.size()));
	}
	return 0;
}

