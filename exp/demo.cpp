#include "DataSheet.hpp"
#include "FeaTable.hpp"

#include "GBM.hpp"

int main(int argc, char* argv[]) {
	typedef uint32_t FeaType;
	Parameters param;
	param.eta = 1.0f;
	param.l1reg = 1.0f;
	param.l2reg = 1.0f;
	// Stopping criteria: fabs(diff) < inner_prec * (inner_thres+fabs(value))
	param.inner_thres = 1e-7;
	param.inner_precs = 1e-3;
	param.min_node_weight = 10.0f;
	GBM<FeaType> gbm(param);
	gbm._obj = new LogLoss<double>();
	/*
	DataSheet<FeaType> ds;
	ds.from_libsvm("../data/fsg/test.10k.txt",y);
	ft.from_datasheet(ds);
	ft.sort();
	auto it = ft.begin();
	for(auto&& each: it->second) {
		printf("%d %f\n",each._row,each._val);
	}
	*/
	gbm.read_data_from_libsvm("../data/fsg/test.10k.txt");
	gbm.assign_weights(); // all 1.0f
	gbm.update_stats();
	printf("loss: %le\n",gbm.loss());
	while(gbm.update_intercept_and_f())
		gbm.update_stats();
	printf("loss: %le\n",gbm.loss());
	// Find best cut
	NodeIndex ni(gbm._y.size(),0);
	Forest<FeaType> forest;
	forest.push_back(Tree<FeaType>());
	//
	forest[0].grow(-1, Node<FeaType>());
	if(gbm.update_tree_beta_and_f(forest[0], ni))
		// won't change f if intercept is good
		qlog_warning("intercept is not converged. "
				"Try a smaller inner_thres and inner_precs.\n");
	//gbm.update_intercept_and_f();
	//gbm.update_stats();
	forest[0].dbginfo();
	// Split root
	auto res = gbm.find_best_fea(forest[0], ni, 0);
	gbm.split(forest[0], 0, ni, res);
	for(int iter=0; iter<100; iter++) {
		bool updated = false;
		if(gbm.update_tree_beta_and_f(forest[0], ni)) {
			gbm.update_stats();
			updated = true;
		}
		if(gbm.update_intercept_and_f()) {
			gbm.update_stats();
			updated = true;
		}
		qlog_info("[%s] Refining: iter: %d, loss: %le\n",qstrtime(),iter,gbm.loss());
		forest[0].dbginfo();
		char c; scanf("%c",&c);
		if(not updated)
			break;
	}
	//
	res = gbm.find_best_fea(forest[0], ni, 1);
	delete gbm._obj;
	return 0;
}

