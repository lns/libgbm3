#include "DataSheet.hpp"
#include "FeaTable.hpp"

#include "GBM.hpp"

int main(int argc, char* argv[]) {
	typedef uint32_t FeaType;
	Parameters param;
	param.eta = 1.0f;
	param.l1reg = 1.0f;
	param.l2reg = 1.0f;
	// Stopping criteria: fabs(diff) < inner_prec * max(inner_thres, fabs(value))
	param.inner_thres = 1e-7;
	param.inner_precs = 1e-3;
	param.min_node_weight = 10.0f;
	GBM<FeaType> gbm(param);
	gbm._obj = new LogLoss<double>();
	//
	gbm.read_data_from_libsvm("../data/fsg/test.10k.txt");
	gbm.assign_weights(); // all 1.0f
	gbm.update_stats();
	printf("loss: %le\n",gbm.loss());
	while(gbm.update_intercept_and_f())
		gbm.update_stats();
	printf("loss: %le\n",gbm.loss());
	// Find best cut
	gbm.add_new_tree();
	gbm._forest[0].dbginfo();
	// Split root
	auto res = gbm.find_best_fea(0, 0);
	gbm.split(res);
	gbm.update_tree_beta_and_f(0);
	gbm.refine(100);
	gbm.add_new_tree();
	res = gbm.find_best_fea(0, 1);
	res = gbm.find_best_fea(0, 2);
	res = gbm.find_best_fea(1, 0);
	// teardown
	delete gbm._obj;
	return 0;
}

