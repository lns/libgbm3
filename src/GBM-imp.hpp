#pragma once

#include <algorithm> // std::max()
#include <vector>
#include <cstdint>
#include <random>
#include "Objective.hpp"
#include "FeaTable.hpp"
#include "Tree.hpp"
#include "Parameters.hpp"
#include "qstdlib.hpp" // qlib::to_string
#include "qrand.hpp" // qlib::drand
#include "util.hpp"
#include "auc.hpp"

#if 0
#define CHECK_F() do{ if(not check_f()) {_forest.print(); qlog_error("check_f() failed.\n");}} while(false)
#define CHECK_STATS() do{ if(not check_stats()) {_forest.print(); qlog_error("check_stats() failed.\n");}} while(false)
#define CHECK_LOSS(omit) do{ if(not check_loss(omit)) {qlog_error("check_loss() failed.\n");}} while(false)
#else
#define CHECK_F()
#define CHECK_STATS()
#define CHECK_LOSS(x)
#endif

/**
 * Set train data
 */
template<typename FeaType>
inline void GBM<FeaType>::set_train_data(const FeaTable<FeaType>& ft,
		const std::vector<double>& y) {
	_ft = &ft;
	_y = &y;
	_stats.resize(_y->size());
	_f.resize(_y->size());
	_sample_weight.resize(_y->size(),1.0f);
}

/**
 * Set test data
 */
template<typename FeaType>
inline void GBM<FeaType>::set_test_data(const FeaTable<FeaType>& ft,
		const std::vector<double>& y) {
	_test_ft = &ft;
	_test_y = &y;
	_test_f.resize(_test_y->size());
}

/**
 * Add a new tree with root node.
 * And update tree sum_g,h and beta and f
 * @return whether f is altered
 */
template<typename FeaType>
inline bool GBM<FeaType>::add_new_tree() {
	_forest.push_back(Tree<FeaType>());
	_forest.back().grow(-1, Node<FeaType>());
	_vec_ni.push_back(NodeIndex(_stats.size(), 0));
	CHECK_F();
	CHECK_LOSS(true);
	CHECK_STATS();
	if(update_tree_pred_and_f(_forest.size()-1))
		return true;
	else
		return false;
}

/**
 * Assign weights
 */
template<typename FeaType>
inline void GBM<FeaType>::assign_weights(bool do_balance) {
	// At least two things can be done here:
	// 1. Reweight positive samples to balance the labels
	// 2. Set some weights to zero to do row sampling
	// (maybe we need another vector to mark for train/test and CV
	if(not do_balance)
		for(size_t i=0;i<_stats.size();++i)
			_stats[i]._w = _sample_weight[i];
	else {
		// count positive and negative weights
		double sum_positive_w = 0, sum_negative_w = 0;
		for(size_t i=0;i<_y->size();++i) {
			if(_y->at(i)>0.5)
				sum_positive_w += _sample_weight[i];
			else
				sum_negative_w += _sample_weight[i];
		}
		if(sum_positive_w<=0 or sum_negative_w<=0)
			qlog_error("sample_weight error! (%le,%le)\n",
					sum_positive_w, sum_negative_w);
		double scale_positive = 0.5*_y->size()/sum_positive_w;
		double scale_negative = 0.5*_y->size()/sum_negative_w;
		for(size_t i=0;i<_stats.size();++i) {
			if(_y->at(i)>0.5)
				_stats[i]._w = scale_positive * _sample_weight[i];
			else
				_stats[i]._w = scale_negative * _sample_weight[i];
		}
	}
}

/**
 * Compute _g, _h from _y, _f and _w
 */
template<typename FeaType>
inline void GBM<FeaType>::update_stats() {
	if(_y->size()==0)
		return;
	_obj->FirstOrder(_y->data(), 1, &_f[0], 1, _y->size(),
			&_stats[0]._g, sizeof(_stats[0])/sizeof(_stats[0]._g));
	_obj->SecondOrder(_y->data(), 1, &_f[0], 1, _y->size(),
			&_stats[0]._h, sizeof(_stats[0])/sizeof(_stats[0]._h));
	for(auto&&s: _stats) {
		s._g *= s._w;
		s._h *= s._w;
	}
}

/**
 * Get sum of stats, based on _stats._g and _h. 
 * should be called after update_stats().
 */
template<typename FeaType>
inline Stats GBM<FeaType>::sum_stats() const {
	CHECK_STATS();
	Stats res;
	res._g = res._h = res._w = 0.0f;
	for(const auto&s: _stats)
		res += s;
	return res;
}

/**
 * Get loss, based on _y and _f (and _pred in _forest for reg)
 */
template<typename FeaType>
inline double GBM<FeaType>::loss(bool including_reg) const {
	double res = 0;
	static std::vector<double> loss_vec;
	loss_vec.resize(_y->size());
	_obj->Loss(_y->data(), 1, &_f[0], 1, _y->size(), &loss_vec[0], 1);
	for(size_t i=0;i<loss_vec.size();++i)
		res += loss_vec[i]*_stats[i]._w; // weighted with _w
	if(including_reg)
		for(const auto& tree : _forest)
			for(const auto& node : tree)
				if(node.is_leaf()) {
					res += 0.5 * _param.l2reg * node._pred * node._pred;
					res += _param.l1reg * fabs(node._pred);
				}
	return res;
}

/**
 * update the intercept (one newton step), according to _stats._g _h
 * @return whether f is altered.
 */
template<typename FeaType>
inline bool GBM<FeaType>::update_intercept_and_f() {
	CHECK_F();
	CHECK_STATS();
	Stats res;
	res = sum_stats();
	const double H = res._h + _param.proximal_l2*res._w;
	const double G = res._g - _intercept*H;
	const double new_intercept = argmin_reg_loss<double>(H, G, 0, 1e-6);
	const double gain
		= reg_loss<double>(H, G, 0, 1e-6, _intercept) 
		- reg_loss<double>(H, G, 0, 1e-6, new_intercept);
	if(gain < _param.relative_tol * _recent_loss)
		return false;
	const double diff = _param.eta*(new_intercept - _intercept);
	CHECK_LOSS(true);
	for(auto&& f : _f)
		f += diff;
	_intercept += diff;
	CHECK_LOSS();
	return true;
}

/**
 * Find Best Feature to split
 */
template<typename FeaType>
Cut<FeaType> GBM<FeaType>::find_best_fea(int tree_id, NodeIdType node_id,
		double l2reg, double l1reg) const {
	//qlog_info("[%s] find_best_fea() for forest[%d][%d] ...\n",
	//		qlib::timestr().c_str(),tree_id,node_id);
	const Tree<FeaType>& tree = _forest[tree_id];
	const NodeIndex& ni = _vec_ni[tree_id];
	CHECK_STATS();
	std::vector<FeaType> features;
	features.reserve(_ft->size());
	for(const auto& it : *_ft)
		features.push_back(it.first);
	std::vector<Cut<FeaType>> cuts(features.size());
	#pragma omp parallel for
	for(size_t i=0;i<features.size();i++) {
		// todo: col_sampling: the use of a global PRNG may affect parallelization
		if(qlib::drand()<=_param.col_sampling)
			cuts[i] = find_best_cut(features[i], tree, ni, node_id, l2reg, l1reg);
	}
	Cut<FeaType> best;
	for(const auto& c: cuts)
		if(c.gain > best.gain)
			best = c;
	//qlog_info("[%s] best_fea found:\n",qlib::timestr().c_str());
	best.tree_id = tree_id;
	best.node_id = node_id;
	//best.print();
	return best;
}

/**
 * Find Best Cut for a feature
 */
template<typename FeaType>
inline Cut<FeaType> GBM<FeaType>::find_best_cut(const FeaType& fea,
		const Tree<FeaType>& tree, const NodeIndex& ni, NodeIdType node_id,
		double l2reg, double l1reg) const {
	Stats total;
	const Node<FeaType>& node = tree[node_id];
	total._g = node._sum_g;
	total._h = node._sum_h;
	total._w = node._sum_w;
	const std::vector<FTEntry>& fea_vec = _ft->find(fea)->second;
	size_t n = fea_vec.size();
	Cut<FeaType> best; // for return results
	best.fea = fea; // of course
	best.miss_go_left = true; // always
	Stats accum; // for accumulate stats
	const double H = total._h + _param.proximal_l2*total._w;
	const double G = total._g - node._pred*H;
	const double obj_cur = min_reg_loss<double>(H, G, l2reg, l1reg);
	for(size_t i=0; i<n; i++) {
		const auto r = fea_vec[i]._row;
		// TODO: This can be optimized so that if ni[r]!=current node_id
		// the statistics can be accumulated to another candidates.
		// This is especially useful when a node is split into two
		// and we want to find best fea for these two nodes at the same time
		if(ni[r]!=node_id)
			continue;
		accum += _stats[r];
		if(i<n-1 and fea_vec[i]._val <= fea_vec[i+1]._val+_param.cut_thres)
			continue;
		if(accum._w < _param.min_node_weight)
			continue;
		if(total._w - accum._w < _param.min_node_weight)
			break;
		const double HR = accum._h + _param.proximal_l2*accum._w;
		const double HL = total._h - accum._h + _param.proximal_l2*(total._w - accum._w);
		const double GR = accum._g - node._pred*HR;
		const double GL = total._g - accum._g - node._pred*HL;
		const double obj_R = min_reg_loss<double>(HR, GR, l2reg, l1reg);
		const double obj_L = min_reg_loss<double>(HL, GL, l2reg, l1reg);
		const double gain = obj_cur - (obj_R + obj_L);
		if(gain > best.gain) {
			if(i<n-1)
				best.cut = 0.5*(fea_vec[i]._val + fea_vec[i+1]._val);
			else
				best.cut = fea_vec[i]._val; //todo: fea_vec[i]._val - epsilon?
			best.gain = gain;
			//NOTE: this is not shrinked by _param.eta
			//And indeed not used.
			best.pred_R = argmin_reg_loss<double>(HR, GR, l2reg, l1reg);
			best.pred_L = argmin_reg_loss<double>(HL, GL, l2reg, l1reg);
		}
	}
	return best;
}

/**
 * When g,h,w in _stats are changed, or tree changed,
 * update sum_g,h,w of each node in the tree,
 * if loss can be decreased, update pred of each node, also update global f.
 * @return whether f is altered (which means beta is also altered)
 */
template<typename FeaType>
inline bool GBM<FeaType>::update_tree_pred_and_f(int tree_id) {
	CHECK_F();
	CHECK_STATS();
	Tree<FeaType>& tree = _forest[tree_id];
	const NodeIndex& ni = _vec_ni[tree_id];
	for(auto&& node : tree)
		if(not node.is_empty()) {
			node._sum_g = 0.0f;
			node._sum_h = 0.0f;
			node._sum_w = 0.0f;
		}
	for(size_t i=0;i<ni.size();++i) {
		if(ni[i]<0) // not used
			continue;
		tree[ni[i]]._sum_g += _stats[i]._g;
		tree[ni[i]]._sum_h += _stats[i]._h;
		tree[ni[i]]._sum_w += _stats[i]._w;
	}
	// Optional: traverse in reverse order
	// so that children are visited before their parent
	for(long i=tree.size()-1; i>=0; i--) {
		if(tree[i].is_branch()) {
			const auto& L = tree[tree[i]._left];
			const auto& R = tree[tree[i]._right];
			tree[i]._sum_g = L._sum_g + R._sum_g;
			tree[i]._sum_h = L._sum_h + R._sum_h;
			tree[i]._sum_w = L._sum_w + R._sum_w;
		}
	}
	// calculate diff of pred for leaf nodes
	bool f_altered = false;
	std::vector<double> leaf_diff(tree.size(), 0.0f);
	for(auto&& node : tree) {
		if(not node.is_leaf())
			continue;
		if(_param.check_loss)
			_recent_loss = loss(true);
		const double H = node._sum_h + _param.proximal_l2*node._sum_w;
		const double G = node._sum_g - node._pred*H;
		const double L2 = _param.l2reg;
		const double L1 = _param.l1reg;
		const double new_pred = argmin_reg_loss<double>(H, G, L2, L1);
		const double gain
			= reg_loss<double>(H, G, L2, L1, node._pred)
			- reg_loss<double>(H, G, L2, L1, new_pred);
		if(gain < _param.relative_tol * _recent_loss)
			continue;
		leaf_diff[node._self] = _param.eta*(new_pred - node._pred);
		node._pred += leaf_diff[node._self];
		f_altered = true;
	}
	if(f_altered) {
		for(size_t i=0;i<ni.size();i++)
			_f[i] += ni[i]>=0 ? leaf_diff[ni[i]] : 0;
	}
	if(_param.check_loss) {
		double loss_after = loss(true);
		if(loss_after > _recent_loss) {
			qlog_warning("Loss increased!\n"
					"Please try a larger proximal_l2 and/or smaller eta.\n");
			//todo: step-back when loss increased.
		}
		_recent_loss = loss_after;
	}
	return f_altered;
}

/**
 * Refine weight (fully corrective update)
 * todo: Try to refine cut_val as well
 * @return number of iteration
 */
template<typename FeaType>
inline int GBM<FeaType>::refine(int max_iter) {
	qlog_info("[%s] Refine: loss():%le\n",qlib::timestr().c_str(),loss());
	CHECK_F();
	update_stats();
	_recent_loss = loss(true);
	int iter = 0;
	std::vector<size_t> indexes(_forest.size(),0);
	for(size_t i=0;i<indexes.size();i++)
		indexes[i] = i;
	//std::shuffle(indexes.begin(), indexes.end(), _rand_gen);
	for(; iter<max_iter; iter++) {
		bool updated = false;
		if(update_intercept_and_f()) {
			update_stats();
			updated = true;
		}
		//random order of update
		for(const auto i : indexes) {
			if(update_tree_pred_and_f(i)) {
				update_stats();
				updated = true;
			}
		}
		_recent_loss = loss(true);
		if(not updated)
			break;
	}
	qlog_info("[%s] Done refine[%d]: loss():%le\n",qlib::timestr().c_str(),iter,loss());
	return iter;
}

/**
 * Split a node (usually a leaf) with Cut found
 */
template<typename FeaType>
inline void GBM<FeaType>::split(const Cut<FeaType>& cut) {
	CHECK_STATS();
	Tree<FeaType>& tree = _forest[cut.tree_id];
	NodeIndex& ni = _vec_ni[cut.tree_id];
	if(not tree[cut.node_id].is_leaf()) {
		qlog_warning("Split a non-leaf node:\n");
		tree[cut.node_id].dbginfo();
		cut.print();
	}
	// 0. Check
	if(_ft->find(cut.fea)==_ft->end())
		qlog_error("Split with unknown feature: '%s'\n",
				qlib::to_string(cut.fea).c_str());
	// 1. Record cut info in current node
	tree[cut.node_id]._fea = cut.fea;
	tree[cut.node_id]._cut = cut.cut;
	tree[cut.node_id]._miss_go_left = cut.miss_go_left;
	Node<FeaType> L, R;
	L._depth = R._depth = tree[cut.node_id]._depth+1;
	// These predictions should be remain identical to parent's prediction.
	// Which will be later updated in update_tree_pred_and_f()
	L._pred = tree[cut.node_id]._pred; //cut.pred_L;
	R._pred = tree[cut.node_id]._pred; //cut.pred_R;
	tree[cut.node_id]._pred = 0.0f; // for safety
	// 2. Add to tree // We should not assign value while operate on the vector.
	int L_node_id = tree.grow(cut.node_id,L);
	tree[cut.node_id]._left = L_node_id;
	int R_node_id = tree.grow(cut.node_id,R);
	tree[cut.node_id]._right = R_node_id;
	CHECK_STATS();
	// 3. Update nodeIndex
	CHECK_LOSS(true);
	// move sample from parent_id to children_id
	const std::vector<FTEntry>& fea_vec = _ft->find(cut.fea)->second;
	for(const auto& entry : fea_vec) {
		if(ni[entry._row]!=cut.node_id)
			continue;
		if(entry._val < cut.cut)
			ni[entry._row] = tree[cut.node_id]._left;
		else
			ni[entry._row] = tree[cut.node_id]._right;
	}
	int miss = cut.miss_go_left? tree[cut.node_id]._left : tree[cut.node_id]._right;
	for(auto&&x: ni)
		if(x==cut.node_id)
			x = miss;
	// 4. (Optional) Check
	CHECK_STATS();
	CHECK_LOSS();
	if(not tree.is_correct(cut.node_id))
		qlog_warning("Checking failed.\n");
}

/**
 * Boost the forest
 */
template<typename FeaType>
inline void GBM<FeaType>::boost() {
	refine(_param.max_inner_iter);
	for(size_t iter=0; ; iter++) {
		size_t n_leaves = 0;
		for(const auto& tree : _forest)
			for(const auto& node : tree)
				if(node.is_leaf())
					n_leaves ++;
		if(n_leaves>=_param.max_leaves) {
			qlog_info("[%s] max_leaves=%u reached. Stop.\n",
					qlib::timestr().c_str(),_param.max_leaves);
			break;
		}
		if(_forest.empty() or (_forest.back().size()>1 
					and _forest.size() <= _param.max_trees)) {
			qlog_info("Add a new tree.\n");
			if(add_new_tree()) // f maybe altered
				update_stats();
			refine(_param.max_inner_iter);
			continue;
		}
		std::vector<Task> tasks;
		for(size_t i=std::max<int>(0,_forest.size()-1-_param.search_recent_tree);
				i<_forest.size();i++) {
			const auto& tree = _forest[i];
			if(tree.size() >= _param.max_tree_node)
				continue;
			for(const auto& node : tree) {
				if(node.is_leaf() and node._depth<_param.max_depth)
					tasks.push_back(Task(i,node._self,_param.l2reg,_param.l1reg));
			}
		}
		if(tasks.empty()) {
			qlog_info("[%s] No split candidates. Stop.\n",qlib::timestr().c_str());
			break;
		}
		std::vector<Cut<FeaType>> candidates(tasks.size(), Cut<FeaType>());
		//#pragma omp parallel for
		for(size_t i=0; i<tasks.size(); i++) {
			candidates[i] = find_best_fea(tasks[i].tree_id, tasks[i].node_id, tasks[i].l2reg, tasks[i].l1reg);
		}
		std::sort(candidates.begin(), candidates.end(), CutComp<FeaType>());
		const double last_loss = _recent_loss;
		if(candidates[0].gain < _param.relative_tol * _recent_loss) {
			qlog_info("[%s] candidate's gain meets stop criteria: %g < %g. Stop.\n",
					qlib::timestr().c_str(), candidates[0].gain, _param.relative_tol*_recent_loss);
			break;
		}
		if(_ft->find(candidates[0].fea)==_ft->end()) {
			qlog_warning("Cannot find candidate's feature '%s' in feature table. Stop.\n",
					qlib::to_string(candidates[0].fea).c_str());
			break;
		}
		// Use the cut found to split
		qlog_info("best candidates gain: %le (approx.)\n",candidates[0].gain);
		split(candidates[0]);
		//CHECK_LOSS(); This will increase a little due to regularization
		//TODO: this update_tree_pred_and_f can be omitted,
		//by using cut.pred_L and pred_R instead of recomputing it.
		//Note that when node._pred is changed, _f should be changed as well.
		update_tree_pred_and_f(candidates[0].tree_id);
		update_stats();
		refine(_param.max_inner_iter);
		//printf("      +node reg_loss: %le, loss: %le\n", _recent_loss, loss(false));
		if(_param.check_loss and 
				last_loss - _recent_loss < _param.relative_tol * _recent_loss) {
			qlog_info("[%s] Loss reduction meets stop criteria. Stop.\n",qlib::timestr().c_str());
			break;
		}
		// update prediction on test set
		update_test_f();
		printf("\r%4zu %12le %12le %8lf %12le %8lf %5ld %5ld\n"
				"iter trn-reg-loss   trn-loss    trn-auc   tst-loss    tst-auc #tree #leaf",
				iter, _recent_loss, loss(false)/_y->size(), get_auc(),
				get_test_loss()/(_test_y?_test_y->size():1), get_test_auc(),
				_forest.size(), n_leaves);
		fflush(stdout);
	}
	printf("\n");
	fflush(stdout);
}

/**
 * Output the model
 */
template<typename FeaType>
inline Forest<FeaType> GBM<FeaType>::output_model() const {
	Forest<FeaType> Woods = _forest;
	if(Woods.empty())
		return Woods;
	for(auto&&node: Woods.back())
		if(node.is_leaf())
			node._pred += _intercept;
	return Woods;
}

/**
 * Save the model
 */
template<typename FeaType>
inline void GBM<FeaType>::save_to_file(const char * file_name) const {
	Forest<FeaType> Woods = output_model();
	FILE * f = fopen(file_name, "w");
	//todo: print eval result on test
	//1. print parameters
	_param.opr->set_file(f);
	_param.opr->print_file();
	//2. print model
	Woods.print(f);
	fclose(f);
}

/**
 * Update test f
 */
template<typename FeaType>
inline void GBM<FeaType>::update_test_f() {
	if(not _test_ft or not _test_y)
		return;
	_test_f.resize(_test_y->size());
	for(auto&& x: _test_f)
		x = _intercept;
	for(const auto& tree : _forest)
		tree.predict(*_test_ft, _test_f);
}

/**
 * Get test set loss
 * Should be called after update_test_f()
 */
template<typename FeaType>
inline double GBM<FeaType>::get_test_loss() const {
	if(not _test_ft or not _test_y)
		return std::nan("");
	static std::vector<double> test_loss;
	test_loss.resize(_test_y->size());
	_obj->Loss(_test_y->data(),1, &_test_f[0],1, _test_y->size(), &test_loss[0],1);	
	double res = 0;
	for(size_t i=0;i<test_loss.size();++i)
		res += test_loss[i];
	return res;
}

/**
 * Get train set auc based on _f
 */
template<typename FeaType>
inline double GBM<FeaType>::get_auc() const {
	return calc_auc(_y->data(), _f.data(), _f.size());
}

/**
 * Get test set auc
 * Should be called after update_test_f()
 */
template<typename FeaType>
inline double GBM<FeaType>::get_test_auc() const {
	if(not _test_ft or not _test_y)
		return std::nan("");
	return calc_auc(_test_y->data(), _test_f.data(), _test_f.size());
}

/**
 * Check if loss is decreasing.
 */
template<typename FeaType>
inline bool GBM<FeaType>::check_loss(bool omit_test) {
	static double last_loss = loss(true);
	double this_loss = loss(true);
	if(not omit_test and this_loss>last_loss) {
		qlog_info("Loss increased from %le to %le\n",last_loss,this_loss);
		return false;
	}
	else {
		last_loss = this_loss;
		return true;
	}
}

/**
 * Check f, return true if different
 */
template<typename FeaType>
inline bool GBM<FeaType>::check_f() const {
	std::vector<double> pred(_f.size(), 0);
	Forest<FeaType> forest = output_model();
	if(forest.empty()) {
		qlog_warning("output_model() is empty. Skip.\n");
		return true;
	}
	for(const auto& tree : forest) {
		tree.predict(*_ft, pred);
	}
	for(size_t i=0;i<_f.size();i++) {
		if(fabs(pred[i]-_f[i])>1e-6) {
			qlog_warning("[%ld]: pred!=_f: %le!=%le\nforest:\n",i,pred[i],_f[i]);
			forest.print();
			return false;
		}
	}
	return true;
}

/**
 * Check stats, return true if different
 */
template<typename FeaType>
inline bool GBM<FeaType>::check_stats() const {
	// check g and h
	for(size_t i=0;i<_stats.size();i++) {
		double p = 1/(1+exp(-_f[i]));
		double g = p-_y->at(i);
		double h = p*(1-p);
		if(i<5) {
			printf("p,g,h,_f,_g,_h: %le,%le,%le,%le,%le,%le\n",
					p,g,h,_f[i],_stats[i]._g,_stats[i]._h);
		}
		if(fabs(g-_stats[i]._g)>1e-6) {
			qlog_warning("[%ld]: g!=_g: %le!=%le\n",i,g,_stats[i]._g);
			return false;
		}
		if(fabs(h-_stats[i]._h)>1e-6) {
			qlog_warning("[%ld]: h!=_h: %le!=%le\n",i,h,_stats[i]._h);
			return false;
		}
	}
	return true;
}

