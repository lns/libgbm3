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
#include "util.hpp"

class Stats {
public:
	double _g;
	double _h;
	double _w;

	Stats() : _g(0.0f), _h(0.0f), _w(0.0f) {}

	inline Stats& operator+=(const Stats& rhs) {
		_g += rhs._g;
		_h += rhs._h;
		_w += rhs._w;
		return *this;
	}

	inline friend Stats operator+(Stats lhs, const Stats& rhs)
	{
		lhs += rhs;
		return lhs;
	}

	void print() const { printf("g: %le, h: %le, w: %le\n",_g,_h,_w); }
};

// NodeIdType should be a int type, for safety it should be signed.
typedef int8_t NodeIdType;
typedef std::vector<NodeIdType> NodeIndex;

template<typename FeaType>
class Cut {
public:
	int tree_id;
	NodeIdType node_id;
	FeaType fea;
	double cut; // cut val
	double gain; // decreasing of regularized loss
	double pred_L;
	double pred_R;
	bool miss_go_left;

	Cut() : tree_id(-1), node_id(-1), fea(), cut(0.0f), gain(0.0f), 
		pred_L(0.0f), pred_R(0.0f), miss_go_left(false) {}

	void print() const {
		printf("Forest[%d][%d]: ['%s'<%le] L:%le R:%le miss:%c gain:%le\n",
				tree_id, node_id, qlib::to_string(fea).c_str(),
				cut, pred_L, pred_R, miss_go_left?'L':'R', gain);
	}
};

template<typename FeaType>
class CutComp {
public:
	inline bool operator()(const Cut<FeaType>& a, const Cut<FeaType>& b) {
		return (a.gain > b.gain or (a.gain==b.gain 
					and (a.tree_id < b.tree_id or (a.tree_id==b.tree_id
							and a.node_id < b.node_id))));
	}
};

template<typename FeaType>
class GBM {
public:
	std::random_device _rand_dev;
	std::mt19937 _rand_gen;
	//
	const Parameters& _param;
	Objective<double> * _obj;
	FeaTable<FeaType> _ft;
	//DataSheet<uint32_t> _ds;

	/**
	 * g is weight-adjusted gradient
	 * h is weight-adjusted hessian
	 * w is weight, with mean = 1
	 * (this is the weight used for training, and is not directly provided by data.)
	 */
	std::vector<Stats> _stats;
	// raw y
	std::vector<double> _y;
	// raw f
	std::vector<double> _f;
	// raw loss
	std::vector<double> _loss;

	// sample weight (provided by data, may not equal to _stats._w)
	// (i.e. sample-weight-weighted-loss is the optimization target)
	std::vector<double> _sample_weight;

	/**
	 * Model
	 */
	double _intercept;
	Forest<FeaType> _forest;
	std::vector<NodeIndex> _vec_ni;

	GBM(const Parameters& param) : _rand_dev(), _rand_gen(_rand_dev()),
		_param(param) {}

	// Read data from libsvm format file
	inline void read_data_from_libsvm(const char * file_name) {
		_ft.from_libsvm(file_name, _y);
		_ft.sort();
		_stats.resize(_y.size());
		_f.resize(_y.size());
		_loss.resize(_y.size());
		_sample_weight.resize(_y.size(),1.0f);
	}

	/**
	 * Add a new tree with root node.
	 * And update tree sum_g,h and beta and f
	 */
	inline void add_new_tree() {
		_forest.push_back(Tree<FeaType>());
		_forest.back().grow(-1, Node<FeaType>());
		_vec_ni.push_back(NodeIndex(_stats.size(), 0));
		// todo: This can be further optimized,
		// since only root's sum_g,h are updated.
		if(update_tree_beta_and_f(_forest.size()-1))
			qlog_warning("intercept is not converged. "
					"Try a smllaer update_thres and update_precs.\n");
	}

	/**
	 * Assign weights
	 */
	inline void assign_weights(bool do_balance = false) {
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
			for(size_t i=0;i<_y.size();++i) {
				if(_y[i]>0.5)
					sum_positive_w += _sample_weight[i];
				else
					sum_negative_w += _sample_weight[i];
			}
			if(sum_positive_w<=0 or sum_negative_w<=0)
				qlog_error("sample_weight error! (%le,%le)\n",
						sum_positive_w, sum_negative_w);
			double scale_positive = 0.5*_y.size()/sum_positive_w;
			double scale_negative = 0.5*_y.size()/sum_negative_w;
			for(size_t i=0;i<_stats.size();++i) {
				if(_y[i]>0.5)
					_stats[i]._w = scale_positive * _sample_weight[i];
				else
					_stats[i]._w = scale_negative * _sample_weight[i];
			}
		}
	}

	/**
	 * Update _g, _h and using _y, _f and _w
	 */
	inline void update_stats() {
		if(_y.size()==0)
			return;
		_obj->FirstOrder(&_y[0], 1, &_f[0], 1, _y.size(),
				&_stats[0]._g, sizeof(_stats[0])/sizeof(_stats[0]._g));
		_obj->SecondOrder(&_y[0], 1, &_f[0], 1, _y.size(),
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
	inline Stats sum_stats() const {
		Stats res;
		res._g = res._h = res._w = 0.0f;
		for(auto&&s: _stats)
			res += s;
		return res;
	}

	/**
	 * Get loss, based on _y and _f
	 */
	inline double loss(bool including_reg = true) {
		double res = 0;
		_obj->Loss(&_y[0], 1, &_f[0], 1, _y.size(), &_loss[0], 1);
		for(size_t i=0;i<_loss.size();++i)
			res += _loss[i]*_sample_weight[i];
		if(including_reg)
			for(auto&&tree : _forest)
				for(auto&&node : tree) {
					res += 0.5*_param.l2reg_base*std::pow(_param.l2reg_gamma, node._depth)
						* node._beta * node._beta;
					res += _param.l1reg_base*std::pow(_param.l1reg_gamma, node._depth)
						* fabs(node._beta);
				}
		return res;
	}

	/**
	 * update the intercept (one newton step), according to _stats._g _h
	 * @return whether f is altered.
	 */
	inline bool update_intercept_and_f() {
		Stats res;
		res = sum_stats();
		double diff = (argmin_reg_loss<double>(res._h, res._g, 0, 1e-6, _intercept));
		//qlog_warning("intercept diff:%le\n",diff);
		if(fabs(diff)<_param.update_precs*
				std::max(_param.update_thres,fabs(_intercept)))
			return false;
		diff *= _param.eta;
		for(auto&f: _f)
			f += diff;
		_intercept += diff;
		return true;
	}

	/**
	 * Find Best Feature to split
	 */
	Cut<FeaType> find_best_fea(int tree_id, NodeIdType node_id,
			double l2reg, double l1reg) const {
		//qlog_info("[%s] find_best_fea() for forest[%d][%d] ...\n",
		//		qstrtime(),tree_id,node_id);
		const Tree<FeaType>& tree = _forest[tree_id];
		const NodeIndex& ni = _vec_ni[tree_id];
		std::vector<FeaType> features;
		features.reserve(_ft.size());
		for(auto&&it: _ft)
			features.push_back(it.first);
		std::vector<Cut<FeaType>> cuts(features.size());
		#pragma omp parallel for
		for(size_t i=0;i<features.size();i++) {
			cuts[i] = find_best_cut(features[i], tree, ni, node_id, l2reg, l1reg);
		}
		Cut<FeaType> best;
		for(auto&&c: cuts)
			if(c.gain > best.gain)
				best = c;
		//qlog_info("[%s] best_fea found:\n",qstrtime());
		best.tree_id = tree_id;
		best.node_id = node_id;
		//best.print();
		return best;
	}

	/**
	 * Find Best Cut for a feature
	 */
	inline Cut<FeaType> find_best_cut(const FeaType& fea, const Tree<FeaType>& tree,
			const NodeIndex& ni, NodeIdType node_id, double l2reg, double l1reg) const {
		Stats total;
		total._g = tree[node_id]._sum_g;
		total._h = tree[node_id]._sum_h;
		total._w = tree[node_id]._sum_w;
		const std::vector<FTEntry> fea_vec = _ft.find(fea)->second;
		size_t n = fea_vec.size();
		Cut<FeaType> best; // for return results
		best.fea = fea; // of course
		best.miss_go_left = true; // always
		Stats accum; // for accumulate stats
		const FTEntry *this_entry, *last_entry = &fea_vec[0];
		for(size_t i=1; i<n; last_entry=this_entry, i++) {
			this_entry = &fea_vec[i];
			auto r = last_entry->_row;
			// TODO: This can be optimized so that if ni[r]!=current node_id
			// the statistics can be accumulated to another candidates.
			// This is especially useful when a node is split into two
			// and we want to find best fea for these two nodes at the same time
			if(ni[r]!=node_id)
				continue;
			accum += _stats[r];
			if(last_entry->_val <= this_entry->_val+_param.cut_thres)
				continue;
			if(accum._w < _param.min_node_weight)
				continue;
			if(total._w - accum._w < _param.min_node_weight)
				break;
			double gain_R = -min_reg_loss<double>(accum._h, accum._g, l2reg, l1reg, 0);
			double gain_L = -min_reg_loss<double>(total._h-accum._h, total._g-accum._g, l2reg, l1reg, 0);
			if(gain_R+gain_L > best.gain) {
				best.cut = 0.5*(last_entry->_val + this_entry->_val);
				best.gain = gain_R+gain_L;
				best.pred_R = argmin_reg_loss<double>(accum._h, accum._g, l2reg, l1reg, 0);
				best.pred_L = argmin_reg_loss<double>(total._h-accum._h, total._g-accum._g, l2reg, l1reg, 0);
			}
		}
		return best;
	}

	/**
	 * When g,h,w in _stats are changed, update sum_g,h,w of each node in the tree,
	 * if loss can be decreased, update beta of each node, also update global f.
	 * @return whether f is altered (which means beta is also altered)
	 */
	inline bool update_tree_beta_and_f(int tree_id) {
		Tree<FeaType>& tree = _forest[tree_id];
		const NodeIndex& ni = _vec_ni[tree_id];
		for(auto&node: tree)
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
		// traverse in reverse order
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
		// update beta and global f
		bool f_altered = false;
		for(auto&node: tree) {
			// todo: With regularization, the root's diff should be zero.
			// however during optimization, the root's sum_g may not be zero.
			double diff = (argmin_reg_loss<double>(node._sum_h, node._sum_g,
												 _param.l2reg_base*std::pow(_param.l2reg_gamma, node._depth),
												 _param.l1reg_base*std::pow(_param.l1reg_gamma, node._depth),
												 node._beta));
			if(fabs(diff)<_param.update_precs*
					std::max(_param.update_thres,fabs(node._beta)))
				continue;
			//if(node._depth==0)
			//	qlog_warning("Updating root as intercept is not converged.\n");
			diff *= _param.eta;
			for(size_t i=0;i<ni.size();i++)
				if(ni[i]==node._self) // XXX: This should be if(ni[i] is offspring of node._self)
					_f[i] += diff;
			node._beta += diff;
			f_altered = true;
		}
		return f_altered;
	}

	/**
	 * Refine weight (fully corrective update)
	 * todo: Try to refine cut_val as well
	 * @return number of iteration
	 */
	inline int refine(int max_iter) {
		qlog_info("[%s] Refine: loss():%le\n",qstrtime(),loss());
		update_stats();
		int iter = 0;
		std::vector<size_t> indexes(_forest.size(),0);
		for(size_t i=0;i<indexes.size();i++)
			indexes[i] = i;
		for(; iter<max_iter; iter++) {
			bool updated = false;
			if(update_intercept_and_f()) {
				update_stats();
				updated = true;
			}
			//random order of update
			std::shuffle(indexes.begin(), indexes.end(), _rand_gen);
			for(auto&&i : indexes)
				if(update_tree_beta_and_f(i)) {
					update_stats();
					updated = true;
				}
			if(not updated)
				break;
		}
		qlog_info("[%s] Done refine[%d]: loss():%le\n",qstrtime(),iter,loss());
		return iter;
	}

	/**
	 * Split a node (usually a leaf) with Cut found
	 */
	inline void split(const Cut<FeaType>& cut) {
		Tree<FeaType>& tree = _forest[cut.tree_id];
		NodeIndex& ni = _vec_ni[cut.tree_id];
		if(not tree[cut.node_id].is_leaf()) {
			qlog_warning("Split a non-leaf node:\n");
			tree[cut.node_id].dbginfo();
			cut.print();
		}
		// 0. Check
		if(_ft.find(cut.fea)==_ft.end())
			qlog_error("Split with unknown feature: '%s'\n",
					qlib::to_string(cut.fea).c_str());
		// 1. Record cut info in current node
		tree[cut.node_id]._fea = cut.fea;
		tree[cut.node_id]._cut_val = cut.cut;
		tree[cut.node_id]._miss_go_left = cut.miss_go_left;
		Node<FeaType> L, R;
		L._depth = R._depth = tree[cut.node_id]._depth+1;
		// The predictions here won't be used. Normally they will be later updated by
		// update_tree_beta_and_f(), whose value should be same.
		L._beta = cut.pred_L;
		R._beta = cut.pred_R;
		// 2. Add to tree // We should not assign value while operate on the vector.
		int L_node_id = tree.grow(cut.node_id,L);
		tree[cut.node_id]._left = L_node_id;
		int R_node_id = tree.grow(cut.node_id,R);
		tree[cut.node_id]._right = R_node_id;
		// 3. Update nodeIndex
		// move sample from parent_id to children_id
		const std::vector<FTEntry>& fea_vec = _ft.find(cut.fea)->second;
		for(auto&&entry : fea_vec) {
			if(ni[entry._row]!=cut.node_id)
				continue;
			if(entry._val < cut.cut)
				ni[entry._row] = tree[cut.node_id]._left;
			else
				ni[entry._row] = tree[cut.node_id]._right;
		}
		int miss = cut.miss_go_left? tree[cut.node_id]._left : tree[cut.node_id]._right;
		for(auto&x: ni)
			if(x==cut.node_id)
				x = miss;
		// 4. (Optional) Check
		if(not tree.is_correct(cut.node_id))
			qlog_warning("Checking failed.\n");
	}

	class Task {
	public:
		int tree_id;
		NodeIdType node_id;
		double l2reg;
		double l1reg;

		Task(int tid, NodeIdType nid, double l2, double l1) :
			tree_id(tid), node_id(nid), l2reg(l2), l1reg(l1) {}
	};

	/**
	 * Boost the forest
	 */
	inline void boost() {
		refine(_param.max_inner_iter);
		for(size_t iter=0; ; iter++) {
			size_t n_leaves = 0;
			for(auto&&tree : _forest)
				for(auto&&node : tree)
					if(node.is_leaf())
						n_leaves ++;
			if(n_leaves>=_param.max_leaves)
				break;
			if(_forest.empty() or (_forest.back().size()>1 
						and _forest.size() <= _param.max_trees)) {
				printf("Add a new tree.\n");
				add_new_tree();
				continue;
			}
			std::vector<Task> tasks;
			for(size_t i=std::max<int>(0,_forest.size()-1-_param.search_recent_tree);
					i<_forest.size();i++) {
				auto tree = _forest[i];
				if(tree.size() >= _param.max_tree_node)
					continue;
				for(auto&&node : tree) {
					if(node.is_leaf() and node._depth<_param.max_depth)
						tasks.push_back(Task(i,node._self,
									_param.l2reg_base*std::pow(_param.l2reg_gamma, node._depth),
									_param.l1reg_base*std::pow(_param.l1reg_gamma, node._depth)));
				}
			}
			if(tasks.empty())
				break;
			std::vector<Cut<FeaType>> candidates(tasks.size(), Cut<FeaType>());
			#pragma omp parallel for
			for(size_t i=0; i<tasks.size(); i++) {
				candidates[i] = find_best_fea(tasks[i].tree_id, tasks[i].node_id, tasks[i].l2reg, tasks[i].l1reg);
			}
			std::sort(candidates.begin(), candidates.end(), CutComp<FeaType>());
			/*
			char c;
			printf("[ENTER to continue]");
			fflush(stdout);
			scanf("%c",&c);
			*/
			double current_loss = loss();
			printf("iter: %4zu, reg_loss: %le, loss: %le\n", iter,
					current_loss, loss(false));
			if(candidates[0].gain < std::max(_param.outer_thres, 
						_param.outer_precs*current_loss))
				break;
			// Use the cut found to split
			printf("best candidates gain: %le (approx.)\n",candidates[0].gain);
			split(candidates[0]);
			update_tree_beta_and_f(candidates[0].tree_id);
			printf("      +node reg_loss: %le, loss: %le\n", loss(), loss(false));
			refine(_param.max_inner_iter);
			// todo: prune zero node (if gain < threshold, instead of zero node)
			for(auto&tree: _forest) {
				while(tree.prune());
			}
		}
	}

	/**
	 * Output the model
	 */
	inline Forest<FeaType> output_model() const {
		Forest<FeaType> Woods = _forest;
		if(Woods.empty())
			return Woods;
		Woods.back()[0]._beta += _intercept;
		for(auto&tree : Woods)
			tree.update();
		return Woods;
	}

	/**
	 * Save the model
	 */
	inline void save_to_file(const char * file_name) const {
		Forest<FeaType> Woods = output_model();
		FILE * f = fopen(file_name, "w");
		//todo: print parameters
		Woods.print(f);
		fclose(f);
	}

};

