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

/**
 * Sample Statistics
 */
class Stats {
public:
	double _g; // sample's current gradient
	double _h; // sample's current hessian
	double _w; // sample's weight

	Stats() : _g(0.0f), _h(0.0f), _w(0.0f) {}

	inline Stats& operator+=(const Stats& rhs) {
		_g += rhs._g;
		_h += rhs._h;
		_w += rhs._w;
		return *this;
	}

	inline friend Stats operator+(Stats lhs, const Stats& rhs) {
		lhs += rhs;
		return lhs;
	}

	void print() const { printf("g: %le, h: %le, w: %le\n",_g,_h,_w); }
};

// NodeIdType should be a int type, for safety it should be signed.
typedef int8_t NodeIdType;
typedef std::vector<NodeIdType> NodeIndex;

/**
 * A Cut Structure
 */
template<typename FeaType>
class Cut {
public:
	int tree_id;
	NodeIdType node_id;
	FeaType fea;
	double cut; // cut val
	double gain; // decreasing of regularized loss
	double pred_L; // best left pred found by argmin reged approx loss
	double pred_R; // best right pred found by argmin reged approx loss
	bool miss_go_left;

	Cut() : tree_id(-1), node_id(-1), fea(), cut(0.0f), gain(0.0f), 
		pred_L(0.0f), pred_R(0.0f), miss_go_left(false) {}

	void print() const {
		printf("Forest[%d][%d]: ['%s'<%le] L:%le R:%le miss:%c gain:%le\n",
				tree_id, node_id, qlib::to_string(fea).c_str(),
				cut, pred_L, pred_R, miss_go_left?'L':'R', gain);
	}
};

/**
 * Compare of cuts
 */
template<typename FeaType>
class CutComp {
public:
	inline bool operator()(const Cut<FeaType>& a, const Cut<FeaType>& b) {
		return (a.gain > b.gain);
	}
};

/**
 * Greedy Boosting Machine
 */
template<typename FeaType>
class GBM {
public:
	std::random_device _rand_dev;
	std::mt19937 _rand_gen;
	//
	const Parameters& _param;
	// Objective
	const Objective<double>* _obj;
	// Training file FeaTable
	const FeaTable<FeaType>* _ft;
	// Training file targets
	const std::vector<double>* _y;

	/**
	 * f is tree output
	 * g is weight-adjusted gradient
	 * h is weight-adjusted hessian
	 * w is weight, with mean = 1
	 * (this is the weight used for training, and is not directly provided by data.)
	 */
	// raw f
	std::vector<double> _f;
	// g, h, and w
	std::vector<Stats> _stats;
	// raw loss
	std::vector<double> _loss;
	// sample weight (provided by data, may not equal to _stats._w)
	// (i.e. sample-weight-weighted-loss is the optimization target)
	std::vector<double> _sample_weight;
	
	/**
	 * For evaluation on test set
	 */
	const FeaTable<FeaType>* _test_ft;
	const std::vector<double>* _test_y;
	std::vector<double> _test_f;
	std::vector<double> _test_loss;

	/**
	 * Model
	 */
	double _intercept; // do we really need an intercept? (avoid reg on intercept)
	double _recent_loss; // last computed loss
	Forest<FeaType> _forest;
	std::vector<NodeIndex> _vec_ni;

	GBM(const Parameters& param) : _rand_dev(), _rand_gen(_rand_dev()),
		_param(param), _obj(nullptr), _ft(nullptr), _y(nullptr),
		_test_ft(nullptr), _test_y(nullptr)
	{
		if(_param.objective=="LogLoss")
			_obj = new LogLoss<double>();
		else
			qlog_warning("Unknown objective:'%s'\n",_param.objective.c_str());
	}

	~GBM() {
		if(_obj)
			delete _obj;
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
	 * Set train data
	 */
	inline void set_train_data(const FeaTable<FeaType>& ft,
		const std::vector<double>& y);

	/**
	 * Set test data
	 */
	inline void set_test_data(const FeaTable<FeaType>& ft,
		const std::vector<double>& y);

	/**
	 * Add a new tree with root node.
	 * And update tree sum_g,h, node._pred and f
	 * @return whether f is altered
	 */
	inline bool add_new_tree();

	/**
	 * Assign sample weights
	 */
	inline void assign_weights(bool do_balance = false);

	/**
	 * Compute _g, _h from _y, _f and _w
	 */
	inline void update_stats();

	/**
	 * Get sum of stats, based on _stats._g and _h. 
	 * should be called after update_stats().
	 */
	inline Stats sum_stats() const;

	/**
	 * Get loss, based on _y and _f (and _pred in _forest for reg)
	 */
	inline double loss(bool including_reg = true) const;

	/**
	 * update the intercept (one newton step), according to _stats._g _h
	 * @return whether f is altered.
	 */
	inline bool update_intercept_and_f();

	/**
	 * Find Best Feature to split
	 */
	inline Cut<FeaType> find_best_fea(int tree_id, NodeIdType node_id,
			double l2reg, double l1reg) const;

	/**
	 * Find Best Cut for a feature
	 */
	inline Cut<FeaType> find_best_cut(const FeaType& fea, const Tree<FeaType>& tree,
			const NodeIndex& ni, NodeIdType node_id, double l2reg, double l1reg) const;

	/**
	 * When g,h,w in _stats are changed, or tree changed,
	 * update sum_g,h,w of each node in the tree,
	 * if loss can be decreased, update pred of each node, also update global f.
	 * @return whether f is altered (which means beta is also altered)
	 */
	inline bool update_tree_pred_and_f(int tree_id);

	/**
	 * Get train set AUC
	 */
	inline double get_auc() const;

	/**
	 * Update test f
	 */
	inline void update_test_f();

	/**
	 * Get test loss
	 */
	inline double get_test_loss() const;

	/**
	 * Get test AUC
	 */
	inline double get_test_auc() const;

	/**
	 * Refine weight (fully corrective update)
	 * todo: Try to refine cut_val as well
	 * @return number of iteration
	 */
	inline int refine(int max_iter);

	/**
	 * Split a node (usually a leaf) with Cut found
	 */
	inline void split(const Cut<FeaType>& cut);

	/**
	 * Boost the forest
	 */
	inline void boost();

	/**
	 * Output the model
	 */
	inline Forest<FeaType> output_model() const;

	/**
	 * Save the model
	 */
	inline void save_to_file(const char * file_name) const;

	/**
	 * Check if loss is decreasing.
	 */
	inline bool check_loss(bool omit_test = false);

	/**
	 * Check f, return true if different
	 */
	inline bool check_f() const;

	/**
	 * Check stats, return true if different
	 */
	inline bool check_stats() const;
};

#include "GBM-imp.hpp"

