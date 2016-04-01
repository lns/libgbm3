#pragma once

#include <unordered_map>
#include <vector>

template<typename FeaType>
class Node {
public:
	// Index
	int _self; // start from 0 as root
	int _left; // -1 indicating null
	int _right;
	int _depth; // 0 = root
	// Split
	FeaType _fea;
	double _cut;
	bool _miss_go_left;
	// Value
	double _pred; // prediction
	// Statistics
	double _sum_g; // gradient
	double _sum_h; // hessian
	double _sum_w; // sample weight

	Node() { clear(); }
	// reset a node
	inline void clear();
	// parse a node from text
	// @return error number and print to stdout
	inline int parse(const char * text, int hint_id = -1);
	// print a node
	inline void print(FILE * fo = stdout) const;
	// whether the node is empty (initialized)
	inline bool is_empty() const;
	// whether the node is a leaf node
	inline bool is_leaf() const;
	// whether the node is a branch node
	inline bool is_branch() const;
	// Print debug info
	inline void dbginfo() const;	
};

template<typename FeaType>
class Tree: public std::vector<Node<FeaType>> {
public:
	// parse a tree from a file stream
	// @return error number and print to stdout
	inline int parse(FILE * f);
	// make prediction based on node._pred
	// FeaType should be convertable to T. (i.e. sample.find(node._fea))
	template<typename T>
	inline double predict(const std::unordered_map<T, float>& sample,
			int node_id = 0) const;
	// make predictions on a FeaType, add results to f
	inline void predict(const FeaTable<FeaType>& ft, std::vector<double>& f) const;
	// [Recursive] print a tree (or subtree of node_id)
	inline void print(int node_id = 0, FILE * fo = stdout) const;
	// Print debug info
	inline void dbginfo() const;
	// Grow a tree with an existing node
	// @param parent: node_id of parent. (for grow a root, parent should be -1)
	// @return node_id in the tree
	inline int grow(int parent, const Node<FeaType>& node);	
	// [Recursive] Check the dependencies are correct. (depth, etc.)
	inline bool is_correct(int node_id = 0) const;
	// [Recursive] whether the tree is constructed completely.
	inline bool is_complete(int node_id = 0) const;
	// [Recursive] Prune nodes with both left and right _beta zero.
	inline bool prune(int node_id = 0);
};

template<typename FeaType>
class Forest: public std::vector<Tree<FeaType>> {
public:
	// parse a forest
	// @return error number and print to stdout
	inline int parse(FILE * f);
	// make raw prediction
	template<typename T>
	inline double predict(const std::unordered_map<T, float>& sample) const;
	// print a forest
	inline void print(FILE * fo = stdout) const;
};

#include "Tree-imp.hpp"

