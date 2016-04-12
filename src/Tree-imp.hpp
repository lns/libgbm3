#pragma once

#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <string>
#include "qlog.hpp"
#include "util.hpp"
#include "qstdlib.hpp" // to_string

template<typename F>
bool Node<F>::is_empty() const {
	return _depth==-1;
}

template<typename F>
bool Node<F>::is_branch() const {
	return (_depth>=0 and _left!=-1 and _right!=-1);
}

template<typename F>
bool Node<F>::is_leaf() const {
	return (_depth>=0 and _left==-1 and _right==-1);
}

template<typename F>
void Node<F>::clear() {
	_self = _left = _right = _depth = -1;
	_fea = F();
	_cut = _pred = _sum_g = _sum_h = _sum_w = 0.0f;
}

template<typename F>
void Node<F>::dbginfo() const {
	fprintf(stderr,"Node[%d]: L:%d R:%d depth:%d\n",_self,_left,_right,_depth);
	fprintf(stderr,"|Cut:'%s'<%le miss_go_to:%c\n",qlib::to_string(_fea).c_str(),
			_cut,_miss_go_left?'L':'R');
	fprintf(stderr,"|G:%le H:%le W:%le pred:%le\n",_sum_g,_sum_h,_sum_w,_pred);
}

template<typename F>
void Node<F>::print(FILE * fo) const {
	if(is_empty())
		fprintf(fo,"%d:EMPTY\n", _self);
	for(int i=0;i<_depth;i++)
		fprintf(fo,"\t");
	if(is_leaf())
		fprintf(fo,"%d:leaf=%g\n", _self, _pred);
	else if(is_branch())
		fprintf(fo,"%d:[%s<%g] yes=%d,no=%d,missing=%d\n",
				_self, qlib::to_string(_fea).c_str(), _cut, _left, _right,
				_miss_go_left ? _left : _right);
	else {
		qlog_warning("Incomplete node:\n");
		dbginfo();
	}
}

template<typename F>
int Node<F>::parse(const char * text, int hint_id) {
	clear();
	_depth = strspn(text, "\t"); // number of preceding tab
	char * p = NULL;
	_self = strtol(text, &p, 10); // node_id
	if(hint_id>=0 and _self!=hint_id) {
		qlog_warning("hint_id mismatch: (_self, hint_id)=(%d, %d)", _self, hint_id);
		return -1;
	}
	if (*p != ':') {
		qlog_warning("Expect ':' but got '%c'(%x)\n",*p,static_cast<unsigned>(*p));
		return -2;
	}
	p++;
	if (*p == '[') { // branch
		char * q = strchr(p + 1, '<');
		if (*q != '<') {
			qlog_warning("Expect '<' but got '%c'(%x)\n",*q,static_cast<unsigned>(*q));
			return -3;
		}
		_fea = parse_cstr<F>(p + 1, q);
		p = q + 1;
		_cut = strtod(p, &q);
		if (*q != ']') {
			qlog_warning("Expect ']' but got '%c'(%x)\n",*q,static_cast<unsigned>(*q));
			return -4;
		}
		int yes, no, miss;
		if (3 != sscanf(q + 2, "yes=%d,no=%d,missing=%d", &yes, &no, &miss)) {
			qlog_warning("Read 'yes no missing' error: '%s'\n",q+2);
			return -5;
		}
		_left = yes;
		_right = no;
		if (miss == yes)
			_miss_go_left = true;
		else if (miss == no)
			_miss_go_left = false;
		else {
			qlog_warning("Missing node %d not in (yes:%d,no:%d)\n", miss, yes, no);
			return -6;
		}
		return 0;
	} else if (*p=='l') { // leaf
		char * q = strchr(p, '=');
		if(!q) {
			qlog_warning("Cannot find '=' : %s\n",p);
			return -7;
		}
		_pred = strtod(q+1,NULL);
		return 0;
	}
	else {
		qlog_warning("Wrong format: %s\n",text);
		return -20;
	}
}

template<typename F>
void Tree<F>::print(int node_id, FILE * fo) const {
	const Node<F>& node = (*this)[node_id];
	node.print(fo);
	if (node.is_branch()) {
		print(node._left,fo);
		print(node._right,fo);
	}
}

template<typename F>
void Tree<F>::dbginfo() const {
	fprintf(stderr, "Tree of size: %ld\n",this->size());
	for(size_t i=0;i<this->size();++i) {
		fprintf(stderr, "tree[%ld]:\n", i);
		(*this)[i].dbginfo();
	}
}

template<typename F>
int Tree<F>::grow(int parent, const Node<F>& in_node) {
	int node_id = this->size();
	(*this).push_back(in_node);
	Node<F>& node = (*this)[node_id];
	node._self = node_id;
	if(parent==-1)
		node._depth = 0;
	else
		node._depth = (*this)[parent]._depth+1;
	return node_id;
}

template<typename F> template<typename T>
double Tree<F>::predict(const std::unordered_map<T, float>& sample,
		int node_id) const {
	const Node<F>& node = (*this)[node_id];
	if (node.is_leaf())
		return node._pred;
	auto it = sample.find(node._fea);
	if (it == sample.end())
		return predict(sample, (node._miss_go_left ? node._left : node._right));
	if (it->second < node._cut)
		return predict(sample, node._left);
	else
		return predict(sample, node._right);
}

template<typename F>
void Tree<F>::predict(const FeaTable<F>& ft,
		std::vector<double>& f) const {
	std::vector<int> ni(f.size(),0);
	// we can rely on the fact that parent appears before children
	for(auto&& node : (*this)) {
		if(node.is_empty())
			continue;
		if(node.is_leaf()) { // make prediction
			for(size_t i=0;i<f.size();i++) {
				if(ni[i]==node._self) {
					f[i] += node._pred;
				}
			}
		}
		if(node.is_branch()) {
			auto it = ft.find(node._fea);
			if(it!=ft.end()) {
				const auto& fea_vec = it->second;
				for(auto&&entry : fea_vec) {
					if(ni[entry._row]!=node._self)
						continue;
					if(entry._val < node._cut)
						ni[entry._row] = node._left;
					else
						ni[entry._row] = node._right;
				}
			} else { // feature not found
				qlog_warning("Feature '%s' not found.\n",qlib::to_string(node._fea).c_str());
				node.dbginfo();
			}
			int miss = node._miss_go_left ? node._left : node._right;
			for(auto&x: ni)
				if(x==node._self)
					x = miss;
		}
	}
}

template<typename F>
bool Tree<F>::is_correct(int node_id) const {
	if(node_id>this->size()) {
		qlog_warning("Illegal node_id > size(): %d > %ld\n",node_id,this->size());
		return false;
	}
	const Node<F>& node = (*this)[node_id];
	if (node.is_empty())
		return true;
	bool test = true;
	// depth
	if (node._self==0 and node._depth!=0) {
		qlog_warning("Wrong depth for root.\n");
		node.dbginfo();
		return false;
	}
	if (node.is_leaf())
		return true;
	if (not node.is_branch()) {
		qlog_warning("Expect branching node.\n");
		node.dbginfo();
		return false;
	}
	std::vector<int> childs_id(2);
	childs_id[0] = node._left; childs_id[1] = node._right;
	for(auto child_id: childs_id) {
		if(child_id >= this->size()) {
			qlog_warning("Illegal node_id >= size(): %d > %ld\n",child_id, this->size());
			node.dbginfo();
			return false;
		}
		auto child = (*this)[child_id];
		if(child._depth != node._depth+1) {
			qlog_warning("Wrong depth for child.\n");
			node.dbginfo();
			child.dbginfo();
			return false;
		}
		test = test and is_correct(child_id);
	}
	return test;
}

template<typename F>
bool Tree<F>::is_complete(int node_id) const {
	if(node_id >= this->size()) {
		qlog_warning("Illegal node_id >= size(): %d > %ld\n",node_id,this->size());
		return false;
	}
	const Node<F>& node = (*this)[node_id];
	if (node.is_empty())
		return false;
	if (node.is_leaf())
		return true;
	else {
		bool test = true;
		if(node._left > node._self) {
			test = test and is_complete(node._left);
		} else {
			qlog_warning("node._left <= node._self : %d <= %d\n",node._left,node._self);
			node.dbginfo();
		}
		if(node._right > node._self) {
			test = test and is_complete(node._right);
		} else {
			qlog_warning("node._right <= node._self : %d <= %d\n",node._right,node._self);
			node.dbginfo();
		}
		return test;
	}
}

template<typename F>
int Tree<F>::parse(FILE * f) {
	size_t buffer_size = 1024;
	char * buffer = (char*)malloc(buffer_size * sizeof(char));
	this->clear();
	do { // Skip comments (line starts with a '#')
		if (getline(&buffer, &buffer_size, f) <= 0) {
			free(buffer);
			return 0;
		}
	} while ( buffer[0]=='#' );
	int tree_id = -1;
	if (1 != sscanf(buffer, "booster[%d]:", &tree_id)) {
		qlog_warning("Parse tree header failed: '%s'.\n",buffer);
		free(buffer);
		return -1;
	}
	this->push_back(Node<F>());
	//size_t n_parse = 0;
	while(not is_complete()) {
		if (getline(&buffer, &buffer_size, f) <= 0) {
			qlog_warning("Pre-mature EOF.\n");
			free(buffer);
			return -10;
		}
		Node<F> node;
		int ret = node.parse(buffer); // no check on node_id
		if(ret) {
			free(buffer);
			return ret;
		}
		if(static_cast<size_t>(node._self)>=this->size()) {
			qlog_warning("Wrong node id: %d\n",node._self);
			node.dbginfo();
			free(buffer);
			return -11;
		}
		if(not (*this)[node._self].is_empty()) {
			qlog_warning("Duplicated node id: %d\n",node._self);
			(*this)[node._self].dbginfo();
			node.dbginfo();
			free(buffer);
			return -12;
		}
		(*this)[node._self] = node;
		//n_parse++;
		if (node.is_branch()) {
			int bigger = std::max(node._left, node._right);
			this->resize(std::max(static_cast<int>(this->size()), 1+bigger));
		}
	}
	free(buffer);
	return 0;
}

template<typename F>
void Forest<F>::print(FILE * fo) const {
	for (size_t i=0; i < this->size(); i++) {
		fprintf(fo,"booster[%lu]:\n", i);
		(*this)[i].print(0,fo);
	}
}

template<typename F> template<typename T>
double Forest<F>::predict(const std::unordered_map<T, float>& sample) const {
	double s = 0;
	for (size_t i=0; i < this->size(); i++)
		s += (*this)[i].predict(sample);
	return s;
}

template<typename F>
int Forest<F>::parse(FILE * f) {
	this->clear();
	Tree<F> tree;
	while (true) {
		int ret = tree.parse(f);
		if (ret != 0) {
			qlog_warning("Tree Parse Error: %d\n", ret);
			return ret;
		}
		if (tree.size() == 0)
			break;
		this->push_back(tree);
	}
	return 0;
}

