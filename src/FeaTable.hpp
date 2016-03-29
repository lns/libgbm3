#pragma once

#include <algorithm> // std::sort
#include <cstdint>
#include <vector>
#include <unordered_map>
#include "qlog.hpp"
#include "cpu_timer.hpp"
#include "DataSheet.hpp"

class FTEntry {
public:
	uint32_t _row;
	float _val;
	FTEntry(uint32_t r, float v): _row(r), _val(v) {}
};

/**
 * Feature Table
 *
 * ft[fea] = vector<Entry>
 * Each column is sorted by value in descending (XXX) order
 */
template<typename FeaType>
class FeaTable: public std::unordered_map<FeaType, std::vector<FTEntry> > {
public:
	class Comp {
	public:
		inline bool operator()(const FTEntry& a, const FTEntry& b) const {
			return (a._val > b._val or (a._val == b._val and a._row < b._row));
		}
	};

	// Sort the entries in ft
	inline void sort() {
		std::vector<std::vector<FTEntry>*> tasks;
		for(auto& it: (*this))
			tasks.push_back(&it.second);
		qlog_info("[%s] Sorting ...\n",qstrtime());
		#pragma omp parallel for
		for(size_t i=0;i<tasks.size();i++)
			std::sort(tasks[i]->begin(), tasks[i]->end(), Comp());
		qlog_info("[%s] Done.\n",qstrtime());
	}

	// Read from a DataSheet
	inline void from_datasheet(const DataSheet<FeaType>& ds) {
		this->clear();
		qlog_info("[%s] Reading from datasheet ...\n",qstrtime());
		for(size_t i=0; i<ds.size(); i++)
			for(auto&x: ds[i]) {
				(*this)[x.first].push_back(FTEntry(i,x.second));
			}
		qlog_info("[%s] Done.\n",qstrtime());
	}

	// Read from a libsvm text file
	template<typename T>
	inline void from_libsvm(const char * file_name, std::vector<T>& out_y) {
		this->clear();
		out_y.clear();
		qlog_info("[%s] Reading from libsvm file '%s' ...\n",qstrtime(),file_name);
		std::ifstream ifs(file_name);
		std::string line;
		size_t r = 0;
		while(getline(ifs,line)) {
			out_y.push_back(qstrtod(line.c_str(),nullptr));
			for(const char * head = qnextok(line.c_str());
					head != line.c_str()+line.size(); head = qnextok(head)) {
				char * q = strchr(head,':');
				if(not q) {
					qlog_warning("Parsing token failed: %s\n",
							std::string(head,qnextok(head)).c_str());
					continue;
				}
				FeaType fea = parse_cstr<FeaType>(head,q);
				float val = qstrtod(q+1,nullptr);
				(*this)[fea].push_back(FTEntry(r,val));
			}
			++r;
		}
		qlog_info("[%s] Done. nr: %ld, nc: %ld\n",qstrtime(),r,this->size());
	}

	// print info
	inline void print() const {
		qlog_info("size(): %lu\n", this->size());
	}
};

