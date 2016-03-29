#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include "qlog.hpp"
#include "cpu_timer.hpp"
#include "qstdlib.hpp"
#include "util.hpp"

// ds[row][fea] = val
template<typename FeaType>
class DataSheet: public std::vector<std::unordered_map<FeaType, float> > {
public:
	// print info
	inline void print(FILE * fo = stdout) const {
		fprintf(fo, "nrow: %ld\n", this->size());
	}
	// Read from libsvm format txt
	inline void from_libsvm(const char * file_name, std::vector<float>& out_y);
};

template<typename FeaType>
void DataSheet<FeaType>::from_libsvm(const char * file_name, 
		std::vector<float>& out_y) {
	this->clear();
	out_y.clear();
	std::ifstream ifs(file_name);
	std::string line;
	long nr = 0, nz = 0;
	qlog_info("[%s] Reading %s ...\n",qstrtime(),file_name);
	while(getline(ifs,line)) {
		this->push_back(std::unordered_map<FeaType, float>());
		auto& sample = (*this)[this->size()-1];
		++nr;
		const char * head = line.c_str();
		out_y.push_back(qstrtod(head,nullptr));
		for(const char * head = qnextok(line.c_str());
				head != line.c_str()+line.size(); head = qnextok(head)) {
			const char * q = strchr(head,':');
			if(not q) {
				qlog_warning("Wrong token at %s\n",head);
				continue;
			}
			FeaType fea = parse_cstr<FeaType>(head,q);
			float val = qstrtod(q+1,nullptr);
			sample[fea] = val;
			++nz;
		}
	}
	qlog_info("[%s] Done. nr: %ld, nz: %ld\n",qstrtime(),nr,nz);
}

