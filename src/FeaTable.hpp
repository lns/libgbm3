#pragma once

#include <algorithm> // std::sort
#include <cstdint>
#include <vector>
#include <unordered_map>
#include "qlog.hpp"
#include "cpu_timer.hpp"
#include "DataSheet.hpp"
#include "MP_FileReader.hpp"

class FTEntry {
public:
	uint32_t _row;
	float _val;
	FTEntry(): _row(0), _val(0.0f) {}
	FTEntry(uint32_t r, float v): _row(r), _val(v) {}
};

/**
 * Merge two sorted vectors. Save results to the left one.
 */
template<typename T, class Compare>
void merge_to_left(std::vector<T>& L, const std::vector<T>& R, Compare comp) {
	size_t old_size = L.size();
	L.resize(old_size + R.size());
	for(size_t i=0; i<R.size(); i++)
		L[old_size+i] = R[i];
#if SORT
	// merge
	std::inplace_merge(L.begin(), L.begin()+old_size, L.end(), comp);
#endif
}

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
			out_y.push_back(qstrtod(line.c_str(),nullptr)>0?1:0);
			for(const char * head = qlib::svm::next(line.c_str());
					head != nullptr; head = qlib::svm::next(head)) {
				const char * q = strchr(head,':');
				if(not q) {
					qlog_warning("Parsing token failed at row %ld: '%s'\n",r,head);
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

	// [MP] Read from a libsvm text file
	template<typename T>
	inline void from_libsvm_mp(const char * file_name, std::vector<T>& out_y,
			int n_threads = -1) {
		this->clear();
		out_y.clear();
		if(n_threads<=0)
			n_threads = std::min<int>(8,std::thread::hardware_concurrency()/2);
		class ThreadEnv {
		public:
			std::thread* th;
			FeaTable<FeaType>* ft;
			std::vector<std::pair<long,T>>* y;
			char _padding[64]; //todo: to be tested and optimized

			ThreadEnv() : th(nullptr) {}
		};
		MP_FileReader reader(file_name);
		std::vector<ThreadEnv> thread_env(n_threads);
		if(true) { // Read text file with multi-thread
			auto func = [&](int thread_id) {
				ThreadEnv& te = thread_env[thread_id];
				te.y = new std::vector<std::pair<long,T>>();
				if(thread_id==0) {
					te.ft = this;
				} else {
					te.ft = new FeaTable<FeaType>();
				}
				char * buf = nullptr;
				size_t len = 0;
				long row = 0;
				while(reader.readline(&buf, &len, &row)>0) {
					te.y->push_back(std::pair<long,T>(row,qstrtod(buf, nullptr)>0?1:0));
					for(const char * head = qlib::svm::next(buf);
							head != nullptr; head = qlib::svm::next(head)) {
						const char * q = strchr(head,':');
						if(not q) {
							qlog_warning("Parsing token failed at row %ld: '%s'\n",row,head);
							continue;
						}
						FeaType fea = parse_cstr<FeaType>(head,q);
						float val = qstrtod(q+1,nullptr);
						(*te.ft)[fea].push_back(FTEntry(row,val));
					}
				}
				free(buf);
				// sort
#if SORT
				for(const auto& it : (*te.ft)) {
					std::sort(it.second.begin(), it.second.end(), Comp());
				}
#endif
			};
			qlog_info("[%s] Reading from libsvm file '%s' ...\n",qstrtime(),file_name);
			for(int i=0;i<n_threads;i++) {
				thread_env[i].th = new std::thread(func, i);
			}
			for(int i=0;i<n_threads;i++) {
				thread_env[i].th->join();
				delete thread_env[i].th;
			}
			qlog_info("[%s] Done reading and sorting. Now merging ...\n",qstrtime());
		}
		if(true) { // Merge
			// Merge two sorted vector to the left one.
			auto func = [&](int thread_id, int stride) {
				if(thread_id % (2*stride) == 0 and thread_id+stride < n_threads) {
					// Merge thread_id and thread_id+stride
					ThreadEnv& te = thread_env[thread_id];
					ThreadEnv& te2 = thread_env[thread_id+stride];
					// Merge y
					merge_to_left(*te.y, *te2.y,
							[](const std::pair<long,T>& a, const std::pair<long,T>& b) -> bool {
							return a.first < b.first; });
					// Merge FeaTable
					for(const auto& each : (*te2.ft)) {
						auto it = te.ft->find(each.first);
						if(it==te.ft->end()) {
							(*te.ft)[each.first] = each.second;
						} else {
							merge_to_left(it->second, each.second, Comp());
						}
					}
				}
			};
			for(int stride=1; stride<n_threads; stride*=2) {
				for(int i=0;i<n_threads;i++) {
					thread_env[i].th = new std::thread(func, i, stride);
				}
				for(int i=0;i<n_threads;i++) {
					thread_env[i].th->join();
					delete thread_env[i].th;
				}
			}
			// copy y
			out_y.resize(thread_env[0].y->size());
			for(size_t i=0;i<thread_env[0].y->size();i++)
				out_y[i] = (*thread_env[0].y)[i].second;
			delete thread_env[0].y;
			for(int i=1;i<n_threads;i++) { // omit thread_env[0]
				delete thread_env[i].ft;
				delete thread_env[i].y;
			}
			qlog_info("[%s] Done merging. nr: %ld, nc: %ld\n",qstrtime(),
					out_y.size(), this->size());
		}
	}

	// print info
	inline void print() const {
		qlog_info("#features: %lu\n", this->size());
	}
};

