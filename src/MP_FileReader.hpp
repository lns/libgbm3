#pragma once

#include <thread>
#include <mutex>
#include <cstdio>

class MP_FileReader {
public:
	FILE * f;
	std::mutex f_mutex;
	long row;

	MP_FileReader(const char * filename) : f(nullptr), row(-1) {
		f = fopen(filename, "r");
		if(not f) {
			fprintf(stderr, "[ERROR] Cannot open '%s' for reading.\n", filename);
			abort();
		}
	}

	~MP_FileReader() {
		if(f) {
			fclose(f);
		}
	}

	/**
	 * Thread-safe line-oriented file reader.
	 * Return characters read into buf, *row_addr set to row number (start from 0)
	 */
	template<typename T>
	ssize_t readline(char ** buf_addr, size_t * buf_len_addr, T * row_addr) {
		ssize_t ret = 0;
		std::lock_guard<std::mutex> lock(f_mutex);
		ret = getline(buf_addr, buf_len_addr, f);
		if(ret>0)
			row++;
		*row_addr = row;
		return ret;
	}

};

/*
int main(int argc, char* argv[]) {
	MP_FileReader reader(argv[1]);
	std::thread* t[32];
	size_t row[32] = {0};
	auto func = [&](int i) {
		char * buf = nullptr;
		size_t len = 0;
		while(reader.readline(&buf, &len, &row[i])>0);
		if(buf)
			free(buf);
	};
	const int n_thread = 4;
	for(int i=0;i<n_thread;i++) {
		t[i] = new std::thread(func, i);
	}
	for(int i=0;i<n_thread;i++) {
		t[i]->join();
	}
	for(int i=0;i<n_thread;i++) {
		printf("%lu\n",row[i]+1);
	}
	for(int i=0;i<n_thread;i++) {
		delete t[i];
	}
	return 0;
}
*/

