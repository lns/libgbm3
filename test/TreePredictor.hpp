#pragma once

#include <cstdlib>
#include <fstream>
#include <string>
#include "gtest/gtest.h"
#include "Tree.hpp"
#include "qstdlib.hpp"

TEST(GBM, TreePredictor) {
	Forest<unsigned> forest;
	forest.parse(fopen("model3.txt","r"));
	EXPECT_EQ(forest.size(), 200);

	std::ifstream ifs("data.sample.txt");
	std::string line;
	std::unordered_map<unsigned, float> sample;
	std::vector<double> res;
	while(getline(ifs,line)) {
		sample.clear();
		for(const char * head = qnextok(line.c_str());
				head != line.c_str()+line.size(); head = qnextok(head)) {
			const char * p = nullptr;
			unsigned col = qstrtoul(head, &p);
			float val = qstrtod(p+1, nullptr);
			sample[col] = val;
		}
		res.push_back(forest.predict(sample));
	}
	EXPECT_EQ(res.size(), 10);
	EXPECT_FLOAT_EQ(res[0], -6.789630);
	EXPECT_FLOAT_EQ(res[1], -6.845609);
	EXPECT_FLOAT_EQ(res[2], -1.922202);
	EXPECT_FLOAT_EQ(res[3], -4.395687);
	EXPECT_FLOAT_EQ(res[4], -6.011687);
	EXPECT_FLOAT_EQ(res[5], -6.390823);
	EXPECT_FLOAT_EQ(res[6], -4.528843);
	EXPECT_FLOAT_EQ(res[7], -4.739420);
	EXPECT_FLOAT_EQ(res[8], -6.320057);
	EXPECT_FLOAT_EQ(res[9], -6.969229);
}

