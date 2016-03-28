#pragma once

#include <cstdlib>
#include "gtest/gtest.h"
#include "Tree.hpp"

TEST(GBM, TreeParser) {
	Forest<size_t> forest;
	forest.parse(fopen("model3.txt","r"));
	//forest.print();
	EXPECT_EQ(forest.size(), 200);
	//forest[2].dbginfo();
	EXPECT_EQ(forest[197][4]._fea, 1595);
	EXPECT_FLOAT_EQ(forest[198][6]._cut_val, 3.700650e-02);
	EXPECT_FLOAT_EQ(forest[199][8]._pred, 0.03521960);

	for(auto&& tree: forest)
		EXPECT_EQ(tree.is_correct(), true);
}

