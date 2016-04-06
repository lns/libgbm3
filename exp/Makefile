PROJ_ROOT=..

# It seems that clang-omp++ is very buggy..
#CXX=clang-omp++
CXX=g++
CXXFLAGS=-std=c++11 -O2 -Wall
INCLUDES=-I${PROJ_ROOT}/src -I${PROJ_ROOT}/include
# Linux
#LDFLAGS= -ldw
# Mac
LDFLAGS=-lpthread

TARGETS=demo

all: $(TARGETS)

demo: demo.cpp ${PROJ_ROOT}/src/*
	$(CXX) $(CXXFLAGS) -o $@ $(INCLUDES) $< $(LDFLAGS)

DATA=../data/SCS/train.svm

#demo: demo.s
#	$(CXX) $(CXXFLAGS) -o $@ $(INCLUDES) $< $(LDFLAGS)

#demo.s: demo.cpp ${PROJ_ROOT}/src/*
#	$(CXX) $(CXXFLAGS) -S -o $@ $(INCLUDES) $< $(LDFLAGS)

pred.txt: model.txt $(DATA)
	./predictor $^ > $@

auc: truth.txt pred.txt
	auc $^

clean:
	$(RM) demo demo.s

.PHONY: clean

