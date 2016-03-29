PROJ_ROOT=..

CXX=clang-omp++
CXXFLAGS=-std=c++11 -O2 -Wall -fopenmp
INCLUDES=-I${PROJ_ROOT}/src
# Linux
#LDFLAGS= -ldw
# Mac
LDFLAGS=

TARGETS=demo

all: $(TARGETS)

demo: demo.s
	$(CXX) $(CXXFLAGS) -o $@ $(INCLUDES) $< $(LDFLAGS)

demo.s: demo.cpp ${PROJ_ROOT}/src/*
	$(CXX) $(CXXFLAGS) -S -o $@ $(INCLUDES) $< $(LDFLAGS)

