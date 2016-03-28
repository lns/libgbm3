PROJ_ROOT=..

CXX=clang-omp++
CXXFLAGS=-std=c++11 -O2 -Wall
INCLUDES=-I${PROJ_ROOT}/src
# Linux
#LDFLAGS= -ldw -lgtest
# Mac
LDFLAGS= -lgtest

TARGETS=main-test

all: $(TARGETS)

test: $(TARGETS)
	./main-test

main-test: main-test.cpp *.hpp ${PROJ_ROOT}/src/*
	$(CXX) $(CXXFLAGS) -o $@ $(INCLUDES) $< $(LDFLAGS)

