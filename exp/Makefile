PROJ_ROOT=..

CXX=clang++
CXXFLAGS=-std=c++11 -O2 -Wall
INCLUDES=-I${PROJ_ROOT}/src
# Linux
#LDFLAGS= -ldw
# Mac
LDFLAGS=

TARGETS=demo

all: $(TARGETS)

demo: demo.cpp ${PROJ_ROOT}/src/*
	$(CXX) $(CXXFLAGS) -o $@ $(INCLUDES) $< $(LDFLAGS)

