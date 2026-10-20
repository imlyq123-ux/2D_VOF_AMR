# Makefile for the 2D VOF/AMR two-phase flow solver.
#   make            -> bin/vof2d
#   make tests      -> bin/test_* (one per tests/test_*.cpp)
#   make check      -> build and run all tests
#   make DEBUG=1    -> -O0 -g with sanitizers
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -fopenmp
ifeq ($(DEBUG),1)
  CXXFLAGS += -O0 -g -fsanitize=address,undefined
  LDFLAGS  += -fsanitize=address,undefined
else
  CXXFLAGS += -O3 -march=native -DNDEBUG
endif
LDFLAGS += -fopenmp

LIB_SRC := $(filter-out src/main.cpp,$(wildcard src/*.cpp))
LIB_OBJ := $(LIB_SRC:src/%.cpp=build/%.o)
TESTS   := $(patsubst tests/%.cpp,bin/%,$(wildcard tests/test_*.cpp))

all: bin/vof2d

bin/vof2d: build/main.o $(LIB_OBJ) | bin
	$(CXX) -o $@ $^ $(LDFLAGS)

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

bin/test_%: tests/test_%.cpp $(LIB_OBJ) | bin
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $^ $(LDFLAGS)

tests: $(TESTS)

check: tests
	@set -e; for t in $(TESTS); do echo "== $$t"; ./$$t; done

build bin:
	mkdir -p $@

clean:
	rm -rf build bin

.PHONY: all tests check clean
-include $(LIB_OBJ:.o=.d) build/main.d
