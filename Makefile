# Makefile for the 2D VOF/AMR two-phase flow solver.
#   make            -> bin/vof2d
#   make tests      -> bin/test_* (one per tests/test_*.cpp)
#   make check      -> build and run all tests
#   make DEBUG=1    -> -O0 -g with sanitizers
#   make CUDA=1     -> also build the CUDA linear solver (needs nvcc; runs on the GPU
#                      unless gpu=false). CUDA_HOME (default: from nvcc in PATH, else
#                      /usr/local/cuda) and CUDA_ARCH (default: native = the GPU of this
#                      machine, e.g. sm_86, or "all") can be overridden.
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -fopenmp
ifeq ($(DEBUG),1)
  CXXFLAGS += -O0 -g -fsanitize=address,undefined
  LDFLAGS  += -fsanitize=address,undefined
else
  CXXFLAGS += -O3 -march=native -DNDEBUG
endif
LDFLAGS += -fopenmp

ifeq ($(CUDA),1)
  NVCC_PATH := $(shell command -v nvcc 2>/dev/null)
  CUDA_HOME ?= $(if $(NVCC_PATH),$(patsubst %/bin/nvcc,%,$(NVCC_PATH)),/usr/local/cuda)
  NVCC      ?= $(CUDA_HOME)/bin/nvcc
  CUDA_ARCH ?= native
  NVCCFLAGS ?= -std=c++17 -O3 -arch=$(CUDA_ARCH) -Xcompiler -Wall,-Wextra,-fopenmp
  CXXFLAGS  += -DVOF2D_CUDA
  LDFLAGS   += -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib -lcudart_static -ldl -lrt -lpthread
  VARIANT   := cuda
  OBJDIR    := build/cuda
else
  VARIANT   := cpu
  OBJDIR    := build
endif

LIB_SRC := $(filter-out src/main.cpp,$(wildcard src/*.cpp))
LIB_OBJ := $(LIB_SRC:src/%.cpp=$(OBJDIR)/%.o)
ifeq ($(CUDA),1)
  LIB_OBJ += $(OBJDIR)/poisson_gpu.o
endif
TESTS   := $(patsubst tests/%.cpp,bin/%,$(wildcard tests/test_*.cpp))

all: bin/vof2d

# bin/.variant changes only when switching between CPU and CUDA builds, which forces
# the binaries to be relinked from the other object directory.
bin/.variant: FORCE | bin
	@echo '$(VARIANT)' | cmp -s - $@ || echo '$(VARIANT)' > $@

bin/vof2d: $(OBJDIR)/main.o $(LIB_OBJ) bin/.variant | bin
	$(CXX) -o $@ $(filter %.o,$^) $(LDFLAGS)

$(OBJDIR)/%.o: src/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(OBJDIR)/%.o: src/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -c $< -o $@

bin/test_%: tests/test_%.cpp $(LIB_OBJ) bin/.variant | bin
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $(filter %.cpp %.o,$^) $(LDFLAGS)

tests: $(TESTS)

check: tests
	@set -e; for t in $(TESTS); do echo "== $$t"; ./$$t; done

build build/cuda bin:
	mkdir -p $@

clean:
	rm -rf build bin

.PHONY: all tests check clean FORCE
-include $(LIB_OBJ:.o=.d) $(OBJDIR)/main.d
