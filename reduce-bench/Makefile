include make.defs.local
SHELL=/bin/bash

# Depending on the platform it may be also mpicxx --> specify as an environment
# parameter
MPICXX?=mpic++

DASHROOT?=$(HOME)/opt/dash-0.4.0
DASHCXX=$(DASHROOT)/bin/dash-mpic++
DASHNVCC=$(DASHROOT)/bin/dash-nvcc
MEPHISTOROOT=$(HOME)/src/mephisto/mephisto
ALPAKAROOT=$(HOME)/src/mephisto/alpaka
BOOSTROOT=$(BOOST_ROOT)
# BOOST_ROOT?=$(BOOST_ROOT)

OMPFLAG?=-fopenmp

CXXFLAGS+= -O3 -Iinclude -Iexternal -DNDEBUG

DASHCXXFLAGS = $(CXXFLAGS)
NVCC_DIAG_SUPPRESS="-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -Xcudafe --diag_suppress=code_is_unreachable -Xcudafe --diag_suppress=set_but_not_used -Xcudafe --diag_suppress=initialization_not_reachable -Xcudafe --diag_suppress=expr_has_no_effect"
MEPHISTOCXXFLAGS = $(DASHCXXFLAGS) -I$(DASHROOT)/include -I$(BOOSTROOT)/include -I$(ALPAKAROOT)/include -DUSE_MEPHISTO

SIZE ?= $$((500 * (2**20)))

OS := $(shell uname)

ifeq ($(NPROCS),)
ifeq ($(OS),Linux)
	NPROCS:=$(shell grep -c ^processor /proc/cpuinfo)
endif
endif

NPROCS?=4

ENABLE_TRACE?=0

.PHONY: clean


all: build/dash.x build/mpi.x mephisto

run: all
	DASH_ENABLE_TRACE=$(ENABLE_TRACE)	mpirun -n $(NPROCS) ./build/dash.x $$(($(SIZE) / $(NPROCS)))
						mpirun -n $(NPROCS) ./build/mpi.x $$(($(SIZE) / $(NPROCS)))
						mpirun -n $(NPROCS) ./build/mephisto.x $$(($(SIZE) / $(NPROCS)))

run-mephisto: build/mephisto.x
						mpirun -n $(NPROCS) ./build/mephisto.x $$(($(SIZE) / $(NPROCS)))

mephisto: build/mephisto.serial.x build/mephisto.threads.x build/mephisto.omp.x build/mephisto.nv

build/dash.x: include/dash/reducebench.h reducebench.cc
	@mkdir -p build
	$(DASHCXX) $(DASHCXXFLAGS) -o $@ -DUSE_DASH reducebench.cc

build/mephisto.serial.x: include/mephisto/reducebench.h reducebench.cc
	@mkdir -p build
	$(DASHCXX) $(MEPHISTOCXXFLAGS) -o $@ -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED  reducebench.cc

build/mephisto.threads.x: include/mephisto/reducebench.h reducebench.cc
	@mkdir -p build
	$(DASHCXX) $(MEPHISTOCXXFLAGS) -o $@ -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED  reducebench.cc

build/mephisto.omp.x: include/mephisto/reducebench.h reducebench.cc
	@mkdir -p build
	$(DASHCXX) $(MEPHISTOCXXFLAGS) $(OMPFLAG) -o $@ -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED reducebench.cc

build/mephisto.nv: include/mephisto/reducebench.h reducebench.cu
	@mkdir -p build
	$(DASHNVCC) $(MEPHISTOCXXFLAGS) $(NVCC_DIAG_SUPPRESS) --expt-extended-lambda --expt-relaxed-constexpr -o $@ -DALPAKA_ACC_GPU_CUDA_ENABLED reducebench.cu -G

build/mpi.x: include/mpi/reducebench.h reducebench.cc
	@mkdir -p build
	$(MPICXX) $(CXXFLAGS) -Iexternal -o $@ -DUSE_MPI reducebench.cc

clean:
	rm -f build/*.x
	rm -f build/*.nv
