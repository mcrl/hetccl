#
# Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
# Modifications are Copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
#

ROCM_PATH ?= /opt/rocm
MPI_HOME ?= /usr/lib/openmpi
PREFIX ?= /usr/local
VERBOSE ?= 0
DEBUG ?= 0
NCCL_HOME ?= ""
BUILD_LOCAL_GPU_TARGET_ONLY ?= 1

HIPCC = $(ROCM_PATH)/bin/hipcc
CXX = $(HIPCC)
NVCC = $(PREFIX)/cuda/bin/nvcc

HIPCUFLAGS := -std=c++14
LDFLAGS    :=
HIPLDFLAGS :=

ifneq ($(NCCL_HOME), "")
HIPCUFLAGS += -I$(NCCL_HOME)/ -I$(NCCL_HOME)/include
HIPLDFLAGS   += -Wl,-rpath,$(NCCL_HOME) -L$(NCCL_HOME) -L$(NCCL_HOME)/lib
endif
HIPCUFLAGS += -I$(ROCM_PATH)/include
HIPCUFLAGS += -I$(ROCM_PATH)/include/hip
HIPCUFLAGS += -I$(dir $(CUSTOM_RCCL_LIB))include
HIPCUFLAGS += -I$(dir $(CUSTOM_RCCL_LIB))../src/tacc
LDFLAGS    += -L$(ROCM_PATH)/lib -lhsa-runtime64 -lrt
HIPLDFLAGS += -L$(dir $(CUSTOM_RCCL_LIB)) -L$(ROCM_PATH)/lib -lhsa-runtime64 -lrt -pthread
HIPLDFLAGS += -L/usr/local/cuda/lib64 -lcudart
ifeq ($(DEBUG), 0)
HIPCUFLAGS += -O3
else
HIPCUFLAGS += -O0 -g -ggdb3
endif

ifeq ($(VERBOSE), 0)
.SILENT:
endif

.PHONY: build clean

BUILDDIR ?= ../build

ifeq ($(MPI), 1)
HIPCUFLAGS += -DMPI_SUPPORT -I${MPI_HOME}/include -I${MPI_HOME}/include/mpi
HIPLDFLAGS += -L${MPI_HOME}/lib -lmpi
else ifeq ($(MPICH), 1)
HIPCUFLAGS += -DMPI_SUPPORT -I/usr/include/mpich -I/usr/include/x86_64-linux-gnu/mpich
HIPLDFLAGS += -L/usr/lib -lmpich
endif

# HIPCUFLAGS += --offload-arch=gfx908
#DEFAULT_GPUS := gfx803 gfx900:xnack- gfx906:xnack- gfx908:xnack- gfx90a:xnack- gfx90a:xnack+ gfx940 gfx941 gfx942 gfx1030 gfx1100 gfx1101 gfx1102
DEFAULT_GPUS := gfx908:xnack-
detect_local_gpu_arch = $(shell rocminfo | grep -oP 'gfx[0-9a-f]+' | sort | uniq)

ifeq ($(BUILD_LOCAL_GPU_TARGET_ONLY), 1)
	LOCAL_GPUS := $(call detect_local_gpu_arch)
	ifneq ($(LOCAL_GPUS),)
		DEFAULT_GPUS := $(LOCAL_GPUS)
	endif
endif

HIPCUFLAGS += $(foreach gpu,$(DEFAULT_GPUS),--offload-arch=$(gpu))

LIBRARIES += hetccl
HIPLDFLAGS   += $(LIBRARIES:%=-l%)

DST_DIR := $(BUILDDIR)
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := all_reduce sendrecv all_gather broadcast reduce_scatter reduce alltoall scatter gather alltoallv
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf)

GIT_VERSION_FILE := ${DST_DIR}/src/git_version.cpp
GIT_REV          := $(shell git log --pretty=format:'%h' -n 1)
GIT_DIFF         := $(shell git diff --quiet --exit-code || echo +)
GIT_BRANCH       := $(shell git rev-parse --abbrev-ref HEAD)

build: ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}

TEST_VERIFIABLE_SRCDIR := ../verifiable
TEST_VERIFIABLE_BUILDDIR := $(BUILDDIR)/verifiable
include ../verifiable/verifiable.mk

# Rule to create git_version.cpp
$(GIT_VERSION_FILE):
	@mkdir -p ${DST_DIR}/src
	@echo 'const char* rcclTestsGitHash = "$(GIT_BRANCH):$(GIT_REV)$(GIT_DIFF)";' > $@

${DST_DIR}/%.o: %.cpp common.h $(TEST_VERIFIABLE_HDRS) $(GIT_VERSION_FILE)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	echo "$(HIPCC) -o $@ $(HIPCUFLAGS) -I. -c $<"
	$(HIPCC) -o $@ $(HIPCUFLAGS) -I. -c $<

${DST_DIR}/timer.o: timer.cc timer.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(CXXFLAGS) -o $@ -c timer.cc

${DST_DIR}/%_perf:${DST_DIR}/%.o ${DST_DIR}/common.o ${DST_DIR}/timer.o $(TEST_VERIFIABLE_OBJS) $(DST_DIR)/src/git_version.cpp
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	echo "$(HIPCC) -o $@ $(HIPCUFLAGS) $^ ${HIPLDFLAGS}"
	$(HIPCC) -o $@ $(HIPCUFLAGS) $^ ${HIPLDFLAGS}

