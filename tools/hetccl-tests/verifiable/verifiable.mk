# Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information


# We requires both of the following paths to be set upon including this makefile
# TEST_VERIFIABLE_SRCDIR = <points to this directory>
# TEST_VERIFIABLE_BUILDDIR = <points to destination of .o file>

TEST_VERIFIABLE_HDRS = $(TEST_VERIFIABLE_SRCDIR)/verifiable.h
TEST_VERIFIABLE_OBJS = $(TEST_VERIFIABLE_BUILDDIR)/verifiable.o $(TEST_VERIFIABLE_BUILDDIR)/verifiable_hip.o $(TEST_VERIFIABLE_BUILDDIR)/verifiable_cuda.o 

$(TEST_VERIFIABLE_BUILDDIR)/verifiable_cuda.o: $(TEST_VERIFIABLE_SRCDIR)/verifiable_cuda.cu $(TEST_VERIFIABLE_SRCDIR)/verifiable_cuda.h
	@printf "Compiling %s\n" $@
	@mkdir -p $(TEST_VERIFIABLE_BUILDDIR)
	echo " $(NVCC) -o $@ -I/usr/local/cuda/include -lcudart -I$(dir $(CUSTOM_RCCL_LIB))include -I$(dir $(CUSTOM_RCCL_LIB))../src/tacc -c $<"
	$(NVCC) -o $@ -I/usr/local/cuda/include -lcudart -I$(dir $(CUSTOM_RCCL_LIB))include -I$(dir $(CUSTOM_RCCL_LIB))../src/tacc -c $<

$(TEST_VERIFIABLE_BUILDDIR)/verifiable_hip.o: $(TEST_VERIFIABLE_SRCDIR)/verifiable_hip.cpp $(TEST_VERIFIABLE_SRCDIR)/verifiable_hip.h $(TEST_VERIFIABLE_SRCDIR)/../src/rccl_float8.h
	@printf "Compiling %s\n" $@
	@mkdir -p $(TEST_VERIFIABLE_BUILDDIR)
	echo " $(HIPCC) -o $@ $(HIPCUFLAGS) -c $<"
	$(HIPCC) -o $@ $(HIPCUFLAGS) -c $<

$(TEST_VERIFIABLE_BUILDDIR)/verifiable.o: $(TEST_VERIFIABLE_SRCDIR)/verifiable.cc $(TEST_VERIFIABLE_SRCDIR)/verifiable.h $(TEST_VERIFIABLE_BUILDDIR)/verifiable_cuda.o $(TEST_VERIFIABLE_BUILDDIR)/verifiable_hip.o
	@printf "Compiling %s\n" $@
	@mkdir -p $(TEST_VERIFIABLE_BUILDDIR)
	echo " $(HIPCC) -o $@ $(HIPCUFLAGS) -c $<"
	$(HIPCC) -o $@ $(HIPCUFLAGS) -c $<
