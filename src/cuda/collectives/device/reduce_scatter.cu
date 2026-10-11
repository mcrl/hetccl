/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "reduce_scatter.h"
#include "common.h"
#include "cuda/collectives_cuda.h"

IMPL_COLL_R(ReduceScatter);
