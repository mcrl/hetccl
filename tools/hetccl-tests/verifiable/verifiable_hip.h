#include <tacc.h>
#include "verifiable.h"

// Enqueue kernel to generate data which is to be reduced.
void hipVerifiablePrepareInput(
  void *elts, intptr_t elt_n, int elt_ty, int red_op, int rank_n, int rank_me,
  uint64_t seed, intptr_t elt_ix0, taccStream_t stream_
);

// Enqueue kernel to generate expected results of reduction.
void hipVerifiablePrepareExpected(
  void *elts, intptr_t elt_n, int elt_ty, int red_op, int rank_n,
  uint64_t seed, intptr_t elt_ix0, taccStream_t stream_
);

// Enqueue kernel to verify reduced data matches expectation. The number of
// failed elements is written to bad_elt_n which must be in cudaHost memory.
// If `expected == nullptr` then the expected results are generated on-the-fly
// which can be costly. Thus if you plan to run the same reduction multiple
// times it is advantageous to precompute the expected values with
// ncclVerifiablePrepareExpected and pass them as `expected` here.
void hipVerifiableVerify(
  void const *results, void const *expected, intptr_t elt_n, int elt_ty,
  int red_op, int rank_n, uint64_t seed, intptr_t elt_ix0,
  int64_t *bad_elt_n, taccStream_t stream_
);