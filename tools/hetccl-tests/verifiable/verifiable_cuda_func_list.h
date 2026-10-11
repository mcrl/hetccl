CMD(cudaVerifiablePrepareInput, void,
  void *elts, intptr_t elt_n, int elt_ty, int red_op, int rank_n, int rank_me,
  uint64_t seed, intptr_t elt_ix0, taccStream_t stream);
CMD(cudaVerifiablePrepareExpected, void,
  void *elts, intptr_t elt_n, int elt_ty, int red_op, int rank_n,
  uint64_t seed, intptr_t elt_ix0, taccStream_t stream);
CMD(cudaVerifiableVerify, void,
  void const *results, void const *expected, intptr_t elt_n, int elt_ty,
  int red_op, int rank_n, uint64_t seed, intptr_t elt_ix0,
  int64_t *bad_elt_n, taccStream_t stream);