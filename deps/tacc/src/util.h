#pragma once

#include <cstdio>

#define WARN(...) \
  do { \
    fprintf(stderr, "[%s:%s:%d] ", __FILE__, __func__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
  } while (0)
