#include <cstdio>
#include "tacc.h"

#define CHECK_TACC(call) \
  do { \
    taccError_t err = call; \
    if (err != taccSuccess) { \
      fprintf(stderr, "[%s:%s:%d] taccError %d (%s)\n", __FILE__, __func__, __LINE__, err, taccGetErrorString(err)); \
      return err; \
    } \
  } while (0)

int main() {
  CHECK_TACC(taccSetPlatformAuto());
  taccPlatform platform;
  CHECK_TACC(taccGetPlatform(&platform));
  printf("Platform: %d\n", platform);
  return 0;
}