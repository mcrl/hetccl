#pragma once

#include "devcomm.h"

#define CUDA_WARP_SIZE 32
#define HETCCL_CUDA_MAX_NTHREADS 512 // 640 in NCCL
#define HETCCL_CUDA_SIMPLE_MAX_NTHREADS 512
#define HETCCL_CUDA_LL_MAX_NTHREADS 512

#define HETCCL_CUDA_LL128_LINESIZE 128
#define HETCCL_CUDA_LL128_LINEELEMS (HETCCL_CUDA_LL128_LINESIZE/sizeof(uint64_t))
#define HETCCL_CUDA_LL128_DATAELEMS (HETCCL_CUDA_LL128_LINEELEMS-1)

#define HETCCL_CUDA_LL128_MAX_NTHREADS 640
#define HETCCL_CUDA_LL128_ELEMS_PER_THREAD 120

#define HETCCL_CUDA_LL128_SHMEM_ELEMS_PER_THREAD 8
#define HETCCL_CUDA_LL128_SHMEM_SIZE (HETCCL_CUDA_LL128_SHMEM_ELEMS_PER_THREAD*HETCCL_CUDA_LL128_MAX_NTHREADS)

struct hetcclCudaConnInfo {
  // Regular comm mechanism
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int flags;          // Direct communication / other flags
  int shared;         // Buffers are shared
  void **ptrExchange; // Pointer exchange for direct communication
  uint64_t* redOpArgExchange; // PreOp scaler exchange for direct pull case

  int *sizesFifo;     // Sizes fifo from GPU to proxy
  int *offsFifo;      // Buffer fifo from proxy to GPU

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;
};

struct hetcclCudaConnector {
  int connected;
  struct ncclProxyConnector proxyConn;
  struct ncclTransportComm* transportComm;
  void* transportResources;
  struct hetcclCudaConnInfo conn;
};

#define HETCCL_CUDA_MAX_CONNS 2
struct hetcclCudaChannelPeer {
  struct hetcclCudaConnector send[HETCCL_CUDA_MAX_CONNS];
  struct hetcclCudaConnector recv[HETCCL_CUDA_MAX_CONNS];
  int refCount;
};

#define HETCCL_CUDA_WORK_SIZE 512

typedef enum ncclWorkP2PType hetcclCudaWorkP2PType;
typedef struct ncclWorkHeader hetcclCudaWorkHeader;

struct hetcclCudaWorkElem {
  union {
    uint8_t flagBits;
    struct {
      uint8_t isUsed:1, redOpArgIsPtr:1, regUsed:1;
    };
  };
  uint8_t nWarps;
  uint8_t direct;

  const void * sendbuff;
  void * recvbuff;

  size_t count;
  size_t lastChunkSize;
  uint32_t root;
  uint8_t bid;
  uint8_t nChannels;
  uint64_t redOpArg;
};

#define HETCCL_CUDA_MAX_WORK_ELEMENTS ((HETCCL_CUDA_WORK_SIZE - alignUp(sizeof(hetcclCudaWorkHeader), alignof(hetcclCudaWorkElem)))/sizeof(hetcclCudaWorkElem))
static_assert(HETCCL_CUDA_MAX_WORK_ELEMENTS == 9, "Sanity check: HETCCL_CUDA_MAX_WORK_ELEMENTS == 9");

struct hetcclCudaWorkElemP2p {
  int peer : 30;
  int proto : 2;

  hetcclCudaWorkP2PType p2pType;
  uint8_t nWarps;
  uint8_t warpStart;
  uint8_t ngroups;
  // Important not to use any fields with greater than 4-byte alignment since
  // we need sizeof(ncclWorkElemP2p)==28, but that would be padded up to 32 if
  // there were 8-byte fields.
  //void* buff;
  uint32_t buffHi32, buffLo32; // buff = buffHi32<<32 | buffLo32;
  //size_t count;
  uint32_t countHi32, countLo32; // count = countHi32<<32 | countLo32;
  int chunkSize;
};

static_assert(((HETCCL_CUDA_WORK_SIZE - alignUp(sizeof(hetcclCudaWorkHeader), alignof(hetcclCudaWorkElemP2p)))/sizeof(hetcclCudaWorkElemP2p)) >= 16, "Sanity check: HETCCL_CUDA_MAX_WORK_ELEMENTS_P2P == 16");
#define HETCCL_CUDA_MAX_WORK_ELEMENTS_P2P 16

struct hetcclCudaWorkElemReg {
  struct hetcclCudaWorkElem elem;
  void* dnInputs[NCCL_MAX_DIRECT_ARITY+1];
  void* dnOutputs[NCCL_MAX_DIRECT_ARITY+1];
  void* upOutputs[NCCL_MAX_DIRECT_ARITY+1];
};

#define HETCCL_CUDA_MAX_WORK_ELEMENTS_REG ((HETCCL_CUDA_WORK_SIZE - alignUp(sizeof(hetcclCudaWorkHeader), alignof(hetcclCudaWorkElemReg)))/sizeof(hetcclCudaWorkElemReg))
static_assert(HETCCL_CUDA_MAX_WORK_ELEMENTS_REG == 2, "Sanity check: HETCCL_CUDA_MAX_WORK_ELEMENTS_REG == 2");

// Number of named barriers supported by CUDA
#define HETCCL_CUDA_MAX_GROUPS 16

struct hetcclCudaWork {
  hetcclCudaWorkHeader header;
  union {
    char pad[HETCCL_CUDA_WORK_SIZE - sizeof(hetcclCudaWorkHeader)];
    struct hetcclCudaWorkElem elems[HETCCL_CUDA_MAX_WORK_ELEMENTS];
    struct hetcclCudaWorkElemP2p p2pElems[HETCCL_CUDA_MAX_WORK_ELEMENTS_P2P];
    struct hetcclCudaWorkElemReg regElems[HETCCL_CUDA_MAX_WORK_ELEMENTS_REG];
  };
};

static_assert(sizeof(hetcclCudaWork) == HETCCL_CUDA_WORK_SIZE, "Sanity check: sizeof(struct hetcclCudaWork) == HETCCL_CUDA_WORK_SIZE");
static_assert(sizeof(hetcclCudaWork)%16 == 0, "Sanity check: sizeof(struct hetcclCudaWork)%16 == 0");

struct hetcclCudaDevChannelPeer {
  // Stripped version of ncclChannelPeer where we only keep the ncclConnInfo
  // instead of the full ncclConnector.
  struct hetcclCudaConnInfo send[HETCCL_CUDA_MAX_CONNS];
  struct hetcclCudaConnInfo recv[HETCCL_CUDA_MAX_CONNS];
};

struct alignas(16) hetcclCudaDevChannel {
  struct hetcclCudaDevChannelPeer** peers;
  struct ncclRing ring;
  struct ncclTree tree;
  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;
  struct ncclNvls nvls;
  uint32_t* workFifoDone; // Location of done counter, device writes index+1 of last work processed
};

struct hetcclCudaDevComm {
  int rank;
  int nRanks;
  int buffSizes[NCCL_NUM_PROTOCOLS];

  // Operation list for aggregation
  int workFifoDepth;
  struct hetcclCudaWork* workFifoHeap; // may be cudaHost or GDR memory

  // Flag to ask NCCL kernels to abort
  volatile uint32_t* abortFlag;

  // Channels, device side
  struct hetcclCudaDevChannel* channels/*[MAXCHANNELS]*/;
};

struct alignas(16) hetcclCudaDevCommAndChannels {
  struct hetcclCudaDevComm comm;
  struct hetcclCudaDevChannel channels[MAXCHANNELS];
};

__host__ __device__ constexpr int hetcclCudaCollUnroll(int cudaArch = NCCL_CUDA_ARCH) {
  // Our collective unroll should move to the same bytes&insns model as NVLS.
  return cudaArch >= 800 ? 8 : 4;
}

// The amount of dynamic shmem per warp
__host__ __device__ constexpr int hetcclCudaShmemScratchWarpSize(int cudaArch = NCCL_CUDA_ARCH) {
  return (max_constexpr<int>(
      /*LL    */0,
      /*LL128 */(HETCCL_CUDA_LL128_SHMEM_ELEMS_PER_THREAD*CUDA_WARP_SIZE)*sizeof(uint64_t),
      /*SIMPLE*/(hetcclCudaCollUnroll(cudaArch)*CUDA_WARP_SIZE + 1)*16,
      // NVLS needs an extra 16B to read unaligned data.
      /*NVLS  */CUDA_WARP_SIZE*(cudaArch >= 900 ? 64 : 0) + 16
    ) + 15) & -16; // pad to 16 bytes
}

// The amount of dynamic shmem per block
__host__ __device__ constexpr int hetcclCudaShmemDynamicSize(int cudaArch = NCCL_CUDA_ARCH) {
  return cudaArch < 700 ? 0 : hetcclCudaShmemScratchWarpSize(cudaArch)*(HETCCL_CUDA_MAX_NTHREADS/CUDA_WARP_SIZE);
}
