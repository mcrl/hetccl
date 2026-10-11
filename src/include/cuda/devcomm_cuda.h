#pragma once

#include "devcomm.h"

#define CUDA_WARP_SIZE 32
#define SCCL_CUDA_MAX_NTHREADS 512 // 640 in NCCL
#define SCCL_CUDA_SIMPLE_MAX_NTHREADS 512
#define SCCL_CUDA_LL_MAX_NTHREADS 512

#define SCCL_CUDA_LL128_LINESIZE 128
#define SCCL_CUDA_LL128_LINEELEMS (SCCL_CUDA_LL128_LINESIZE/sizeof(uint64_t))
#define SCCL_CUDA_LL128_DATAELEMS (SCCL_CUDA_LL128_LINEELEMS-1)

#define SCCL_CUDA_LL128_MAX_NTHREADS 640
#define SCCL_CUDA_LL128_ELEMS_PER_THREAD 120

#define SCCL_CUDA_LL128_SHMEM_ELEMS_PER_THREAD 8
#define SCCL_CUDA_LL128_SHMEM_SIZE (SCCL_CUDA_LL128_SHMEM_ELEMS_PER_THREAD*SCCL_CUDA_LL128_MAX_NTHREADS)

struct scclCudaConnInfo {
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

struct scclCudaConnector {
  int connected;
  struct ncclProxyConnector proxyConn;
  struct ncclTransportComm* transportComm;
  void* transportResources;
  struct scclCudaConnInfo conn;
};

#define SCCL_CUDA_MAX_CONNS 2
struct scclCudaChannelPeer {
  struct scclCudaConnector send[SCCL_CUDA_MAX_CONNS];
  struct scclCudaConnector recv[SCCL_CUDA_MAX_CONNS];
  int refCount;
};

#define SCCL_CUDA_WORK_SIZE 512

typedef enum ncclWorkP2PType scclCudaWorkP2PType;
typedef struct ncclWorkHeader scclCudaWorkHeader;

struct scclCudaWorkElem {
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

#define SCCL_CUDA_MAX_WORK_ELEMENTS ((SCCL_CUDA_WORK_SIZE - alignUp(sizeof(scclCudaWorkHeader), alignof(scclCudaWorkElem)))/sizeof(scclCudaWorkElem))
static_assert(SCCL_CUDA_MAX_WORK_ELEMENTS == 9, "Sanity check: SCCL_CUDA_MAX_WORK_ELEMENTS == 9");

struct scclCudaWorkElemP2p {
  int peer : 30;
  int proto : 2;

  scclCudaWorkP2PType p2pType;
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

static_assert(((SCCL_CUDA_WORK_SIZE - alignUp(sizeof(scclCudaWorkHeader), alignof(scclCudaWorkElemP2p)))/sizeof(scclCudaWorkElemP2p)) >= 16, "Sanity check: SCCL_CUDA_MAX_WORK_ELEMENTS_P2P == 16");
#define SCCL_CUDA_MAX_WORK_ELEMENTS_P2P 16

struct scclCudaWorkElemReg {
  struct scclCudaWorkElem elem;
  void* dnInputs[NCCL_MAX_DIRECT_ARITY+1];
  void* dnOutputs[NCCL_MAX_DIRECT_ARITY+1];
  void* upOutputs[NCCL_MAX_DIRECT_ARITY+1];
};

#define SCCL_CUDA_MAX_WORK_ELEMENTS_REG ((SCCL_CUDA_WORK_SIZE - alignUp(sizeof(scclCudaWorkHeader), alignof(scclCudaWorkElemReg)))/sizeof(scclCudaWorkElemReg))
static_assert(SCCL_CUDA_MAX_WORK_ELEMENTS_REG == 2, "Sanity check: SCCL_CUDA_MAX_WORK_ELEMENTS_REG == 2");

// Number of named barriers supported by CUDA
#define SCCL_CUDA_MAX_GROUPS 16

struct scclCudaWork {
  scclCudaWorkHeader header;
  union {
    char pad[SCCL_CUDA_WORK_SIZE - sizeof(scclCudaWorkHeader)];
    struct scclCudaWorkElem elems[SCCL_CUDA_MAX_WORK_ELEMENTS];
    struct scclCudaWorkElemP2p p2pElems[SCCL_CUDA_MAX_WORK_ELEMENTS_P2P];
    struct scclCudaWorkElemReg regElems[SCCL_CUDA_MAX_WORK_ELEMENTS_REG];
  };
};

static_assert(sizeof(scclCudaWork) == SCCL_CUDA_WORK_SIZE, "Sanity check: sizeof(struct scclCudaWork) == SCCL_CUDA_WORK_SIZE");
static_assert(sizeof(scclCudaWork)%16 == 0, "Sanity check: sizeof(struct scclCudaWork)%16 == 0");

struct scclCudaDevChannelPeer {
  // Stripped version of ncclChannelPeer where we only keep the ncclConnInfo
  // instead of the full ncclConnector.
  struct scclCudaConnInfo send[SCCL_CUDA_MAX_CONNS];
  struct scclCudaConnInfo recv[SCCL_CUDA_MAX_CONNS];
};

struct alignas(16) scclCudaDevChannel {
  struct scclCudaDevChannelPeer** peers;
  struct ncclRing ring;
  struct ncclTree tree;
  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;
  struct ncclNvls nvls;
  uint32_t* workFifoDone; // Location of done counter, device writes index+1 of last work processed
};

struct scclCudaDevComm {
  int rank;
  int nRanks;
  int buffSizes[NCCL_NUM_PROTOCOLS];

  // Operation list for aggregation
  int workFifoDepth;
  struct scclCudaWork* workFifoHeap; // may be cudaHost or GDR memory

  // Flag to ask NCCL kernels to abort
  volatile uint32_t* abortFlag;

  // Channels, device side
  struct scclCudaDevChannel* channels/*[MAXCHANNELS]*/;
};

struct alignas(16) scclCudaDevCommAndChannels {
  struct scclCudaDevComm comm;
  struct scclCudaDevChannel channels[MAXCHANNELS];
};

__host__ __device__ constexpr int scclCudaCollUnroll(int cudaArch = NCCL_CUDA_ARCH) {
  // Our collective unroll should move to the same bytes&insns model as NVLS.
  return cudaArch >= 800 ? 8 : 4;
}

// The amount of dynamic shmem per warp
__host__ __device__ constexpr int scclCudaShmemScratchWarpSize(int cudaArch = NCCL_CUDA_ARCH) {
  return (max_constexpr<int>(
      /*LL    */0,
      /*LL128 */(SCCL_CUDA_LL128_SHMEM_ELEMS_PER_THREAD*CUDA_WARP_SIZE)*sizeof(uint64_t),
      /*SIMPLE*/(scclCudaCollUnroll(cudaArch)*CUDA_WARP_SIZE + 1)*16,
      // NVLS needs an extra 16B to read unaligned data.
      /*NVLS  */CUDA_WARP_SIZE*(cudaArch >= 900 ? 64 : 0) + 16
    ) + 15) & -16; // pad to 16 bytes
}

// The amount of dynamic shmem per block
__host__ __device__ constexpr int scclCudaShmemDynamicSize(int cudaArch = NCCL_CUDA_ARCH) {
  return cudaArch < 700 ? 0 : scclCudaShmemScratchWarpSize(cudaArch)*(SCCL_CUDA_MAX_NTHREADS/CUDA_WARP_SIZE);
}
