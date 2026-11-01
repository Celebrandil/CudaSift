#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>
#include "cudaSift.h"
#include "cudautils.h"
#include "cudaSiftH.h"
#include "cudaSiftBatch.h"
#include "cudaSiftRandom.h"

// Device memory kept between the calls of MatchSiftData, FindHomographyGPU and
// ImproveHomographyGPU, per device, and grown when needed. The calls hold the lock of
// their device (ScratchLock) until they have completed, so that they can be made from
// several host threads.
//
// In a batch (cudaSiftBatch.h), the point counts are on the GPU, not on the host: the calls
// size their grids and scratch memory for the capacities (SiftData::maxPts), and their
// kernels take pointers to the counts on the GPU (d_numPts...), which they read instead of
// numPts when they are set. Outside batches, these pointers are NULL.
enum { SCRATCH_COORD, SCRATCH_VALID, SCRATCH_RANDPTS, SCRATCH_HOMO, SCRATCH_RESULT,   // FindHomographyGPU
       SCRATCH_IMPROVE_RESULT, SCRATCH_IMPROVE_STATE,                               // ImproveHomographyGPU
       SCRATCH_BEST,                                                                // MatchSiftData
       SCRATCH_BUFFERS };
#define MAX_DEVICES 64
struct DeviceScratch {
  std::mutex lock;
  void *ptr[SCRATCH_BUFFERS];
  size_t size[SCRATCH_BUFFERS];
  int maxBlocks;   // resident FindMaxCorr11 blocks, 0 until known
  // Batches: each FindHomographyGPU and ImproveHomographyGPU call of a batch writes its
  // result (a HomographyResult) to a slot of its own, read when the batch completes.
  // homographies maps each homography array to the slot of the last call of the batch that
  // computed it, so that a later ImproveHomographyGPU call on it starts from that result.
  std::vector<void *> results;
  size_t usedResults;
  unsigned int batchId;
  std::map<const float *, void *> homographies;
};
static DeviceScratch scratch[MAX_DEVICES];

static DeviceScratch &DeviceScratchOf(int device)
{
  if (device>=MAX_DEVICES) {
    fprintf(stderr, "CudaSift: device %d not supported (at most %d devices)\n", device, MAX_DEVICES);
    exit(EXIT_FAILURE);
  }
  return scratch[device];
}

static std::unique_lock<std::mutex> ScratchLock()
{
  int device;
  safeCall(cudaGetDevice(&device));
  return std::unique_lock<std::mutex>(DeviceScratchOf(device).lock);
}

// Frees the scratch memory at exit (registered by the first allocation, so it runs before
// the CUDA runtime's own exit handler). Errors are ignored, e.g. after cudaDeviceReset().
static void FreeScratch()
{
  int device;
  if (cudaGetDevice(&device)!=cudaSuccess)
    return;
  for (int d=0;d<MAX_DEVICES;d++) {
    for (int i=0;i<SCRATCH_BUFFERS;i++)
      if (scratch[d].ptr[i]!=NULL && cudaSetDevice(d)==cudaSuccess)
        cudaFree(scratch[d].ptr[i]);
    for (void *r : scratch[d].results)
      if (cudaSetDevice(d)==cudaSuccess)
        cudaFree(r);
  }
  cudaSetDevice(device);
  cudaGetLastError();
}

// Registers FreeScratch, at the first allocation
static void FreeScratchAtExit()
{
  static std::once_flag atExit;
  std::call_once(atExit, [] { std::atexit(FreeScratch); });
}

static void *Scratch(int index, size_t size)   // with the ScratchLock of the device held
{
  FreeScratchAtExit();
  int device;
  safeCall(cudaGetDevice(&device));
  DeviceScratch &s = DeviceScratchOf(device);
  if (size>s.size[index]) {
    if (s.ptr[index]!=NULL)
      safeCall(cudaFree(s.ptr[index]));   // waits for the device, also for the work of a batch
    safeCall(cudaMalloc(&s.ptr[index], size));
    s.size[index] = size;
  }
  return s.ptr[index];
}

//================= Device matching functions =====================//

__global__ void MatchSiftPoints(SiftPoint *sift1, SiftPoint *sift2, float *corrData, int numPts1, int numPts2)
{
  __shared__ float siftPoint[128];
  __shared__ float sums[16*16];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1 = blockIdx.x;
  const int p2 = blockIdx.y*16 + ty;
  const float *ptr1 = sift1[p1].data;
  const float *ptr2 = sift2[p2].data;
  const int i = 16*ty + tx;
  if (ty<8)
    siftPoint[i] = ptr1[i];
  __syncthreads();
  float sum = 0.0f;
  if (p2<numPts2)
    for (int j=0;j<8;j++)
      sum += siftPoint[16*j+tx] * ptr2[16*j+tx];
  sums[i] = sum;
  __syncthreads();
  if (tx<8)
    sums[i] += sums[i+8];
  __syncthreads();
  if (tx<4)
    sums[i] += sums[i+4];
  __syncthreads();
  if (ty==0) {
    sum = sums[16*tx+0] + sums[16*tx+1] + sums[16*tx+2] + sums[16*tx+3];
    corrData[p1*gridDim.y*16 + blockIdx.y*16 + tx] = sum;
  }
  __syncthreads();
}

__global__ void MatchSiftPoints2(SiftPoint *sift1, SiftPoint *sift2, float *corrData, int numPts1, int numPts2)
{
  __shared__ float siftPoints1[16*128];
  __shared__ float siftPoints2[16*128];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const float *ptr1 = sift1[min(numPts1-1,blockIdx.x*16 + ty)].data;
  const float *ptr2 = sift2[min(numPts2-1,blockIdx.y*16 + ty)].data;
  for (int i=0;i<8;i++) {
    siftPoints1[128*ty+16*i+tx] = ptr1[16*i+tx];
    siftPoints2[128*ty+16*i+tx] = ptr2[16*i+tx];
  }
  __syncthreads();
  const int p1 = blockIdx.x*16 + ty;
  const int p2 = blockIdx.y*16 + tx;
  const float *pt1 = &siftPoints1[ty*128];
  const float *pt2 = &siftPoints2[tx*128];
  float sum = 0.0f;
  for (int i=0;i<128;i++) {
    int itx = (i + tx)&127; // avoid bank conflicts
    sum += pt1[itx]*pt2[itx];
  }
  if (p1<numPts1)
    corrData[p1*gridDim.y*16 + p2] = (p2<numPts2 ? sum : -1.0f);
}

__global__ void FindMaxCorr(float *corrData, SiftPoint *sift1, SiftPoint *sift2, int numPts1, int corrWidth, int siftSize)
{
  __shared__ float maxScore[16*16];
  __shared__ float maxScor2[16*16];
  __shared__ int maxIndex[16*16];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = ty*16 + tx;
  int p1 = blockIdx.x*16 + threadIdx.y;
  p1 = (p1>=numPts1 ? numPts1-1 : p1);
  maxScore[idx] = -1.0f;
  maxScor2[idx] = -1.0f;
  maxIndex[idx] = -1;
  __syncthreads();
  float *corrs = &corrData[p1*corrWidth];
  for (int i=tx;i<corrWidth;i+=16) {
    float val = corrs[i];
    if (val>maxScore[idx]) {
      maxScor2[idx] = maxScore[idx];
      maxScore[idx] = val;
      maxIndex[idx] = i;
    } else if (val>maxScor2[idx])
      maxScor2[idx] = val;
  }
  __syncthreads();
  for (int len=8;len>0;len/=2) {
    if (tx<8) {
      float val = maxScore[idx+len];
      int i = maxIndex[idx+len];
      if (val>maxScore[idx]) {
	maxScor2[idx] = maxScore[idx];
	maxScore[idx] = val;
	maxIndex[idx] = i;
      } else if (val>maxScor2[idx])
	maxScor2[idx] = val;
      float va2 = maxScor2[idx+len];
      if (va2>maxScor2[idx])
	maxScor2[idx] = va2;
    }
    __syncthreads();
  }
  if (tx==0) {
    sift1[p1].score = maxScore[ty*16];
    sift1[p1].ambiguity = maxScor2[ty*16] / (maxScore[ty*16] + 1e-6);
    sift1[p1].match = maxIndex[ty*16];
    sift1[p1].match_xpos = sift2[maxIndex[ty*16]].xpos;
    sift1[p1].match_ypos = sift2[maxIndex[ty*16]].ypos;
  }
}

// Version based on suggestion by Nicholas Lin
__global__ void FindMaxCorr3(float *corrData, SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  int block_dim = blockDim.x; // blockDim.x == 16
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1 = blockIdx.x * block_dim + ty;
  const int idx = ty * 16 + tx;
  
  __shared__ int maxIndex[16 * 16];
  maxIndex[idx] = 0;
  __syncthreads();
  
  float *corrs = NULL;
  if (p1 < numPts1) {
    corrs = &corrData[p1 * block_dim * 2];
    corrs[tx] = 0.0f;
    corrs[tx + 16] = 0.0f;
    const float *pt1 = sift1[p1].data;
    for (int p2 = tx; p2 < numPts2; p2 += 16) {
      float *pt2 = sift2[p2].data;
      float sum = 0.0f;
      for (int i = 0; i < 128; i++) 
	sum += pt1[i] * pt2[i];
      if (sum > corrs[tx]) {
	corrs[tx + 16] = corrs[tx];
	corrs[tx] = sum;
	maxIndex[idx] = p2;
      }
      else if (sum > corrs[tx + 16])
	corrs[tx + 16] = sum;
    }
  }
  __syncthreads();
  if (p1 < numPts1) {
    for (int len = 8; len > 0; len /= 2) {
      if (tx < len) {
	float val = corrs[tx + len];
	int i = maxIndex[idx + len];
	if (val > corrs[tx]) {
	  corrs[tx + 16] = corrs[tx];
	  corrs[tx] = val;
	  maxIndex[idx] = i;
	}
	else if (val > corrs[tx + 16])
	  corrs[tx + 16] = val;
	float va2 = corrs[tx + 16 + len];
	if (va2 > corrs[tx + 16])
	  corrs[tx + 16] = va2;
      }
      __syncthreads();
    }
    if (tx==0) {
      sift1[p1].score = corrs[0];
      sift1[p1].ambiguity = corrs[16] / (corrs[0] + 1e-6);
      sift1[p1].match = maxIndex[ty << 4];
      sift1[p1].match_xpos = sift2[maxIndex[ty << 4]].xpos;
      sift1[p1].match_ypos = sift2[maxIndex[ty << 4]].ypos;
    }
  }
}

#define FMC2W 16
#define FMC2H 4

__global__ void FindMaxCorr2(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float siftPoint[128];
  __shared__ float maxScore[FMC2H]; 
  __shared__ float maxScor2[FMC2H]; 
  __shared__ int maxIndex[FMC2H]; 
  const int p1 = blockIdx.x;
  if (p1>=numPts1)
    return;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = ty*FMC2W + tx;
  if (idx<FMC2H) {
    maxScore[idx] = -1.0f;
    maxScor2[idx] = -1.0f;
    maxIndex[idx] = 0;
  }
  __syncthreads();
  const float *pt1 = sift1[p1].data;
  for (int i=idx;i<128;i+=FMC2W*FMC2H)
    siftPoint[i] = pt1[i];
  __syncthreads();
  for (int p2=ty;p2<numPts2;p2+=FMC2H) {
    const float *pt2 = sift2[p2].data;
    float sum = 0.0f;
    for (int j=tx;j<128;j+=FMC2W)
      sum += siftPoint[j] * pt2[j];
    for (int j=FMC2W/2;j>0;j/=2)
      sum += ShiftDown(sum, j);
    if (tx==0) {
      if (sum>maxScore[ty]) {
	maxScor2[ty] = maxScore[ty];
	maxScore[ty] = sum;
	maxIndex[ty] = p2;
      } else if (sum>maxScor2[ty])
	maxScor2[ty] = sum;
    }
  }
  __syncthreads();
  for (int len=FMC2H/2;len>0;len/=2) {
    if (ty==0 && tx<len) {
      float val = maxScore[tx+len];
      int p2 = maxIndex[tx+len];
      if (val>maxScore[tx]) {
	maxScor2[tx] = maxScore[tx];
	maxScore[tx] = val;
	maxIndex[tx] = p2;
      } else if (val>maxScor2[tx])
	maxScor2[tx] = val;
      float va2 = maxScor2[tx+len];
      if (va2>maxScor2[tx])
	maxScor2[tx] = va2;
    }
    __syncthreads();
  }
  if (ty==0 && tx==0) {
    sift1[p1].score = maxScore[0];
    sift1[p1].ambiguity = maxScor2[0] / (maxScore[0] + 1e-6);
    sift1[p1].match = maxIndex[0];
    sift1[p1].match_xpos = sift2[maxIndex[0]].xpos;
    sift1[p1].match_ypos = sift2[maxIndex[0]].ypos;
  }
}

__global__ void FindMaxCorr4(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float siftPoint[128*FMC2H];
  __shared__ float maxScore[FMC2H]; 
  __shared__ float maxScor2[FMC2H]; 
  __shared__ int maxIndex[FMC2H]; 
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  if (tx==0) {
    maxScore[ty] = -1.0f;
    maxScor2[ty] = -1.0f;
    maxIndex[ty] = 0;
  }
  const int p1 = blockIdx.x*FMC2H + ty;
  const float *pt1 = sift1[p1].data;
  for (int j=tx;j<128;j+=FMC2W)
    siftPoint[128*ty + j] = pt1[j];
  __syncthreads();
  for (int p2=0;p2<numPts2;p2++) {
    const float *pt2 = sift2[p2].data;
    float sum = 0.0f;
    for (int j=tx;j<128;j+=FMC2W)
      sum += siftPoint[128*ty + j] * pt2[j]; 
    for (int j=FMC2W/2;j>0;j/=2)
      sum += ShiftDown(sum, j);
    if (tx==0) {
      if (sum>maxScore[ty]) {
	maxScor2[ty] = maxScore[ty];
	maxScore[ty] = sum;
	maxIndex[ty] = p2;
      } else if (sum>maxScor2[ty])
	maxScor2[ty] = sum;
    }
  }
  __syncthreads();
  if (tx==0) {
    sift1[p1].score = maxScore[ty];
    sift1[p1].ambiguity = maxScor2[ty] / (maxScore[ty] + 1e-6);
    sift1[p1].match = maxIndex[ty];
    sift1[p1].match_xpos = sift2[maxIndex[ty]].xpos;
    sift1[p1].match_ypos = sift2[maxIndex[ty]].ypos;
  }
}


// d_numPts1, d_numPts2: the point counts on the GPU in a batch, else NULL (see the top of the file)
__global__ void CleanMatches(SiftPoint *sift1, int numPts1, int numPts2, const int *d_numPts1, const int *d_numPts2)
{
  if (d_numPts1) {
    numPts1 = *d_numPts1;
    numPts2 = *d_numPts2;
  }
  // Nothing to match: outside batches, MatchSiftData returns before launching
  if (!numPts1 || !numPts2)
    return;
  const int p1 = min(blockIdx.x*64 + threadIdx.x, numPts1-1);
  sift1[p1].score = 0.0f;
}

#define M7W   32
#define M7H   32
#define M7R    4
#define NRX    2
#define NDIM 128

__global__ void FindMaxCorr10(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float4 buffer1[M7W*NDIM/4]; 
  __shared__ float4 buffer2[M7H*NDIM/4];       
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bp1 = M7W*blockIdx.x;
  for (int j=ty;j<M7W;j+=M7H/M7R) {    
    int p1 = min(bp1 + j, numPts1 - 1);
    for (int d=tx;d<NDIM/4;d+=M7W)
      buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)&sift1[p1].data)[d];
  }
      
  float max_score[NRX];
  float sec_score[NRX];
  int index[NRX];
  for (int i=0;i<NRX;i++) {
    max_score[i] = 0.0f;
    sec_score[i] = 0.0f;
    index[i] = -1;
  }
  int idx = ty*M7W + tx;
  int ix = idx%(M7W/NRX);
  int iy = idx/(M7W/NRX);
  for (int bp2=0;bp2<numPts2 - M7H + 1;bp2+=M7H) {
    for (int j=ty;j<M7H;j+=M7H/M7R) {      
      int p2 = min(bp2 + j, numPts2 - 1);
      for (int d=tx;d<NDIM/4;d+=M7W)
	buffer2[j*NDIM/4 + d] = ((float4*)&sift2[p2].data)[d];
    }
    __syncthreads();

    if (idx<M7W*M7H/M7R/NRX) {
      float score[M7R][NRX];                                    
      for (int dy=0;dy<M7R;dy++)
	for (int i=0;i<NRX;i++)
	  score[dy][i] = 0.0f;
      for (int d=0;d<NDIM/4;d++) {
	float4 v1[NRX];
	for (int i=0;i<NRX;i++) 
	  v1[i] = buffer1[((M7W/NRX)*i + ix)*NDIM/4 + (d + (M7W/NRX)*i + ix)%(NDIM/4)];
	for (int dy=0;dy<M7R;dy++) {
	  float4 v2 = buffer2[(M7R*iy + dy)*(NDIM/4) + d];    
	  for (int i=0;i<NRX;i++) {
	    score[dy][i] += v1[i].x*v2.x;
	    score[dy][i] += v1[i].y*v2.y;
	    score[dy][i] += v1[i].z*v2.z;
	    score[dy][i] += v1[i].w*v2.w;
	  }
	}
      }
      for (int dy=0;dy<M7R;dy++) {
	for (int i=0;i<NRX;i++) {
	  if (score[dy][i]>max_score[i]) {
	    sec_score[i] = max_score[i];
	    max_score[i] = score[dy][i];     
	    index[i] = min(bp2 + M7R*iy + dy, numPts2-1);
	  } else if (score[dy][i]>sec_score[i])
	    sec_score[i] = score[dy][i]; 
	}
      }
    }
    __syncthreads();
  }

  float *scores1 = (float*)buffer1;
  float *scores2 = &scores1[M7W*M7H/M7R];
  int *indices = (int*)&scores2[M7W*M7H/M7R];
  if (idx<M7W*M7H/M7R/NRX) {
    for (int i=0;i<NRX;i++) {
      scores1[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];  
      scores2[iy*M7W + (M7W/NRX)*i + ix] = sec_score[i];  
      indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
    }
  }
  __syncthreads();
  
  if (ty==0) {
    float max_score = scores1[tx];
    float sec_score = scores2[tx];
    int index = indices[tx];
    for (int y=0;y<M7H/M7R;y++)
      if (index != indices[y*M7W + tx]) {
	if (scores1[y*M7W + tx]>max_score) {
	  sec_score = max(max_score, sec_score);
	  max_score = scores1[y*M7W + tx]; 
	  index = indices[y*M7W + tx];
	} else if (scores1[y*M7W + tx]>sec_score)
	  sec_score = scores1[y*M7W + tx];

        if (scores2[y*M7W + tx]>sec_score)
          sec_score = scores2[y*M7W + tx];
      }
    sift1[bp1 + tx].score = max_score;
    sift1[bp1 + tx].match = index;
    sift1[bp1 + tx].match_xpos = sift2[index].xpos;
    sift1[bp1 + tx].match_ypos = sift2[index].ypos;
    sift1[bp1 + tx].ambiguity = sec_score / (max_score + 1e-6f);
  }
}

// Register-blocked version: the (sift1 tile, sift2 tile) pairs are distributed over
// a grid of persistent blocks, so that a few thousand points still fill the GPU
// evenly. Each of the 64 threads computes 8x8 correlations per tile pair from
// k-major shared memory. Per-tile best/second-best scores go to d_best and are
// combined by MergeMaxCorr11. Every sift2 index belongs to exactly one tile, so
// the merged result is the exact global best and second-best.
#define M11_T   64            // points per tile (sift1 and sift2)
#define M11_K   16            // descriptor dimensions per shared memory stage
#define M11_P   (M11_T + 4)   // padded stage row length, keeps float4 alignment
#define M11_NT  64            // threads per block
#define M11_CHUNKS 64         // sift2 chunks in a batch (d_best: M11_CHUNKS*maxPts1)
static_assert(M11_T==64 && M11_K==16 && M11_NT==64, "FindMaxCorr11 indexing assumes 64 points per tile, 16 dimensions per stage and 64 threads");

// The sift2 tiles are split in numChunks chunks of consecutive tiles, and d_best[c*stride + p1]
// is the best and second best match of sift1 point p1 in chunk c. Outside batches, numChunks
// is the number of sift2 tiles (one tile per chunk) and stride numPts1. In a batch, the point
// counts are on the GPU (d_numPts1, d_numPts2), numChunks is M11_CHUNKS and stride maxPts1,
// so that d_best has a size known on the host. The results of the tiles of a chunk are
// combined as MergeMaxCorr11 combines those of the chunks, in the same order, so the matches
// don't depend on the chunks.
__global__ void __launch_bounds__(M11_NT) FindMaxCorr11(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2, float4 *d_best, int numChunks, int stride, const int *d_numPts1, const int *d_numPts2)
{
  if (d_numPts1) {
    numPts1 = *d_numPts1;
    numPts2 = *d_numPts2;
  }
  __shared__ __align__(16) float smem[2*M11_K*M11_P];
  float *As = smem;
  float *Bs = smem + M11_K*M11_P;
  const int tid = threadIdx.x;
  const int tx = tid%8;         // sift1 rows 4*tx+{0..3} and 32+4*tx+{0..3}
  const int ty = tid/8;         // sift2 cols 4*ty+{0..3} and 32+4*ty+{0..3}
  const int lp = tid/4;         // loader: point within tile (four passes of 16)
  const int lq = tid%4;         // loader: float4 within stage
  const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int tiles1 = (numPts1 + M11_T - 1)/M11_T;
  const int tiles2 = (numPts2 + M11_T - 1)/M11_T;
  if (!tiles1 || !tiles2)
    return;
  const int perChunk = (tiles2 + numChunks - 1)/numChunks;   // tiles per chunk
  const int chunks = (tiles2 + perChunk - 1)/perChunk;

  for (int item=blockIdx.x;item<tiles1*chunks;item+=gridDim.x) {
    const int bp1 = (item/chunks)*M11_T;
    const int c = item%chunks;
    float chunk_max = 0.0f, chunk_sec = 0.0f;   // best and second best in the chunk, row tid
    int chunk_index = -1;
    for (int t2=c*perChunk;t2<min((c + 1)*perChunk, tiles2);t2++) {
    const int bp2 = t2*M11_T;
    float acc[8][8];
#pragma unroll
    for (int i=0;i<8;i++)
#pragma unroll
      for (int j=0;j<8;j++)
	acc[i][j] = 0.0f;
    for (int k0=0;k0<NDIM;k0+=M11_K) {
#pragma unroll
      for (int r=0;r<M11_T;r+=16) {
	int p = lp + r;
	// Zero padding: zero descriptors score 0 and can never become best or second best
	float4 a = (bp1 + p < numPts1 ? ((float4*)sift1[bp1 + p].data)[k0/4 + lq] : zero);
	float4 b = (bp2 + p < numPts2 ? ((float4*)sift2[bp2 + p].data)[k0/4 + lq] : zero);
	As[(4*lq + 0)*M11_P + p] = a.x;
	As[(4*lq + 1)*M11_P + p] = a.y;
	As[(4*lq + 2)*M11_P + p] = a.z;
	As[(4*lq + 3)*M11_P + p] = a.w;
	Bs[(4*lq + 0)*M11_P + p] = b.x;
	Bs[(4*lq + 1)*M11_P + p] = b.y;
	Bs[(4*lq + 2)*M11_P + p] = b.z;
	Bs[(4*lq + 3)*M11_P + p] = b.w;
      }
      __syncthreads();
#pragma unroll
      for (int k=0;k<M11_K;k++) {
	float4 a0 = *(float4*)&As[k*M11_P + 4*tx];
	float4 a1 = *(float4*)&As[k*M11_P + 32 + 4*tx];
	float4 b0 = *(float4*)&Bs[k*M11_P + 4*ty];
	float4 b1 = *(float4*)&Bs[k*M11_P + 32 + 4*ty];
	float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
	float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
	for (int i=0;i<8;i++)
#pragma unroll
	  for (int j=0;j<8;j++)
	    acc[i][j] += av[i]*bv[j];
      }
      __syncthreads();
    }

    // Best and second best of this thread's 8 columns, then over the 8 threads sharing a row
    float *red = smem;  // M11_T*8*3 floats, fits in smem
#pragma unroll
    for (int i=0;i<8;i++) {
      float max_score = 0.0f, sec_score = 0.0f;
      int index = -1;
#pragma unroll
      for (int j=0;j<8;j++) {
	if (acc[i][j]>max_score) {
	  sec_score = max_score;
	  max_score = acc[i][j];
	  index = bp2 + (j<4 ? 4*ty + j : 32 + 4*ty + j - 4);
	} else if (acc[i][j]>sec_score)
	  sec_score = acc[i][j];
      }
      int r = (i<4 ? 4*tx + i : 32 + 4*tx + i - 4);
      red[(r*8 + ty)*3 + 0] = max_score;
      red[(r*8 + ty)*3 + 1] = sec_score;
      red[(r*8 + ty)*3 + 2] = __int_as_float(index);
    }
    __syncthreads();
    if (bp1 + tid<numPts1) {
      float max_score = 0.0f, sec_score = 0.0f;
      int index = -1;
      for (int t=0;t<8;t++) {
	float m = red[(tid*8 + t)*3 + 0];
	float s = red[(tid*8 + t)*3 + 1];
	if (m>max_score) {
	  sec_score = fmaxf(max_score, s);
	  max_score = m;
	  index = __float_as_int(red[(tid*8 + t)*3 + 2]);
	} else
	  sec_score = fmaxf(sec_score, m);
      }
      if (max_score>chunk_max) {
	chunk_sec = fmaxf(chunk_max, sec_score);
	chunk_max = max_score;
	chunk_index = index;
      } else
	chunk_sec = fmaxf(chunk_sec, max_score);
    }
    __syncthreads();
    }
    if (bp1 + tid<numPts1)
      d_best[c*stride + bp1 + tid] = make_float4(chunk_max, chunk_sec, __int_as_float(chunk_index), 0.0f);
  }
}

__global__ void MergeMaxCorr11(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2, float4 *d_best, int numChunks, int stride, const int *d_numPts1, const int *d_numPts2)
{
  if (d_numPts1) {
    numPts1 = *d_numPts1;
    numPts2 = *d_numPts2;
  }
  const int p1 = blockIdx.x*blockDim.x + threadIdx.x;
  if (p1>=numPts1 || !numPts2)
    return;
  const int tiles2 = (numPts2 + M11_T - 1)/M11_T;
  const int perChunk = (tiles2 + numChunks - 1)/numChunks;
  const int chunks = (tiles2 + perChunk - 1)/perChunk;
  float max_score = 0.0f, sec_score = 0.0f;
  int index = -1;
  for (int t=0;t<chunks;t++) {
    float4 b = d_best[t*stride + p1];
    if (b.x>max_score) {
      sec_score = fmaxf(max_score, b.y);
      max_score = b.x;
      index = __float_as_int(b.z);
    } else
      sec_score = fmaxf(sec_score, b.x);
  }
  sift1[p1].score = max_score;
  sift1[p1].ambiguity = sec_score / (max_score + 1e-6f);
  if (index>=0 && index<numPts2) {
    sift1[p1].match = index;
    sift1[p1].match_xpos = sift2[index].xpos;
    sift1[p1].match_ypos = sift2[index].ypos;
  } else {
    sift1[p1].match = -1;
    sift1[p1].match_xpos = 0.0f;
    sift1[p1].match_ypos = 0.0f;
  }
}

#define FMC_GH  512
#define FMC_BW   32
#define FMC_BH   32
#define FMC_BD   16
#define FMC_TW    1
#define FMC_TH    4
#define FMC_NW   (FMC_BW/FMC_TW)   //  32
#define FMC_NH   (FMC_BH/FMC_TH)   //   8
#define FMC_NT   (FMC_NW*FMC_NH)   // 256 = 8 warps

__device__ volatile int lock = 0;

__global__ void FindMaxCorr9(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float4 siftParts1[FMC_BW*FMC_BD]; // 4*32*8 = 1024
  __shared__ float4 siftParts2[FMC_BH*FMC_BD]; // 4*32*8 = 1024
  //__shared__ float blksums[FMC_BW*FMC_BH];     // 32*32  = 1024
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = ty*FMC_NW + tx;
  float4 *pts1 = 0, *pts2 = 0;
  if (idx<FMC_BW) {
    const int p1l = min(blockIdx.x*FMC_BW + idx, numPts1-1);
    pts1 = (float4*)sift1[p1l].data;
  }
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k=0;k<min(FMC_GH, numPts2 - FMC_BH + 1);k+=FMC_BH) {
    if (idx<FMC_BH) {
      const int p2l = min(blockIdx.y*FMC_GH + k + idx, numPts2-1);
      pts2 = (float4*)sift2[p2l].data;
    }
    float sums[FMC_TW*FMC_TH];
    for (int i=0;i<FMC_TW*FMC_TH;i++) 
      sums[i] = 0.0f;

    if (idx<FMC_BW)
      for (int i=0;i<FMC_BD/2;i++) 
	siftParts1[(i + 0)*FMC_BW + idx] = pts1[0 + i];
    if (idx<FMC_BH)
      for (int i=0;i<FMC_BD/2;i++) 
	siftParts2[(i + 0)*FMC_BH + idx] = pts2[0 + i];
    __syncthreads();
    
    int b = FMC_BD/2;
    for (int d=FMC_BD/2;d<32;d+=FMC_BD/2) {
      if (idx<FMC_BW)
	for (int i=0;i<FMC_BD/2;i++) 
	  siftParts1[(i + b)*FMC_BW + idx] = pts1[d + i];
      if (idx<FMC_BH)
	for (int i=0;i<FMC_BD/2;i++) 
	  siftParts2[(i + b)*FMC_BH + idx] = pts2[d + i];

      b ^= FMC_BD/2;
      for (int i=0;i<FMC_BD/2;i++) {
	float4 v1[FMC_TW];
	for (int ix=0;ix<FMC_TW;ix++)
	  v1[ix] = siftParts1[(i + b)*FMC_BW + (tx*FMC_TW + ix)];
	for (int iy=0;iy<FMC_TH;iy++) {
	  float4 v2 = siftParts2[(i + b)*FMC_BH + (ty*FMC_TH + iy)];
	  for (int ix=0;ix<FMC_TW;ix++) {
	    sums[iy*FMC_TW + ix] += v1[ix].x * v2.x;
	    sums[iy*FMC_TW + ix] += v1[ix].y * v2.y;
	    sums[iy*FMC_TW + ix] += v1[ix].z * v2.z;
	    sums[iy*FMC_TW + ix] += v1[ix].w * v2.w;
	  }
	}
      }
      __syncthreads();
    }
    
    b ^= FMC_BD/2;
    for (int i=0;i<FMC_BD/2;i++) {
      float4 v1[FMC_TW];
      for (int ix=0;ix<FMC_TW;ix++)
	v1[ix] = siftParts1[(i + b)*FMC_BW + (tx*FMC_TW + ix)];
      for (int iy=0;iy<FMC_TH;iy++) {
	float4 v2 = siftParts2[(i + b)*FMC_BH + (ty*FMC_TH + iy)];
	for (int ix=0;ix<FMC_TW;ix++) {
	  sums[iy*FMC_TW + ix] += v1[ix].x * v2.x;
	  sums[iy*FMC_TW + ix] += v1[ix].y * v2.y;
	  sums[iy*FMC_TW + ix] += v1[ix].z * v2.z;
	  sums[iy*FMC_TW + ix] += v1[ix].w * v2.w;
	}
      }
    }
    __syncthreads();
    
    float *blksums = (float*)siftParts1;
    for (int iy=0;iy<FMC_TH;iy++) 
      for (int ix=0;ix<FMC_TW;ix++) 
	blksums[(ty*FMC_TH + iy)*FMC_BW + (tx*FMC_TW + ix)] = sums[iy*FMC_TW + ix];
    __syncthreads();
    if (idx<FMC_BW) { 
      for (int j=0;j<FMC_BH;j++) {
	float sum = blksums[j*FMC_BW + idx];
	if (sum>maxScore) { 
	  maxScor2 = maxScore;
	  maxScore = sum;
	  maxIndex = min(blockIdx.y*FMC_GH + k + j, numPts2-1);
	} else if (sum>maxScor2)
	  maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  const int p1 = min(blockIdx.x*FMC_BW + idx, numPts1-1);
  if (idx==0)
    while (atomicCAS((int *)&lock, 0, 1) != 0);
  __syncthreads();
  if (idx<FMC_BW) {
    float maxScor2Old = sift1[p1].ambiguity*(sift1[p1].score + 1e-6f);
    if (maxScore>sift1[p1].score) {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    } else if (maxScore>maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (idx==0)
    atomicExch((int* )&lock, 0);
}

__global__ void FindMaxCorr8(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float4 siftParts1[FMC_BW*FMC_BD]; // 4*32*8 = 1024
  __shared__ float4 siftParts2[FMC_BH*FMC_BD]; // 4*32*8 = 1024
  __shared__ float blksums[FMC_BW*FMC_BH];     // 32*32  = 1024
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = ty*FMC_NW + tx;
  float4 *pts1 = 0, *pts2 = 0;
  if (idx<FMC_BW) {
    const int p1l = min(blockIdx.x*FMC_BW + idx, numPts1-1);
    pts1 = (float4*)sift1[p1l].data;
  }
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k=0;k<min(FMC_GH, numPts2 - FMC_BH + 1);k+=FMC_BH) {
    if (idx<FMC_BH) {
      const int p2l = min(blockIdx.y*FMC_GH + k + idx, numPts2-1);
      pts2 = (float4*)sift2[p2l].data;
    }
    float sums[FMC_TW*FMC_TH];
    for (int i=0;i<FMC_TW*FMC_TH;i++) 
      sums[i] = 0.0f;
    for (int d=0;d<32;d+=FMC_BD) {
      if (idx<FMC_BW)
	for (int i=0;i<FMC_BD;i++) 
	  siftParts1[i*FMC_BW + idx] = pts1[d + i];
      if (idx<FMC_BH)
	for (int i=0;i<FMC_BD;i++) 
	  siftParts2[i*FMC_BH + idx] = pts2[d + i];
      __syncthreads();
      
      for (int i=0;i<FMC_BD;i++) {
	float4 v1[FMC_TW];
	for (int ix=0;ix<FMC_TW;ix++)
	  v1[ix] = siftParts1[i*FMC_BW + (tx*FMC_TW + ix)];
	for (int iy=0;iy<FMC_TH;iy++) {
	  float4 v2 = siftParts2[i*FMC_BH + (ty*FMC_TH + iy)];
	  for (int ix=0;ix<FMC_TW;ix++) {
	    sums[iy*FMC_TW + ix] += v1[ix].x * v2.x;
	    sums[iy*FMC_TW + ix] += v1[ix].y * v2.y;
	    sums[iy*FMC_TW + ix] += v1[ix].z * v2.z;
	    sums[iy*FMC_TW + ix] += v1[ix].w * v2.w;
	  }
	}
      }
      __syncthreads();
    }
    //float *blksums = (float*)siftParts1;
    for (int iy=0;iy<FMC_TH;iy++) 
      for (int ix=0;ix<FMC_TW;ix++) 
	blksums[(ty*FMC_TH + iy)*FMC_BW + (tx*FMC_TW + ix)] = sums[iy*FMC_TW + ix];
    __syncthreads();
    if (idx<FMC_BW) { 
      for (int j=0;j<FMC_BH;j++) {
	float sum = blksums[j*FMC_BW + idx];
	if (sum>maxScore) { 
	  maxScor2 = maxScore;
	  maxScore = sum;
	  maxIndex = min(blockIdx.y*FMC_GH + k + j, numPts2-1);
	} else if (sum>maxScor2)
	  maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  const int p1 = min(blockIdx.x*FMC_BW + idx, numPts1-1);
  if (idx==0)
    while (atomicCAS((int *)&lock, 0, 1) != 0);
  __syncthreads();
  if (idx<FMC_BW) {
    float maxScor2Old = sift1[p1].ambiguity*(sift1[p1].score + 1e-6f);
    if (maxScore>sift1[p1].score) {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    } else if (maxScore>maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (idx==0)
    atomicExch((int* )&lock, 0);
}

__global__ void FindMaxCorr7(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float siftParts1[17*64]; // features in columns
  __shared__ float siftParts2[16*64]; // one extra to avoid shared conflicts
  float4 *pts1 = (float4*)siftParts1;
  float4 *pts2 = (float4*)siftParts2;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1l = min(blockIdx.x*16 + ty, numPts1-1);
  const float4 *p1l4 = (float4*)sift1[p1l].data;
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k=0;k<512/16;k++) {
    const int p2l = min(blockIdx.y*512 + k*16 + ty, numPts2-1);
    const float4 *p2l4 = (float4*)sift2[p2l].data;
#define NUM 4
    float sum[NUM];
    if (ty<(16/NUM))
      for (int l=0;l<NUM;l++)
	sum[l] = 0.0f;
    __syncthreads();
    for (int i=0;i<2;i++) {
      pts1[17*tx + ty] = p1l4[i*16 + tx];
      pts2[16*ty + tx] = p2l4[i*16 + tx];
      __syncthreads(); 
      if (ty<(16/NUM)) {
#pragma unroll
	for (int j=0;j<16;j++) {
	  float4 p1v = pts1[17* j + tx];
#pragma unroll
	  for (int l=0;l<NUM;l++) {
	    float4 p2v = pts2[16*(ty + l*(16/NUM)) +  j];
	    sum[l] += p1v.x * p2v.x;
	    sum[l] += p1v.y * p2v.y;
	    sum[l] += p1v.z * p2v.z;
	    sum[l] += p1v.w * p2v.w;
	  }
	}
      }
      __syncthreads();
    }
    float *sums = siftParts1;
    if (ty<(16/NUM))
      for (int l=0;l<NUM;l++) 
	sums[16*(ty + l*(16/NUM)) + tx] = sum[l];
    __syncthreads();
    if (ty==0) { 
      for (int j=0;j<16;j++) {
	float sum = sums[16*j + tx];
	if (sum>maxScore) { 
	  maxScor2 = maxScore;
	  maxScore = sum;
	  maxIndex = min(blockIdx.y*512 +  k*16 + j, numPts2-1);
	} else if (sum>maxScor2)
	  maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  const int p1 = min(blockIdx.x*16 + tx, numPts1-1);
  if (tx==0 && ty==0)
    while (atomicCAS((int *)&lock, 0, 1) != 0);
  __syncthreads();
  if (ty==0) {
    float maxScor2Old = sift1[p1].ambiguity*(sift1[p1].score + 1e-6f);
    if (maxScore>sift1[p1].score) {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    } else if (maxScore>maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (tx==0 && ty==0)
    atomicExch((int* )&lock, 0);
}

__global__ void FindMaxCorr6(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  //__shared__ float siftParts1[128*16]; // features in columns
  __shared__ float siftParts2[128*16]; // one extra to avoid shared conflicts
  __shared__ float sums[16*16];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1l = min(blockIdx.x*16 + ty, numPts1-1);
  float *pt1l = sift1[p1l].data;
  float4 part1 = reinterpret_cast<float4*>(pt1l)[tx];
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k=0;k<512;k+=16) {
    const int p2l = min(blockIdx.y*512 + k + ty, numPts2-1);
    float *pt2l = sift2[p2l].data;
    reinterpret_cast<float4*>(siftParts2)[32*ty + tx] = reinterpret_cast<float4*>(pt2l)[tx];
    __syncthreads();
    for (int i=0;i<16;i++) {
      float4 part2 = reinterpret_cast<float4*>(siftParts2)[32*i  + tx];
      float sum = part1.x*part2.x + part1.y*part2.y + part1.z*part2.z + part1.w*part2.w;
      sum += ShiftDown(sum, 16);
      sum += ShiftDown(sum, 8);
      sum += ShiftDown(sum, 4);
      sum += ShiftDown(sum, 2);
      sum += ShiftDown(sum, 1);
      if (tx==0)
	sums[16*i + ty] = sum;
    }
    __syncthreads();
    if (ty==0 && tx<16) { 
      for (int j=0;j<16;j++) {
	float sum = sums[16*j + tx];
	if (sum>maxScore) { 
	  maxScor2 = maxScore;
	  maxScore = sum;
	  maxIndex = min(blockIdx.y*512 +  k + j, numPts2-1);
	} else if (sum>maxScor2)
	  maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  if (tx==0 && ty==0)
    while (atomicCAS((int *)&lock, 0, 1) != 0);
  __syncthreads();
  if (ty==0 && tx<16) {
    const int p1 = min(blockIdx.x*16 + tx, numPts1-1);
    float maxScor2Old = sift1[p1].ambiguity*(sift1[p1].score + 1e-6f);
    if (maxScore>sift1[p1].score) {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    } else if (maxScore>maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (tx==0 && ty==0)
    atomicExch((int* )&lock, 0);
}
 
__global__ void FindMaxCorr5(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float siftParts1[17*16]; // features in columns
  __shared__ float siftParts2[17*16]; // one extra to avoid shared conflicts
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1l = min(blockIdx.x*16 + ty, numPts1-1);
  const float *pt1l = sift1[p1l].data;
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k=0;k<512/16;k++) {
    const int p2l = min(blockIdx.y*512 + k*16 + ty, numPts2-1);
    const float *pt2l = sift2[p2l].data;
    float sum = 0.0f;
    for (int i=0;i<8;i++) {
      siftParts1[17*tx + ty] = pt1l[i*16 + tx]; // load and transpose
      siftParts2[17*tx + ty] = pt2l[i*16 + tx];
      __syncthreads();
      for (int j=0;j<16;j++)
	sum += siftParts1[17*j + tx] * siftParts2[17*j + ty];
      __syncthreads();
    }
    float *sums = siftParts1;
    sums[16*ty + tx] = sum;
    __syncthreads();
    if (ty==0) { 
      for (int j=0;j<16;j++) {
	float sum = sums[16*j + tx];
	if (sum>maxScore) { 
	  maxScor2 = maxScore;
	  maxScore = sum;
	  maxIndex = min(blockIdx.y*512 +  k*16 + j, numPts2-1);
	} else if (sum>maxScor2)
	  maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  const int p1 = min(blockIdx.x*16 + tx, numPts1-1);
  if (tx==0 && ty==0)
    while (atomicCAS((int *)&lock, 0, 1) != 0);
  __syncthreads();
  if (ty==0) {
    float maxScor2Old = sift1[p1].ambiguity*(sift1[p1].score + 1e-6f);
    if (maxScore>sift1[p1].score) {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    } else if (maxScore>maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (tx==0 && ty==0)
    atomicExch((int* )&lock, 0);
}
 

template <int size>
__device__ void InvertMatrix(float elem[size][size], float res[size][size]) 
{  
  int indx[size];
  float b[size];
  float vv[size];
  for (int i=0;i<size;i++)
    indx[i] = 0;
  int imax = 0;
  float d = 1.0;
  for (int i=0;i<size;i++) { // find biggest element for each row
    float big = 0.0;
    for (int j=0;j<size;j++) {
      float temp = fabs(elem[i][j]); 
      if (temp>big) 
	big = temp;
    }
    if (big>0.0)
      vv[i] = 1.0/big;
    else
      vv[i] = 1e16;
  }
  for (int j=0;j<size;j++) { 
    for (int i=0;i<j;i++) { // i<j
      float sum = elem[i][j]; // i<j (lower left)
      for (int k=0;k<i;k++) // k<i<j
	sum -= elem[i][k]*elem[k][j]; // i>k (upper right), k<j (lower left)
      elem[i][j] = sum; // i<j (lower left)
    }
    float big = 0.0;
    for (int i=j;i<size;i++) { // i>=j
      float sum = elem[i][j]; // i>=j (upper right)
      for (int k=0;k<j;k++) // k<j<=i
	sum -= elem[i][k]*elem[k][j]; // i>k (upper right), k<j (lower left)
      elem[i][j] = sum; // i>=j (upper right)
      float dum = vv[i]*fabs(sum);
      if (dum>=big) {
	big = dum;
	imax = i;  
      }
    }
    if (j!=imax) { // imax>j
      for (int k=0;k<size;k++) {
	float dum = elem[imax][k]; // upper right and lower left
	elem[imax][k] = elem[j][k];
	elem[j][k] = dum;
      }
      d = -d;
      vv[imax] = vv[j];
    }
    indx[j] = imax;
    if (elem[j][j]==0.0)  // j==j (upper right)
      elem[j][j] = 1e-16;
    if (j!=(size-1)) {
      float dum = 1.0/elem[j][j];
      for (int i=j+1;i<size;i++) // i>j
	elem[i][j] *= dum; // i>j (upper right)
    }
  }
  for (int j=0;j<size;j++) {
    for (int k=0;k<size;k++) 
      b[k] = 0.0;  
    b[j] = 1.0;
    int ii = -1;
    for (int i=0;i<size;i++) {
      int ip = indx[i];
      float sum = b[ip];
      b[ip] = b[i];
      if (ii!=-1)
	for (int j=ii;j<i;j++) 
	  sum -= elem[i][j]*b[j]; // i>j (upper right)
      else if (sum!=0.0)
        ii = i;
      b[i] = sum;
    }
    for (int i=size-1;i>=0;i--) {
      float sum = b[i];
      for (int j=i+1;j<size;j++) 
	sum -= elem[i][j]*b[j]; // i<j (lower left)
      b[i] = sum/elem[i][i]; // i==i (upper right)
    }
    for (int i=0;i<size;i++)
      res[i][j] = b[i];
  }
}

__global__ void ComputeHomographies(float *coord, int *randPts, float *homo, 
  int numPts) 
{
  float a[8][8], ia[8][8];
  float b[8]; 
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int idx = blockDim.x*bx + tx;
  const int numLoops = blockDim.x*gridDim.x;
  for (int i=0;i<4;i++) {
    int pt = randPts[i*numLoops+idx];
    float x1 = coord[pt+0*numPts];
    float y1 = coord[pt+1*numPts];
    float x2 = coord[pt+2*numPts];
    float y2 = coord[pt+3*numPts];
    float *row1 = a[2*i+0];
    row1[0] = x1;
    row1[1] = y1;
    row1[2] = 1.0;
    row1[3] = row1[4] = row1[5] = 0.0;
    row1[6] = -x2*x1;
    row1[7] = -x2*y1;
    float *row2 = a[2*i+1];
    row2[0] = row2[1] = row2[2] = 0.0;
    row2[3] = x1;
    row2[4] = y1;
    row2[5] = 1.0;
    row2[6] = -y2*x1;
    row2[7] = -y2*y1;
    b[2*i+0] = x2;
    b[2*i+1] = y2;
  }
  InvertMatrix<8>(a, ia);
  __syncthreads();
  for (int j=0;j<8;j++) {
    float sum = 0.0f;
    for (int i=0;i<8;i++) 
      sum += ia[j][i]*b[i];
    homo[j*numLoops+idx] = sum;
  }
  __syncthreads();
}

#define TESTHOMO_TESTS 16 // number of tests per block,  alt. 32, 32
#define TESTHOMO_LOOPS 16 // number of loops per block,  alt.  8, 16 

__global__ void TestHomographies(float *d_coord, float *d_homo, 
  int *d_counts, int numPts, int numPtsUp, float thresh2)
{
  __shared__ float homo[8*TESTHOMO_LOOPS];
  __shared__ int cnts[TESTHOMO_TESTS*TESTHOMO_LOOPS];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = blockIdx.y*blockDim.y + tx;
  const int numLoops = blockDim.y*gridDim.y;
  if (ty<8 && tx<TESTHOMO_LOOPS)
    homo[tx*8+ty] = d_homo[idx+ty*numLoops];
  __syncthreads();
  float a[8];
  for (int i=0;i<8;i++) 
    a[i] = homo[ty*8+i];
  int cnt = 0;
  for (int i=tx;i<numPts;i+=TESTHOMO_TESTS) {
    float x1 = d_coord[i+0*numPtsUp];
    float y1 = d_coord[i+1*numPtsUp];
    float x2 = d_coord[i+2*numPtsUp];
    float y2 = d_coord[i+3*numPtsUp];
    float nomx = __fmul_rz(a[0],x1) + __fmul_rz(a[1],y1) + a[2];
    float nomy = __fmul_rz(a[3],x1) + __fmul_rz(a[4],y1) + a[5];
    float deno = __fmul_rz(a[6],x1) + __fmul_rz(a[7],y1) + 1.0f;
    float errx = __fmul_rz(x2,deno) - nomx;
    float erry = __fmul_rz(y2,deno) - nomy;
    float err2 = __fmul_rz(errx,errx) + __fmul_rz(erry,erry);
    if (err2<__fmul_rz(thresh2,__fmul_rz(deno,deno)))
      cnt ++;
  }
  int kty = TESTHOMO_TESTS*ty;
  cnts[kty + tx] = cnt;
  __syncthreads();
  int len = TESTHOMO_TESTS/2;
  while (len>0) {
    if (tx<len)
      cnts[kty + tx] += cnts[kty + tx + len];
    len /= 2;
    __syncthreads();
  }
  if (tx<TESTHOMO_LOOPS && ty==0)
    d_counts[idx] = cnts[TESTHOMO_TESTS*tx];
  __syncthreads();
}

// SiftSameOrientation (cudaSiftRandom.h) for the points sample[0..3] of coord, which holds
// x1, y1, x2, y2 with the given stride
__host__ __device__ static bool SameOrientation(const float *coord, int stride, const int *sample)
{
  float x1[4], y1[4], x2[4], y2[4];
  for (int k=0;k<4;k++) {
    x1[k] = coord[sample[k]];
    y1[k] = coord[sample[k] + stride];
    x2[k] = coord[sample[k] + 2*stride];
    y2[k] = coord[sample[k] + 3*stride];
  }
  return SiftSameOrientation(x1, y1, x2, y2);
}

//================= Homography estimation on the GPU =====================
// FindHomographyGPU and ImproveHomographyGPU, see the host functions below

struct HomographyResult {   // homography and number of matches / inliers
  float h[9];
  int count;
};

// Returns a result slot for a call of the open batch. Called with the ScratchLock held.
static HomographyResult *BatchResult(DeviceScratch &s)
{
  FreeScratchAtExit();
  // First call of a new batch: the slots of the previous batch were read when it completed
  if (s.batchId!=BatchId()) {
    s.batchId = BatchId();
    s.usedResults = 0;
    s.homographies.clear();
  }
  if (s.usedResults==s.results.size()) {
    void *r = NULL;
    safeCall(cudaMalloc(&r, sizeof(HomographyResult)));
    s.results.push_back(r);
  }
  return (HomographyResult *)s.results[s.usedResults++];
}

__global__ void SetResult(HomographyResult *res, HomographyResult value)
{
  *res = value;
}

// Coordinates of all points, and indices of the valid points (score > minScore and
// ambiguity < maxAmbiguity) in increasing order, as in FindHomography. One block of 1024.
__global__ void HomographyPrepare(SiftPoint *sift, int numPts, float *coord, int *valid, int *numValid, int stride, float minScore, float maxAmbiguity, const int *d_numPts)
{
  if (d_numPts)   // the point count on the GPU, in a batch
    numPts = *d_numPts;
  __shared__ int warpCount[32];
  __shared__ int warpFirst[32];
  __shared__ int total;
  const int tid = threadIdx.x;
  const int lane = tid%32;
  const int warp = tid/32;
  int base = 0;
  for (int c0=0;c0<numPts;c0+=blockDim.x) {
    const int i = c0 + tid;
    bool ok = false;
    if (i<numPts) {
      coord[i + 0*stride] = sift[i].xpos;
      coord[i + 1*stride] = sift[i].ypos;
      coord[i + 2*stride] = sift[i].match_xpos;
      coord[i + 3*stride] = sift[i].match_ypos;
      ok = (sift[i].score>minScore && sift[i].ambiguity<maxAmbiguity);
    }
    unsigned int mask = __ballot_sync(0xffffffff, ok);
    const int pre = __popc(mask & ((1u << lane) - 1));
    if (lane==0)
      warpCount[warp] = __popc(mask);
    __syncthreads();
    if (warp==0) {
      int c = (lane<blockDim.x/32 ? warpCount[lane] : 0);
      int inc = c;
      for (int d=1;d<32;d<<=1) {
        int n = __shfl_up_sync(0xffffffff, inc, d);
        if (lane>=d)
          inc += n;
      }
      warpFirst[lane] = inc - c;
      if (lane==31)
        total = inc;
    }
    __syncthreads();
    if (ok)
      valid[base + warpFirst[warp] + pre] = i;
    base += total;
    __syncthreads();
  }
  if (tid==0)
    *numValid = base;
}

// The coordinates of the valid points, in the order of the points (valid holds their
// indices in increasing order): RANSAC samples and counts the inliers among these only. The
// samples depend on the order of the points, as those of FindHomographySeeded; SortSiftData
// makes it reproducible.
__global__ void HomographyGather(float *coord, int *valid, int *numValid, float *validCoord, int stride)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=*numValid)
    return;
  const int a = valid[i];
  validCoord[i] = coord[a];
  validCoord[i + stride] = coord[a + stride];
  validCoord[i + 2*stride] = coord[a + 2*stride];
  validCoord[i + 3*stride] = coord[a + 3*stride];
}

__global__ void HomographySample(float *validCoord, int *numValid, int *randPts, int *sameOrientation, int numLoops, int stride, unsigned int seed)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=numLoops)
    return;
  const int nv = *numValid;
  int q[4] = {0, 0, 0, 0};
  if (nv>=8)
    SiftRandomSample4(seed, i, nv, q);
  for (int k=0;k<4;k++)
    randPts[i + k*numLoops] = q[k];
  sameOrientation[i] = (nv>=8 && SameOrientation(validCoord, stride, q));
}

// As TestHomographies, over the *numPts valid points, and with 4 hypotheses per thread,
// so that each point is loaded once for 4 hypotheses. 16 x 16 threads: 64 hypotheses per
// block, the 16 threads of a row take every 16th point.
#define TESTHOMO_PER 4
__global__ void TestHomographiesGPU(float *d_coord, float *d_homo, int *d_counts, const int *d_numPts, int stride, int numLoops, float thresh2)
{
  const int numPts = *d_numPts;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int h0 = (blockIdx.y*16 + ty)*TESTHOMO_PER;   // first hypothesis of this thread
  float a[TESTHOMO_PER][8];
  for (int h=0;h<TESTHOMO_PER;h++)
    for (int i=0;i<8;i++)
      a[h][i] = (h0 + h<numLoops ? d_homo[h0 + h + i*numLoops] : 0.0f);
  int cnt[TESTHOMO_PER] = {0};
  for (int i=tx;i<numPts;i+=16) {
    float x1 = d_coord[i+0*stride];
    float y1 = d_coord[i+1*stride];
    float x2 = d_coord[i+2*stride];
    float y2 = d_coord[i+3*stride];
#pragma unroll
    for (int h=0;h<TESTHOMO_PER;h++) {
      float nomx = __fmul_rz(a[h][0],x1) + __fmul_rz(a[h][1],y1) + a[h][2];
      float nomy = __fmul_rz(a[h][3],x1) + __fmul_rz(a[h][4],y1) + a[h][5];
      float deno = __fmul_rz(a[h][6],x1) + __fmul_rz(a[h][7],y1) + 1.0f;
      float errx = __fmul_rz(x2,deno) - nomx;
      float erry = __fmul_rz(y2,deno) - nomy;
      float err2 = __fmul_rz(errx,errx) + __fmul_rz(erry,erry);
      if (err2<__fmul_rz(thresh2,__fmul_rz(deno,deno)))
        cnt[h] ++;
    }
  }
  // Sum over the 16 threads of the row (half a warp)
#pragma unroll
  for (int h=0;h<TESTHOMO_PER;h++)
    for (int d=1;d<16;d<<=1)
      cnt[h] += __shfl_xor_sync(0xffffffff, cnt[h], d);
  if (tx<TESTHOMO_PER && h0 + tx<numLoops)
    d_counts[h0 + tx] = cnt[tx];
}

// Hypothesis with the most inliers among those from samples that pass SameOrientation
// (the first one in case of ties, as FindHomography). Identity and 0 matches if there are
// fewer than 8 valid points or no sample passes. One block of 1024.
__global__ void HomographyBest(int *d_counts, const int *sameOrientation, float *d_homo, int *numValid, HomographyResult *res, int numLoops)
{
  __shared__ int warpMax[32];
  __shared__ int warpIdx[32];
  const int tid = threadIdx.x;
  const int lane = tid%32;
  int maxCount = -1, maxIndex = -1;
  for (int i=tid;i<numLoops;i+=blockDim.x)
    if (sameOrientation[i] && d_counts[i]>maxCount) {
      maxCount = d_counts[i];
      maxIndex = i;
    }
  for (int d=16;d>0;d/=2) {
    int c = __shfl_down_sync(0xffffffff, maxCount, d), j = __shfl_down_sync(0xffffffff, maxIndex, d);
    if (lane+d<32 && (c>maxCount || (c==maxCount && j>=0 && (maxIndex<0 || j<maxIndex)))) {
      maxCount = c;
      maxIndex = j;
    }
  }
  if (lane==0) {
    warpMax[tid/32] = maxCount;
    warpIdx[tid/32] = maxIndex;
  }
  __syncthreads();
  if (tid==0) {
    for (int w=1;w<blockDim.x/32;w++)
      if (warpMax[w]>maxCount || (warpMax[w]==maxCount && warpIdx[w]>=0 && (maxIndex<0 || warpIdx[w]<maxIndex))) {
        maxCount = warpMax[w];
        maxIndex = warpIdx[w];
      }
    if (*numValid<8 || maxIndex<0) {
      for (int j=0;j<9;j++)
        res->h[j] = (j%4==0 ? 1.0f : 0.0f);
      res->count = 0;
    } else {
      for (int j=0;j<8;j++)
        res->h[j] = d_homo[maxIndex + j*numLoops];
      res->h[8] = 1.0f;
      res->count = maxCount;
    }
  }
}

// ImproveHomography (geomFuncs.cpp, in double precision on the CPU) on the GPU, in double
// precision: numLoops iterations of least squares over the inliers, then the inlier count
// and match_error of all points. Each iteration: ImproveAccumulate sums the normal
// equations over the points in IMPROVE_GROUPS blocks, then ImproveSolve adds the partial
// sums (in a fixed order) and solves them (Cholesky). The residuals are rounded to float
// as in the CPU code; the products -x*mx, -y*mx, -x*my, -y*my are not (the CPU code
// computes them in float). Only the non-zero terms of the equations are summed: a point's
// two rows have non-zero coefficients 0,1,2,6,7 and 3,4,5,6,7.
#define IMPROVE_GROUPS  16
#define IMPROVE_NT      256
#define IMPROVE_NV      35   // 27 non-zero entries of the symmetric 8x8 matrix + 8 of the vector

struct ImproveState {       // device state between the kernels of ImproveHomographyGPU
  double A[8];              // current homography
  double partial[IMPROVE_GROUPS][IMPROVE_NV];
  int counts[IMPROVE_GROUPS];
};

__device__ __inline__ float ImproveError(const double *a, float x, float y, float mx, float my)
{
  float den = a[6]*x + a[7]*y + 1.0f;
  float dx = (a[0]*x + a[1]*y + a[2]) / den - mx;
  float dy = (a[3]*x + a[4]*y + a[5]) / den - my;
  return dx*dx + dy*dy;
}

__global__ void ImproveInit(HomographyResult *in, ImproveState *st)
{
  const int i = threadIdx.x;
  if (i<8)
    st->A[i] = in->h[i] / in->h[8];
}

__global__ void ImproveAccumulate(SiftPoint *sift, int numPts, ImproveState *st, float minScore, float maxAmbiguity, float limit, const int *d_numPts)
{
  if (d_numPts)   // the point count on the GPU, in a batch
    numPts = *d_numPts;
  __shared__ double red[IMPROVE_NT/32][IMPROVE_NV];
  __shared__ int counts[IMPROVE_NT/32];
  const int tid = threadIdx.x;
  const int lane = tid%32;
  double a[8];
  for (int k=0;k<8;k++)
    a[k] = st->A[k];
  double acc[IMPROVE_NV];
  for (int k=0;k<IMPROVE_NV;k++)
    acc[k] = 0.0;
  int numfit = 0;
  for (int i=blockIdx.x*IMPROVE_NT + tid;i<numPts;i+=IMPROVE_GROUPS*IMPROVE_NT) {
    const float x = sift[i].xpos, y = sift[i].ypos, mx = sift[i].match_xpos, my = sift[i].match_ypos;
    if (sift[i].score<minScore || sift[i].ambiguity>maxAmbiguity)
      continue;
    if (!(ImproveError(a, x, y, mx, my)<limit))
      continue;
    numfit++;
    // Row for match_xpos: Y = (x, y, 1, 0, 0, 0, -x*mx, -y*mx), for match_ypos: (0, 0, 0, x, y, 1, -x*my, -y*my)
    const double u6 = - (double)x*mx, u7 = - (double)y*mx, v6 = - (double)x*my, v7 = - (double)y*my;
    const double U[5] = {x, y, 1.0, u6, u7}, V[5] = {x, y, 1.0, v6, v7};
    int k = 0;
    for (int r=0;r<3;r++)
      for (int c=r;c<5;c++, k++)
        acc[k] += U[c]*U[r];
    for (int r=0;r<3;r++)
      for (int c=r;c<5;c++, k++)
        acc[k] += V[c]*V[r];
    acc[24] += u6*u6;
    acc[24] += v6*v6;
    acc[25] += u7*u6;
    acc[25] += v7*v6;
    acc[26] += u7*u7;
    acc[26] += v7*v7;
    for (int r=0;r<3;r++) {
      acc[27 + r] += U[r]*mx;
      acc[30 + r] += V[r]*my;
    }
    acc[33] += u6*mx;
    acc[33] += v6*my;
    acc[34] += u7*mx;
    acc[34] += v7*my;
  }
  for (int k=0;k<IMPROVE_NV;k++) {
    double v = acc[k];
    for (int d=16;d>0;d/=2)
      v += __shfl_xor_sync(0xffffffff, v, d);
    if (lane==0)
      red[tid/32][k] = v;
  }
  for (int d=16;d>0;d/=2)
    numfit += __shfl_xor_sync(0xffffffff, numfit, d);
  if (lane==0)
    counts[tid/32] = numfit;
  __syncthreads();
  if (tid<IMPROVE_NV) {
    double v = red[0][tid];
    for (int w=1;w<IMPROVE_NT/32;w++)
      v += red[w][tid];
    st->partial[blockIdx.x][tid] = v;
  }
  if (tid==0) {
    int n = 0;
    for (int w=0;w<IMPROVE_NT/32;w++)
      n += counts[w];
    st->counts[blockIdx.x] = n;   // inliers of this iteration (ImproveFinal overwrites them)
  }
}

// One warp: sums of the partial results, then the normal equations M A = b in shared
// memory, and their Cholesky factorization with the column updates spread over the lanes.
// With fewer than 4 inliers, or if M is not positive definite, the homography is kept,
// as in ImproveHomography, and the next iterations, which have the same inliers, keep it
// too.
__global__ void ImproveSolve(ImproveState *st)
{
  int numfit = 0;
  for (int g=0;g<IMPROVE_GROUPS;g++)
    numfit += st->counts[g];
  if (numfit<4)
    return;
  __shared__ double L[8][8];
  __shared__ double b[8];
  __shared__ double sum[IMPROVE_NV];
  __shared__ bool ok;
  const int tid = threadIdx.x;
  for (int k=tid;k<IMPROVE_NV;k+=32) {
    double v = st->partial[0][k];
    for (int g=1;g<IMPROVE_GROUPS;g++)
      v += st->partial[g][k];
    sum[k] = v;
  }
  for (int k=tid;k<64;k+=32)
    L[k/8][k%8] = 0.0;
  __syncwarp();
  if (tid==0) {
    const int map[5] = {0, 1, 2, 6, 7}, map2[5] = {3, 4, 5, 6, 7};
    int k = 0;
    for (int r=0;r<3;r++)
      for (int c=r;c<5;c++, k++)
        L[map[r]][map[c]] = L[map[c]][map[r]] = sum[k];
    for (int r=0;r<3;r++)
      for (int c=r;c<5;c++, k++)
        L[map2[r]][map2[c]] = L[map2[c]][map2[r]] = sum[k];
    L[6][6] = sum[24];
    L[6][7] = L[7][6] = sum[25];
    L[7][7] = sum[26];
    for (int r=0;r<3;r++) {
      b[map[r]] = sum[27 + r];
      b[map2[r]] = sum[30 + r];
    }
    b[6] = sum[33];
    b[7] = sum[34];
    ok = true;
  }
  __syncwarp();
  for (int j=0;j<8;j++) {
    if (tid==0) {
      double s = L[j][j];
      for (int k=0;k<j;k++)
        s -= L[j][k]*L[j][k];
      if (!(s>0.0))
        ok = false;
      L[j][j] = sqrt(s);
    }
    __syncwarp();
    if (!ok)
      break;
    const int i = j + 1 + tid;
    if (i<8) {
      double t = L[i][j];
      for (int k=0;k<j;k++)
        t -= L[i][k]*L[j][k];
      L[i][j] = t / L[j][j];
    }
    __syncwarp();
  }
  if (tid!=0 || !ok)
    return;
  double A[8];
  {
    double z[8];
    for (int i=0;i<8;i++) {
      double t = b[i];
      for (int k=0;k<i;k++)
        t -= L[i][k]*z[k];
      z[i] = t / L[i][i];
    }
    for (int i=7;i>=0;i--) {
      double t = z[i];
      for (int k=i+1;k<8;k++)
        t -= L[k][i]*A[k];
      A[i] = t / L[i][i];
    }
  }
  for (int i=0;i<8;i++)
    st->A[i] = A[i];
}

__global__ void ImproveFinal(SiftPoint *sift, int numPts, ImproveState *st, float limit, const int *d_numPts)
{
  if (d_numPts)
    numPts = *d_numPts;
  __shared__ int counts[IMPROVE_NT/32];
  const int tid = threadIdx.x;
  double a[8];
  for (int k=0;k<8;k++)
    a[k] = st->A[k];
  int numfit = 0;
  for (int i=blockIdx.x*IMPROVE_NT + tid;i<numPts;i+=IMPROVE_GROUPS*IMPROVE_NT) {
    float err = ImproveError(a, sift[i].xpos, sift[i].ypos, sift[i].match_xpos, sift[i].match_ypos);
    if (err<limit)
      numfit++;
    sift[i].match_error = sqrtf(err);
  }
  for (int d=16;d>0;d/=2)
    numfit += __shfl_xor_sync(0xffffffff, numfit, d);
  if (tid%32==0)
    counts[tid/32] = numfit;
  __syncthreads();
  if (tid==0) {
    int n = 0;
    for (int w=0;w<IMPROVE_NT/32;w++)
      n += counts[w];
    st->counts[blockIdx.x] = n;
  }
}

__global__ void ImproveCount(ImproveState *st, HomographyResult *res)
{
  if (threadIdx.x!=0)
    return;
  int n = 0;
  for (int g=0;g<IMPROVE_GROUPS;g++)
    n += st->counts[g];
  for (int k=0;k<8;k++)
    res->h[k] = st->A[k];
  res->h[8] = 1.0f;
  res->count = n;
}

//================= Host matching functions =====================//

// FindHomography and FindHomographySeeded: the RANSAC samples are drawn on the CPU, with
// rand(), or with the seeded generator of cudaSiftRandom.h if seed is not NULL
static double FindHomographyCPU(SiftData &data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh, const unsigned int *seed)
{
  *numMatches = 0;
  homography[0] = homography[4] = homography[8] = 1.0f;
  homography[1] = homography[2] = homography[3] = 0.0f;
  homography[5] = homography[6] = homography[7] = 0.0f;
#ifdef MANAGEDMEM
  SiftPoint *d_sift = data.m_data;
#else
  if (data.d_data==NULL)
    return 0.0f;
  SiftPoint *d_sift = data.d_data;
#endif
  TimerGPU timer(0);
  numLoops = iDivUp(numLoops,16)*16;
  int numPts = data.numPts;
  if (numPts<8)
    return 0.0f;
  int numPtsUp = iDivUp(numPts, 16)*16;
  float *d_coord, *d_homo;
  int *d_randPts, *h_randPts;
  int randSize = 4*sizeof(int)*numLoops;
  int szFl = sizeof(float);
  int szPt = sizeof(SiftPoint);
  safeCall(cudaMalloc((void **)&d_coord, 4*sizeof(float)*numPtsUp));
  safeCall(cudaMalloc((void **)&d_randPts, randSize));
  safeCall(cudaMalloc((void **)&d_homo, 8*sizeof(float)*numLoops));
  h_randPts = (int*)malloc(randSize);
  float *h_scores = (float *)malloc(sizeof(float)*numPtsUp);
  float *h_ambiguities = (float *)malloc(sizeof(float)*numPtsUp);
  safeCall(cudaMemcpy2D(h_scores, szFl, &d_sift[0].score, szPt, szFl, numPts, cudaMemcpyDeviceToHost));
  safeCall(cudaMemcpy2D(h_ambiguities, szFl, &d_sift[0].ambiguity, szPt, szFl, numPts, cudaMemcpyDeviceToHost));
  int *validPts = (int *)malloc(sizeof(int)*numPts);
  int numValid = 0;
  for (int i=0;i<numPts;i++) {
    if (h_scores[i]>minScore && h_ambiguities[i]<maxAmbiguity)
      validPts[numValid++] = i;
  }
  free(h_scores);
  free(h_ambiguities);
  if (numValid>=8) {
    // Coordinates of the valid points only, so that the inliers are counted among the
    // matches the hypotheses are sampled from
    int numValidUp = iDivUp(numValid, 16)*16;
    float *h_coord = (float *)malloc(4*sizeof(float)*numPts);
    float *h_valid = (float *)calloc(4*numValidUp, sizeof(float));
    safeCall(cudaMemcpy2D(&h_coord[0*numPts], szFl, &d_sift[0].xpos, szPt, szFl, numPts, cudaMemcpyDeviceToHost));
    safeCall(cudaMemcpy2D(&h_coord[1*numPts], szFl, &d_sift[0].ypos, szPt, szFl, numPts, cudaMemcpyDeviceToHost));
    safeCall(cudaMemcpy2D(&h_coord[2*numPts], szFl, &d_sift[0].match_xpos, szPt, szFl, numPts, cudaMemcpyDeviceToHost));
    safeCall(cudaMemcpy2D(&h_coord[3*numPts], szFl, &d_sift[0].match_ypos, szPt, szFl, numPts, cudaMemcpyDeviceToHost));
    for (int c=0;c<4;c++)
      for (int i=0;i<numValid;i++)
        h_valid[i+c*numValidUp] = h_coord[validPts[i]+c*numPts];
    safeCall(cudaMemcpy(d_coord, h_valid, 4*sizeof(float)*numValidUp, cudaMemcpyHostToDevice));
    free(h_coord);
    bool *h_sameOrientation = (bool *)malloc(sizeof(bool)*numLoops);
    for (int i=0;i<numLoops;i++) {
      int p[4];
      if (seed)
        SiftRandomSample4(*seed, i, numValid, p);
      else {
        p[0] = rand() % numValid;
        p[1] = rand() % numValid;
        p[2] = rand() % numValid;
        p[3] = rand() % numValid;
        while (p[1]==p[0]) p[1] = rand() % numValid;
        while (p[2]==p[0] || p[2]==p[1]) p[2] = rand() % numValid;
        while (p[3]==p[0] || p[3]==p[1] || p[3]==p[2]) p[3] = rand() % numValid;
      }
      for (int k=0;k<4;k++)
        h_randPts[i+k*numLoops] = p[k];
      h_sameOrientation[i] = SameOrientation(h_valid, numValidUp, p);
    }
    free(h_valid);
    safeCall(cudaMemcpy(d_randPts, h_randPts, randSize, cudaMemcpyHostToDevice));
    ComputeHomographies<<<numLoops/16, 16>>>(d_coord, d_randPts, d_homo, numValidUp);
    safeCall(cudaDeviceSynchronize());
    checkMsg("ComputeHomographies() execution failed\n");
    dim3 blocks(1, numLoops/TESTHOMO_LOOPS);
    dim3 threads(TESTHOMO_TESTS, TESTHOMO_LOOPS);
    TestHomographies<<<blocks, threads>>>(d_coord, d_homo, d_randPts, numValid, numValidUp, thresh*thresh);
    safeCall(cudaDeviceSynchronize());
    checkMsg("TestHomographies() execution failed\n");
    safeCall(cudaMemcpy(h_randPts, d_randPts, sizeof(int)*numLoops, cudaMemcpyDeviceToHost));
    int maxIndex = -1, maxCount = -1;
    for (int i=0;i<numLoops;i++) 
      if (h_sameOrientation[i] && h_randPts[i]>maxCount) {
	maxCount = h_randPts[i];
	maxIndex = i;
      }
    free(h_sameOrientation);
    if (maxIndex>=0) {
      *numMatches = maxCount;
      safeCall(cudaMemcpy2D(homography, szFl, &d_homo[maxIndex], sizeof(float)*numLoops, szFl, 8, cudaMemcpyDeviceToHost));
    }
  }
  free(validPts);
  free(h_randPts);
  safeCall(cudaFree(d_homo));
  safeCall(cudaFree(d_randPts));
  safeCall(cudaFree(d_coord));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("FindHomography time =         %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

// FindHomography and FindHomographySeeded sample the matches on the CPU: not in a batch
static void NotInBatch()
{
  if (InBatch()) {
    fprintf(stderr, "CudaSift: FindHomography and FindHomographySeeded cannot be called in a batch (BeginSiftBatch)\n");
    exit(EXIT_FAILURE);
  }
  CompleteBatch();
}

double FindHomography(SiftData &data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
  NotInBatch();
  return FindHomographyCPU(data, homography, numMatches, numLoops, minScore, maxAmbiguity, thresh, NULL);
}

double FindHomographySeeded(SiftData &data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh, unsigned int seed)
{
  NotInBatch();
  return FindHomographyCPU(data, homography, numMatches, numLoops, minScore, maxAmbiguity, thresh, &seed);
}

static void SetIdentity(float *homography)
{
  for (int i=0;i<9;i++)
    homography[i] = (i%4==0 ? 1.0f : 0.0f);
}

double FindHomographyGPU(SiftData &data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh, unsigned int seed)
{
  const bool batched = InBatch();
  if (!batched)
    CompleteBatch();
  *numMatches = 0;
  SetIdentity(homography);
#ifdef MANAGEDMEM
  SiftPoint *d_sift = data.m_data;
#else
  SiftPoint *d_sift = data.d_data;
#endif
  int numPts = data.numPts;
  if (d_sift==NULL || (!batched && numPts<8) || data.maxPts<8)
    return 0.0;
  // In a batch, the point count is on the GPU, and numPts is the capacity, for the scratch
  // memory and the grids (with fewer than 8 points, HomographyBest returns the identity)
  const int *d_numPts = NULL;
  if (batched) {
    d_numPts = BatchCount(data);
    numPts = data.maxPts;
  }
  auto lock = ScratchLock();
  TimerGPU timer(0);
  numLoops = iDivUp(numLoops, 16)*16;
  const int stride = iDivUp(numPts, 16)*16;
  float *d_coord = (float *)Scratch(SCRATCH_COORD, 8*sizeof(float)*stride);   // all points, then the valid ones
  float *d_validCoord = d_coord + 4*stride;
  int *d_valid = (int *)Scratch(SCRATCH_VALID, sizeof(int)*numPts + sizeof(int));   // valid, numValid
  int *d_numValid = d_valid + numPts;
  int *d_randPts = (int *)Scratch(SCRATCH_RANDPTS, 5*sizeof(int)*numLoops);   // samples (also the counts), orientation checks
  int *d_sameOrientation = d_randPts + 4*numLoops;
  float *d_homo = (float *)Scratch(SCRATCH_HOMO, 8*sizeof(float)*numLoops);
  int device;
  safeCall(cudaGetDevice(&device));
  DeviceScratch &s = DeviceScratchOf(device);
  HomographyResult *d_res = (batched ? BatchResult(s) : (HomographyResult *)Scratch(SCRATCH_RESULT, sizeof(HomographyResult)));
  HomographyPrepare<<<1, 1024>>>(d_sift, numPts, d_coord, d_valid, d_numValid, stride, minScore, maxAmbiguity, d_numPts);
  HomographyGather<<<iDivUp(numPts, 256), 256>>>(d_coord, d_valid, d_numValid, d_validCoord, stride);
  HomographySample<<<iDivUp(numLoops, 64), 64>>>(d_validCoord, d_numValid, d_randPts, d_sameOrientation, numLoops, stride, seed);
  ComputeHomographies<<<numLoops/16, 16>>>(d_validCoord, d_randPts, d_homo, stride);
  TestHomographiesGPU<<<dim3(1, iDivUp(numLoops, 64)), dim3(16, 16)>>>(d_validCoord, d_homo, d_randPts, d_numValid, stride, numLoops, thresh*thresh);
  HomographyBest<<<1, 1024>>>(d_randPts, d_sameOrientation, d_homo, d_numValid, d_res, numLoops);
  checkMsg("FindHomographyGPU() execution failed\n");
  if (batched) {
    s.homographies[homography] = d_res;   // for ImproveHomographyGPU later in the batch
    AfterBatch([=]() {
      HomographyResult res;
      safeCall(cudaMemcpy(&res, d_res, sizeof(res), cudaMemcpyDeviceToHost));
      memcpy(homography, res.h, sizeof(res.h));
      *numMatches = res.count;
    });
    return 0.0;
  }
  HomographyResult res;
  safeCall(cudaMemcpy(&res, d_res, sizeof(res), cudaMemcpyDeviceToHost));
  memcpy(homography, res.h, sizeof(res.h));
  *numMatches = res.count;
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("FindHomographyGPU time =      %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double ImproveHomographyGPU(SiftData &data, float *homography, int *numFit, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
  const bool batched = InBatch();
  if (!batched)
    CompleteBatch();
  *numFit = 0;
#ifdef MANAGEDMEM
  SiftPoint *d_sift = data.m_data;
#else
  SiftPoint *d_sift = data.d_data;
#endif
  const int numPts = data.numPts;
  if (d_sift==NULL)
    return 0.0;
  const int *d_numPts = (batched ? BatchCount(data) : NULL);   // in a batch, the count on the GPU
  auto lock = ScratchLock();
  TimerGPU timer(0);
  ImproveState *d_state = (ImproveState *)Scratch(SCRATCH_IMPROVE_STATE, sizeof(ImproveState));
  HomographyResult *d_in, *d_out;
  int device;
  safeCall(cudaGetDevice(&device));
  DeviceScratch &s = DeviceScratchOf(device);
  HomographyResult in;
  memcpy(in.h, homography, sizeof(in.h));
  in.count = 0;
  if (batched) {
    // The input: the result of a previous call of the batch for this homography, else the
    // homography on the host now
    d_in = BatchResult(s);
    d_out = BatchResult(s);
    auto it = s.homographies.find(homography);
    if (it!=s.homographies.end())
      safeCall(cudaMemcpyAsync(d_in, it->second, sizeof(HomographyResult), cudaMemcpyDeviceToDevice));
    else
      SetResult<<<1, 1>>>(d_in, in);
  } else {
    d_in = (HomographyResult *)Scratch(SCRATCH_IMPROVE_RESULT, 2*sizeof(HomographyResult));
    d_out = d_in + 1;
    safeCall(cudaMemcpy(d_in, &in, sizeof(in), cudaMemcpyHostToDevice));
  }
  const float limit = thresh*thresh;
  ImproveInit<<<1, 32>>>(d_in, d_state);
  for (int loop=0;loop<numLoops;loop++) {
    ImproveAccumulate<<<IMPROVE_GROUPS, IMPROVE_NT>>>(d_sift, numPts, d_state, minScore, maxAmbiguity, limit, d_numPts);
    ImproveSolve<<<1, 32>>>(d_state);
  }
  ImproveFinal<<<IMPROVE_GROUPS, IMPROVE_NT>>>(d_sift, numPts, d_state, limit, d_numPts);
  ImproveCount<<<1, 32>>>(d_state, d_out);
  checkMsg("ImproveHomographyGPU() execution failed\n");
  if (batched) {
    s.homographies[homography] = d_out;
    // The deferred work runs in call order, after that of the ExtractSift call that set numPts
    SiftData *pd = &data;
    AfterBatch([=]() {
      HomographyResult res;
      safeCall(cudaMemcpy(&res, d_out, sizeof(res), cudaMemcpyDeviceToHost));
      memcpy(homography, res.h, sizeof(res.h));
      *numFit = res.count;
#ifndef MANAGEDMEM
      if (pd->h_data!=NULL && pd->numPts>0)
        safeCall(cudaMemcpy2D(&pd->h_data[0].match_error, sizeof(SiftPoint), &pd->d_data[0].match_error, sizeof(SiftPoint), sizeof(float), pd->numPts, cudaMemcpyDeviceToHost));
#endif
    });
    return 0.0;
  }
  HomographyResult res;
  safeCall(cudaMemcpy(&res, d_out, sizeof(res), cudaMemcpyDeviceToHost));
  memcpy(homography, res.h, sizeof(res.h));
  *numFit = res.count;
#ifndef MANAGEDMEM
  if (data.h_data!=NULL && numPts>0)   // match_error, as ImproveHomography sets it on the host
    safeCall(cudaMemcpy2D(&data.h_data[0].match_error, sizeof(SiftPoint), &d_sift[0].match_error, sizeof(SiftPoint), sizeof(float), numPts, cudaMemcpyDeviceToHost));
#endif
  return timer.read();
}

double MatchSiftData(SiftData &data1, SiftData &data2)
{
  const bool batched = InBatch();
  if (!batched)
    CompleteBatch();
  int numPts1 = data1.numPts;
  int numPts2 = data2.numPts;
  if (!batched && (!numPts1 || !numPts2))
    return 0.0;
  if (batched && (!data1.maxPts || !data2.maxPts))   // no points
    return 0.0;
  auto lock = ScratchLock();   // Scratch, until the end of the call
  TimerGPU timer(0);
#ifdef MANAGEDMEM
  SiftPoint *sift1 = data1.m_data;
  SiftPoint *sift2 = data2.m_data;
#else
  if (data1.d_data==NULL || data2.d_data==NULL)
    return 0.0f;
  SiftPoint *sift1 = data1.d_data;
  SiftPoint *sift2 = data2.d_data;
#endif
  // In a batch, the point counts are on the GPU, and the grids and the scratch memory are
  // sized for the capacities
  const int *d_numPts1 = NULL, *d_numPts2 = NULL;
  if (batched) {
    d_numPts1 = BatchCount(data1);
    d_numPts2 = BatchCount(data2);
    numPts1 = data1.maxPts;
    numPts2 = data2.maxPts;
  }
  
// Original version with correlation and maximization in two different kernels
// Global memory reguirement: O(N^2)
#if 0
  float *d_corrData; 
  int corrWidth = iDivUp(numPts2, 16)*16;
  int corrSize = sizeof(float)*numPts1*corrWidth;
  safeCall(cudaMalloc((void **)&d_corrData, corrSize));
#if 0 // K40c 10.9ms, 1080 Ti 3.8ms
  dim3 blocks1(numPts1, iDivUp(numPts2, 16));
  dim3 threads1(16, 16); // each block: 1 points x 16 points
  MatchSiftPoints<<<blocks1, threads1>>>(sift1, sift2, d_corrData, numPts1, numPts2);
#else // K40c 7.6ms, 1080 Ti 1.4ms
  dim3 blocks(iDivUp(numPts1,16), iDivUp(numPts2, 16));
  dim3 threads(16, 16); // each block: 16 points x 16 points
  MatchSiftPoints2<<<blocks, threads>>>(sift1, sift2, d_corrData, numPts1, numPts2);
#endif
  safeCall(cudaDeviceSynchronize());
  dim3 blocksMax(iDivUp(numPts1, 16));
  dim3 threadsMax(16, 16);
  FindMaxCorr<<<blocksMax, threadsMax>>>(d_corrData, sift1, sift2, numPts1, corrWidth, sizeof(SiftPoint));
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr() execution failed\n");
  safeCall(cudaFree(d_corrData));
#endif

// Version suggested by Nicholas Lin with combined correlation and maximization
// Global memory reguirement: O(N)
#if 0 // K40c 51.2ms, 1080 Ti 9.6ms
  int block_dim = 16;
  float *d_corrData;
  int corrSize = numPts1 * block_dim * 2;
  safeCall(cudaMalloc((void **)&d_corrData, sizeof(float) * corrSize));
  dim3 blocks(iDivUp(numPts1, block_dim));
  dim3 threads(block_dim, block_dim); 
  FindMaxCorr3<<<blocks, threads >>>(d_corrData, sift1, sift2, numPts1, numPts2);
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr3() execution failed\n");
  safeCall(cudaFree(d_corrData));
#endif

// Combined version with no global memory requirement using one 1 point per block
#if 0 // K40c 8.9ms, 1080 Ti 2.1ms, 2080 Ti 1.0ms
  dim3 blocksMax(numPts1);
  dim3 threadsMax(FMC2W, FMC2H);
  FindMaxCorr2<<<blocksMax, threadsMax>>>(sift1, sift2, numPts1, numPts2);
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr2() execution failed\n");
#endif
  
// Combined version with no global memory requirement using one FMC2H points per block
#if 0 // K40c 9.2ms, 1080 Ti 1.3ms, 2080 Ti 1.1ms
  dim3 blocksMax2(iDivUp(numPts1, FMC2H));
  dim3 threadsMax2(FMC2W, FMC2H);
  FindMaxCorr4<<<blocksMax2, threadsMax2>>>(sift1, sift2, numPts1, numPts2);
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr4() execution failed\n");
#endif

// Combined version with no global memory requirement using global locks
#if 1
  dim3 blocksMax3(iDivUp(numPts1, 16), iDivUp(numPts2, 512));
  dim3 threadsMax3(16, 16);
  CleanMatches<<<iDivUp(numPts1, 64), 64>>>(sift1, numPts1, numPts2, d_numPts1, d_numPts2);
  int mode = 11;
  if (mode==5)// K40c 5.0ms, 1080 Ti 1.2ms, 2080 Ti 0.83ms
    FindMaxCorr5<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  else if (mode==6) {                    // 2080 Ti 0.89ms
    threadsMax3 = dim3(32, 16);
    FindMaxCorr6<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  } else if (mode==7)                    // 2080 Ti 0.50ms  
    FindMaxCorr7<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  else if (mode==8) {                    // 2080 Ti 0.45ms
    blocksMax3 = dim3(iDivUp(numPts1, FMC_BW), iDivUp(numPts2, FMC_GH));
    threadsMax3 = dim3(FMC_NW, FMC_NH);
    FindMaxCorr8<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  } else if (mode==9) {                  // 2080 Ti 0.46ms
    blocksMax3 = dim3(iDivUp(numPts1, FMC_BW), iDivUp(numPts2, FMC_GH));
    threadsMax3 = dim3(FMC_NW, FMC_NH);
    FindMaxCorr9<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  } else if (mode==10) {                 // 2080 Ti 0.24ms
    blocksMax3 = dim3(iDivUp(numPts1, M7W));
    threadsMax3 = dim3(M7W, M7H/M7R);
    FindMaxCorr10<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  } else if (mode==11) {                 // T4 (2060 pts): FindMaxCorr10 0.44ms, FindMaxCorr11 0.27ms
    // Per-tile results buffer (Scratch), and number of resident blocks per device
    int device;
    safeCall(cudaGetDevice(&device));
    int &maxBlocks = DeviceScratchOf(device).maxBlocks;   // under the ScratchLock
    if (maxBlocks==0) {
      int numSMs, blocksPerSM;
      safeCall(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
      safeCall(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, FindMaxCorr11, M11_NT, 0));
      maxBlocks = numSMs*blocksPerSM;
    }
    int tiles1 = iDivUp(numPts1, M11_T);
    int tiles2 = iDivUp(numPts2, M11_T);
    // Outside batches, one tile per chunk; in a batch, the counts are on the GPU, and
    // numPts1, numPts2 are the capacities
    const int numChunks = (batched ? M11_CHUNKS : tiles2);
    const int stride = numPts1;
    float4 *d_best = (float4 *)Scratch(SCRATCH_BEST, sizeof(float4)*numChunks*stride);
    FindMaxCorr11<<<min(tiles1*tiles2, maxBlocks), M11_NT>>>(sift1, sift2, numPts1, numPts2, d_best, numChunks, stride, d_numPts1, d_numPts2);
    MergeMaxCorr11<<<iDivUp(numPts1, 128), 128>>>(sift1, sift2, numPts1, numPts2, d_best, numChunks, stride, d_numPts1, d_numPts2);
  }
  if (!batched)
    safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr5() execution failed\n");
#endif

#ifndef MANAGEDMEM
  auto copyMatches = [](SiftData *data) {   // score, ambiguity, match, match_xpos, match_ypos
    if (data->h_data!=NULL && data->numPts>0) {
      float *h_ptr = &data->h_data[0].score;
      float *d_ptr = &data->d_data[0].score;
      safeCall(cudaMemcpy2D(h_ptr, sizeof(SiftPoint), d_ptr, sizeof(SiftPoint), 5*sizeof(float), data->numPts, cudaMemcpyDeviceToHost));
    }
  };
  if (batched) {
    // The deferred work runs in call order, after that of the ExtractSift call that set numPts
    SiftData *d1 = &data1;
    AfterBatch([=]() { copyMatches(d1); });
  } else
    copyMatches(&data1);
#endif
  if (batched)
    return 0.0;

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("MatchSiftData time =          %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}		 
  
