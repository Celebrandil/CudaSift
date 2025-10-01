#include "cudaSift.h"
#include "cudautils.h"

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


__global__ void CleanMatches(SiftPoint *sift1, int numPts1)
{
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
  int *d_counts, int numPts, float thresh2)
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
    float x1 = d_coord[i+0*numPts];
    float y1 = d_coord[i+1*numPts];
    float x2 = d_coord[i+2*numPts];
    float y2 = d_coord[i+3*numPts];
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

//================= Host matching functions =====================//

double FindHomography(SiftData &data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh)
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
    for (int i=0;i<numLoops;i++) {
      int p1 = rand() % numValid;
      int p2 = rand() % numValid;
      int p3 = rand() % numValid;
      int p4 = rand() % numValid;
      while (p2==p1) p2 = rand() % numValid;
      while (p3==p1 || p3==p2) p3 = rand() % numValid;
      while (p4==p1 || p4==p2 || p4==p3) p4 = rand() % numValid;
      h_randPts[i+0*numLoops] = validPts[p1];
      h_randPts[i+1*numLoops] = validPts[p2];
      h_randPts[i+2*numLoops] = validPts[p3];
      h_randPts[i+3*numLoops] = validPts[p4];
    }
    safeCall(cudaMemcpy(d_randPts, h_randPts, randSize, cudaMemcpyHostToDevice));
    safeCall(cudaMemcpy2D(&d_coord[0*numPtsUp], szFl, &d_sift[0].xpos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    safeCall(cudaMemcpy2D(&d_coord[1*numPtsUp], szFl, &d_sift[0].ypos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    safeCall(cudaMemcpy2D(&d_coord[2*numPtsUp], szFl, &d_sift[0].match_xpos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    safeCall(cudaMemcpy2D(&d_coord[3*numPtsUp], szFl, &d_sift[0].match_ypos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    ComputeHomographies<<<numLoops/16, 16>>>(d_coord, d_randPts, d_homo, numPtsUp);
    safeCall(cudaDeviceSynchronize());
    checkMsg("ComputeHomographies() execution failed\n");
    dim3 blocks(1, numLoops/TESTHOMO_LOOPS);
    dim3 threads(TESTHOMO_TESTS, TESTHOMO_LOOPS);
    TestHomographies<<<blocks, threads>>>(d_coord, d_homo, d_randPts, numPtsUp, thresh*thresh);
    safeCall(cudaDeviceSynchronize());
    checkMsg("TestHomographies() execution failed\n");
    safeCall(cudaMemcpy(h_randPts, d_randPts, sizeof(int)*numLoops, cudaMemcpyDeviceToHost));
    int maxIndex = -1, maxCount = -1;
    for (int i=0;i<numLoops;i++) 
      if (h_randPts[i]>maxCount) {
	maxCount = h_randPts[i];
	maxIndex = i;
      }
    *numMatches = maxCount;
    safeCall(cudaMemcpy2D(homography, szFl, &d_homo[maxIndex], sizeof(float)*numLoops, szFl, 8, cudaMemcpyDeviceToHost));
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


double MatchSiftData(SiftData &data1, SiftData &data2)
{
  TimerGPU timer(0);
  int numPts1 = data1.numPts;
  int numPts2 = data2.numPts;
  if (!numPts1 || !numPts2) 
    return 0.0;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = data1.m_data;
  SiftPoint *sift2 = data2.m_data;
#else
  if (data1.d_data==NULL || data2.d_data==NULL)
    return 0.0f;
  SiftPoint *sift1 = data1.d_data;
  SiftPoint *sift2 = data2.d_data;
#endif
  
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
  CleanMatches<<<iDivUp(numPts1, 64), 64>>>(sift1, numPts1);
  int mode = 10;
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
  }
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr5() execution failed\n");
#endif

  if (data1.h_data!=NULL) {
    float *h_ptr = &data1.h_data[0].score;
    float *d_ptr = &data1.d_data[0].score;
    safeCall(cudaMemcpy2D(h_ptr, sizeof(SiftPoint), d_ptr, sizeof(SiftPoint), 5*sizeof(float), data1.numPts, cudaMemcpyDeviceToHost));
  }

  double gpuTime = timer.read();
#ifndef VERBOSE
  printf("MatchSiftData time =          %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}		 
  
