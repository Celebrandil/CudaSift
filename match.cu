//**********************************************************//
//   Matching test code by Marten Bjorkman aka Celebrandil  //
//                                                          //
//   The code includes an example of gradual optimization   //
//   of a kernel for matching two sets of 16K 128D points.  //
//   You are welcome to the code for educational purposes.  //
//                                                          //
//            Fairlight - When Dreams Come True             //
// https://www.youtube.com/channel/UCdHiji77FlppuNK6xemrsVA //
//**********************************************************//

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <immintrin.h>
#include "cudautils.h"

#define RUNCPU 1
#define CHECK  1
#define NPTS (2048*8)
#define NDIM 128

#define M1W  128
#define M2W   16
#define M2H   16
#define M5W   16
#define M5H   16
#define M5R    4
#define M7W   32
#define M7H   32
#define M7R    4


/*
Data size:   16 MB
Allocate:    1.01194 ms
Upload:      3.69939 ms  4.32503 MB/ms
MatchCPU1:   34649.6 ms  1.89139 Gflops
MatchCPU2:   3064.36 ms  21.3866 Gflops
MatchCPU3:   184.762 ms  354.706 Gflops
MatchGPU1:   641.828 ms  102.108 Gflops
MatchGPU2:   148.020 ms  442.752 Gflops
MatchGPU3:   31.9609 ms  2050.50 Gflops
MatchGPU4:   29.7891 ms  2200.00 Gflops
MatchGPU5:   17.1484 ms  3821.69 Gflops
MatchGPU6:   16.3516 ms  4007.94 Gflops
MatchGPU7:   14.7995 ms  4428.27 Gflops
MatchGPU8:   10.5291 ms  6224.28 Gflops
Download:    0.16016 ms  0.780488 MB/ms
*/

void MatchC1(float *h_pts1, float *h_pts2, float *h_score, int *h_index)
{
  std::memset(h_score, 0, sizeof(float)*NPTS);
  for (int p1=0;p1<NPTS;p1++) {
    for (int p2=0;p2<NPTS;p2++) {
      float score = 0.0f;
      for (int d=0;d<NDIM;d++)
	score += h_pts1[p1*NDIM + d]*h_pts2[p2*NDIM + d];
      if (score>h_score[p1]) {
	h_score[p1] = score;
	h_index[p1] = p2;
      }
    }
  }
}

void MatchC2(float *h_pts1, float *h_pts2, float *h_score, int *h_index)
{
#define BSIZ  256
  std::memset(h_score, 0, sizeof(float)*NPTS);
  for (int b1=0;b1<NPTS;b1+=BSIZ) {
    for (int b2=0;b2<NPTS;b2+=BSIZ) {
      for (int p1=b1;p1<b1+BSIZ;p1++) {
	float *pt1 = &h_pts1[p1*NDIM];
	for (int p2=b2;p2<b2+BSIZ;p2++) {
	  float *pt2 = &h_pts2[p2*NDIM];
	  __m256 score8 = _mm256_setzero_ps();
	  for (int d=0;d<NDIM;d+=8) {
	    __m256 v1 = _mm256_load_ps(pt1 + d);
	    __m256 v2 = _mm256_load_ps(pt2 + d);
	    score8 = _mm256_fmadd_ps(v1, v2, score8);
	  }
	  score8 = _mm256_add_ps(score8, _mm256_permute2f128_ps(score8, score8, 1));
	  score8 = _mm256_hadd_ps(score8, score8);
	  float score = _mm256_cvtss_f32(_mm256_hadd_ps(score8, score8));
	  if (score>h_score[p1]) {
	    h_score[p1] = score;
	    h_index[p1] = p2;
	  }
	}
      }
    }
  }
}

void MatchC3(float *h_pts1, float *h_pts2, float *h_score, int *h_index)
{
#define BSIZ  256
  std::memset(h_score, 0, sizeof(float)*NPTS);
#pragma omp parallel for
  for (int b1=0;b1<NPTS;b1+=BSIZ) {
    for (int b2=0;b2<NPTS;b2+=BSIZ) {
      for (int p1=b1;p1<b1+BSIZ;p1++) {
	float *pt1 = &h_pts1[p1*NDIM];
	for (int p2=b2;p2<b2+BSIZ;p2++) {
	  float *pt2 = &h_pts2[p2*NDIM];
	  __m256 score8 = _mm256_setzero_ps();
	  for (int d=0;d<NDIM;d+=8) {
	    __m256 v1 = _mm256_load_ps(pt1 + d);
	    __m256 v2 = _mm256_load_ps(pt2 + d);
	    score8 = _mm256_fmadd_ps(v1, v2, score8);
	  }
	  score8 = _mm256_add_ps(score8, _mm256_permute2f128_ps(score8, score8, 1));
	  score8 = _mm256_hadd_ps(score8, score8);
	  float score = _mm256_cvtss_f32(_mm256_hadd_ps(score8, score8));
	  if (score>h_score[p1]) {
	    h_score[p1] = score;
	    h_index[p1] = p2;
	  }
	}
      }
    }
  }
}

void CheckMatches(int *h_index, int *h_index2, float *h_score, float *h_score2)
{
  int ndiff = 0;
  for (int i=0;i<NPTS;i++) {
    ndiff += (h_index[i] != h_index2[i]);
    if (h_index[i] != h_index2[i])
      std::cout << "  " << i << " " << h_index[i] << " " << h_index2[i] << " " << h_score[i] << " " << h_score2[i] << std::endl;
  }
  std::cout << "Number of incorrect matches: " << ndiff << std::endl;
}
      

__global__ void Match1(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
  int p1 = threadIdx.x + M1W*blockIdx.x;
  float max_score = 0.0f;
  int index = -1;
  
  for (int p2=0;p2<NPTS;p2++) {
    float score = 0.0f;
    for (int d=0;d<NDIM;d++)
      score += d_pts1[p1*NDIM + d]*d_pts2[p2*NDIM + d];
    if (score>max_score) {
      max_score = score;
      index = p2;
    }
  }
  
  d_score[p1] = max_score;
  d_index[p1] = index;
}

__global__ void Match2(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
  __shared__ float buffer1[M2W*NDIM];  //%%%%
  __shared__ float buffer2[M2H*NDIM];  //%%%%
  __shared__ float scores[M2W*M2H];    //%%%%
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int idx = tx + M2W*ty;
  int bp1 = M2W*blockIdx.x;
  if (ty<M2W)
    for (int d=tx;d<NDIM;d+=M2W)
      for (int j=ty;j<M2W;j+=M2H)
	buffer1[j*NDIM + d] = d_pts1[(bp1 + j)*NDIM + d];   //%%%%
  __syncthreads();
  
  float max_score = 0.0f;
  int index = -1;
  for (int bp2=0;bp2<NPTS;bp2+=M2H) {
    for (int d=tx;d<NDIM;d+=M2W)
      buffer2[ty*NDIM + d] = d_pts2[(bp2 + ty)*NDIM + d]; //%%%%
    __syncthreads();

    float score = 0.0f;
    for (int d=0;d<NDIM;d++) 
      score += buffer1[tx*NDIM + d]*buffer2[ty*NDIM + d];   //%%%%
    scores[idx] = score;
    __syncthreads();
    
    if (ty==0) {
      for (int i=0;i<M2H;i++) {
	if (scores[i*M2W + tx]>max_score) {
	  max_score = scores[i*M2W + tx];
	  index = bp2 + i;
	}
      }
    }
    __syncthreads();
  }
  
  if (ty==0) {
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}


__global__ void Match3(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
  __shared__ float buffer1[M2W*(NDIM + 1)]; //%%%%
  __shared__ float buffer2[M2H*NDIM];
  __shared__ float scores[M2W*M2H];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int idx = tx + M2W*ty;
  int bp1 = M2W*blockIdx.x;
  if (ty<M2W)
    for (int d=tx;d<NDIM;d+=M2W)
      for (int j=ty;j<M2W;j+=M2H)
	buffer1[j*(NDIM + 1) + d] = d_pts1[(bp1 + j)*NDIM + d]; //%%%%
  __syncthreads();
  
  float max_score = 0.0f;
  int index = -1;
  for (int bp2=0;bp2<NPTS;bp2+=M2H) {
    for (int d=tx;d<NDIM;d+=M2W)
      buffer2[ty*NDIM + d] = d_pts2[(bp2 + ty)*NDIM + d];
    __syncthreads();

    float score = 0.0f;
    for (int d=0;d<NDIM;d++) 
      score += buffer1[tx*(NDIM + 1) + d]*buffer2[ty*NDIM + d]; //%%%%
    scores[idx] = score;
    __syncthreads();
    
    if (ty==0) {
      for (int i=0;i<M2H;i++) {
	if (scores[i*M2W + tx]>max_score) {
	  max_score = scores[i*M2W + tx];
	  index = bp2 + i;
	}
      }
    }
    __syncthreads();
  }
  
  if (ty==0) {
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}


__global__ void Match4(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
  __shared__ float4 buffer1[M2W*(NDIM/4 + 1)];  //%%%%
  __shared__ float4 buffer2[M2H*NDIM/4];        //%%%%
  __shared__ float scores[M2W*M2H];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int idx = tx + M2W*ty;
  int bp1 = M2W*blockIdx.x;
  if (ty<M2W)
    for (int d=tx;d<NDIM/4;d+=M2W)
      for (int j=ty;j<M2W;j+=M2H)
	buffer1[j*(NDIM/4 + 1) + d] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d]; //%%%%
  __syncthreads();
  
  float max_score = 0.0f;
  int index = -1;
  for (int bp2=0;bp2<NPTS;bp2+=M2H) {
    for (int d=tx;d<NDIM/4;d+=M2W)
      buffer2[ty*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + ty)*(NDIM/4) + d]; //%%%%
    __syncthreads();

    float score = 0.0f;
    for (int d=0;d<NDIM/4;d++) {
      float4 v1 = buffer1[tx*(NDIM/4 + 1) + d]; //%%%%
      float4 v2 = buffer2[ty*(NDIM/4) + d];     //%%%%
      score += v1.x*v2.x; score += v1.y*v2.y;
      score += v1.z*v2.z; score += v1.w*v2.w;
    }
    scores[idx] = score;
    __syncthreads();
    
    if (ty==0) {
      for (int i=0;i<M2H;i++) {
	if (scores[i*M2W + tx]>max_score) {
	  max_score = scores[i*M2W + tx];
	  index = bp2 + i;
	}
      }
    }
    __syncthreads();
  }
  
  if (ty==0) {
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}

__global__ void Match5(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
  __shared__ float4 buffer1[M5W*(NDIM/4 + 1)]; 
  __shared__ float4 buffer2[M5H*NDIM/4];       
  __shared__ float scores[M5W*M5H];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bp1 = M5W*blockIdx.x;
  if (ty<M5W)
    for (int d=tx;d<NDIM/4;d+=M5W)
      for (int j=ty;j<M5W;j+=M5H)
	buffer1[j*(NDIM/4 + 1) + d] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];
  __syncthreads();
  
  float max_score = 0.0f;
  int index = -1;
  for (int bp2=0;bp2<NPTS;bp2+=M5H) {
    for (int d=tx;d<NDIM/4;d+=M5W)
      buffer2[ty*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + ty)*(NDIM/4) + d];
    __syncthreads();

    if (ty<M5H/M5R) {  //%%%%
      float score[M5R];                                    //%%%%
      for (int dy=0;dy<M5R;dy++)
	score[dy] = 0.0f;
      for (int d=0;d<NDIM/4;d++) {
	float4 v1 = buffer1[tx*(NDIM/4 + 1) + d];
	for (int dy=0;dy<M5R;dy++) {
	  float4 v2 = buffer2[(M5R*ty + dy)*(NDIM/4) + d];    //%%%%
	  score[dy] += v1.x*v2.x; score[dy] += v1.y*v2.y;
	  score[dy] += v1.z*v2.z; score[dy] += v1.w*v2.w;
	}
      }
      for (int dy=0;dy<M5R;dy++)
	scores[tx + M5W*(M5R*ty + dy)] = score[dy];
    }
    __syncthreads();
    
    if (ty==0) {
      for (int i=0;i<M5H;i++) {
	if (scores[i*M2W + tx]>max_score) {
	  max_score = scores[i*M5W + tx];
	  index = bp2 + i;
	}
      }
    }
    __syncthreads();
  }

  if (ty==0) {
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}


__global__ void Match6(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
  __shared__ float4 buffer1[M5W*(NDIM/4 + 1)]; 
  __shared__ float4 buffer2[M5H*NDIM/4];       
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bp1 = M5W*blockIdx.x;
  if (ty<M5W)
    for (int d=tx;d<NDIM/4;d+=M5W)
      for (int j=ty;j<M5W;j+=M5H)
	buffer1[j*(NDIM/4 + 1) + d] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];
  
  float max_score = 0.0f;
  int index = -1;    
  for (int bp2=0;bp2<NPTS;bp2+=M5H) {
    for (int d=tx;d<NDIM/4;d+=M5W)
      buffer2[ty*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + ty)*(NDIM/4) + d];
    __syncthreads();

    if (ty<M5H/M5R) {  
      float score[M5R];                                    
      for (int dy=0;dy<M5R;dy++)
	score[dy] = 0.0f;
      for (int d=0;d<NDIM/4;d++) {
	float4 v1 = buffer1[tx*(NDIM/4 + 1) + d];
	for (int dy=0;dy<M5R;dy++) {
	  float4 v2 = buffer2[(M5R*ty + dy)*(NDIM/4) + d];    
	  score[dy] += v1.x*v2.x; score[dy] += v1.y*v2.y;
	  score[dy] += v1.z*v2.z; score[dy] += v1.w*v2.w;
	}
      }
      for (int dy=0;dy<M5R;dy++) {
	if (score[dy]>max_score) {   //%%%%
	  max_score = score[dy];     
	  index = bp2 + M5R*ty + dy;               
	}
      }
    }
    __syncthreads();
  }

  float *scores = (float*)buffer1;
  int *indices = (int*)&scores[M5W*M5H/M5R];
  if (ty<M5H/M5R) {
    scores[ty*M5W + tx] = max_score;  //%%%%
    indices[ty*M5W + tx] = index;     //%%%%
  }
  __syncthreads();
  
  if (ty==0) {
    max_score = scores[tx];
    index = indices[tx];
    for (int y=0;y<M5H/M5R;y++)
      if (scores[y*M5W + tx]>max_score) {
	max_score = scores[y*M5W + tx]; //%%%%
	index = indices[y*M5W + tx];    //%%%%
      }
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}

__global__ void Match7(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
  __shared__ float4 buffer1[M7W*NDIM/4]; //%%%%
  __shared__ float4 buffer2[M7H*NDIM/4];       
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bp1 = M7W*blockIdx.x;
  for (int d=tx;d<NDIM/4;d+=M7W)
    for (int j=ty;j<M7W;j+=M7H/M7R)      //%%%%
      buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];
  
  float max_score = 0.0f;
  int index = -1;    
  for (int bp2=0;bp2<NPTS;bp2+=M7H) {
    for (int d=tx;d<NDIM/4;d+=M7W)
      for (int j=ty;j<M7H;j+=M7H/M7R)       //%%%%
	buffer2[j*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + d];
    __syncthreads();

    float score[M7R];                                    
    for (int dy=0;dy<M7R;dy++)
      score[dy] = 0.0f;
    for (int d=0;d<NDIM/4;d++) {
      float4 v1 = buffer1[tx*NDIM/4 + (d + tx)%(NDIM/4)];
      for (int dy=0;dy<M7R;dy++) {
	float4 v2 = buffer2[(M7R*ty + dy)*(NDIM/4) + d];    
	score[dy] += v1.x*v2.x; score[dy] += v1.y*v2.y;
	score[dy] += v1.z*v2.z; score[dy] += v1.w*v2.w;
      }
    }
    for (int dy=0;dy<M7R;dy++) {
      if (score[dy]>max_score) {   
	max_score = score[dy];     
	index = bp2 + M7R*ty + dy;               
      }
    }
    __syncthreads();
  }

  float *scores = (float*)buffer1;
  int *indices = (int*)&scores[M7W*M7H/M7R];
  scores[ty*M7W + tx] = max_score;  
  indices[ty*M7W + tx] = index;     
  __syncthreads();
  
  if (ty==0) {
    max_score = scores[tx];
    index = indices[tx];
    for (int y=0;y<M7H/M7R;y++)
      if (scores[y*M7W + tx]>max_score) {
	max_score = scores[y*M7W + tx]; 
	index = indices[y*M7W + tx];    
      }
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}

__global__ void Match8(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
  __shared__ float4 buffer1[M7W*NDIM/4]; 
  __shared__ float4 buffer2[M7H*NDIM/4];       
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bp1 = M7W*blockIdx.x;
  for (int d=tx;d<NDIM/4;d+=M7W)
    for (int j=ty;j<M7W;j+=M7H/M7R)     
      buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];

#define NRX 2
  float max_score[NRX];
  int index[NRX];
  for (int i=0;i<NRX;i++) {
    max_score[i] = 0.0f;
    index[i] = -1;
  }
  int idx = ty*M7W + tx;
  int ix = idx%(M7W/NRX);
  int iy = idx/(M7W/NRX);
  for (int bp2=0;bp2<NPTS;bp2+=M7H) {
    for (int d=tx;d<NDIM/4;d+=M7W)
      for (int j=ty;j<M7H;j+=M7H/M7R)       
	buffer2[j*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + d];
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
	    max_score[i] = score[dy][i];     
	    index[i] = bp2 + M7R*iy + dy;
	  }
	}
      }
    }
    __syncthreads();
  }

  float *scores = (float*)buffer1;
  int *indices = (int*)&scores[M7W*M7H/M7R];
  if (idx<M7W*M7H/M7R/NRX) {
    for (int i=0;i<NRX;i++) {
      scores[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];  
      indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
    }
  }
  __syncthreads();
  
  if (ty==0) {
    float max_score = scores[tx];
    int index = indices[tx];
    for (int y=0;y<M7H/M7R;y++)
      if (scores[y*M7W + tx]>max_score) {
	max_score = scores[y*M7W + tx]; 
	index = indices[y*M7W + tx];    
      }
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}

__global__ void Match8small(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
#define NRX 2
  __shared__ float4 buffer1[M7W*NDIM/4]; 
  __shared__ float4 buffer2[M7H*NDIM/4];       
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bp1 = M7W*blockIdx.x;
  for (int d=tx;d<NDIM/4;d+=M7W)
    for (int j=ty;j<M7W;j+=M7H/M7R/NRX)     
      buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];

  float max_score[NRX];
  int index[NRX];
  for (int i=0;i<NRX;i++) {
    max_score[i] = 0.0f;
    index[i] = -1;
  }
  int idx = ty*M7W + tx;
  int ix = idx%(M7W/NRX);
  int iy = idx/(M7W/NRX);
  for (int bp2=0;bp2<NPTS;bp2+=M7H) {
    for (int d=tx;d<NDIM/4;d+=M7W)
      for (int j=ty;j<M7H;j+=M7H/M7R/NRX)       
	buffer2[j*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + d];
    __syncthreads();

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
	  max_score[i] = score[dy][i];     
	  index[i] = bp2 + M7R*iy + dy;
	}
      }
    }
    __syncthreads();
  }

  float *scores = (float*)buffer1;
  int *indices = (int*)&scores[M7W*M7H/M7R];
  if (idx<M7W*M7H/M7R/NRX) {
    for (int i=0;i<NRX;i++) {
      scores[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];  
      indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
    }
  }
  __syncthreads();
  
  if (ty==0) {
    float max_score = scores[tx];
    int index = indices[tx];
    for (int y=0;y<M7H/M7R;y++)
      if (scores[y*M7W + tx]>max_score) {
	max_score = scores[y*M7W + tx]; 
	index = indices[y*M7W + tx];    
      }
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}

__global__ void Match8blocked(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
#define NRX 2
#define NUM (NRX*M7R)                       // 32*8 threads
  __shared__ float4 buffer1[M7W*NDIM/4];    // 32*32
  __shared__ float4 buffer2[M7H*NUM];       // 32*8
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bp1 = M7W*blockIdx.x;
  for (int d=tx;d<NDIM/4;d+=M7W)
    for (int j=ty;j<M7W;j+=M7H/M7R)     
      buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];

  float max_score[NRX];
  int index[NRX];
  for (int i=0;i<NRX;i++) {
    max_score[i] = 0.0f;
    index[i] = -1;
  }
  int idx = ty*M7W + tx;
  int ix = idx%(M7W/NRX);
  int iy = idx/(M7W/NRX);
  for (int bp2=0;bp2<NPTS;bp2+=M7H) {
    float score[M7R][NRX];                                    
    for (int dy=0;dy<M7R;dy++)
      for (int i=0;i<NRX;i++)
	score[dy][i] = 0.0f;

    int d = (idx%NUM);
    int j = (idx/NUM);
    buffer2[j*NUM + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + d];
    __syncthreads();
    for (int dp=0;dp<NDIM/4;dp+=NUM) {
      float4 temp;
      if (dp<(NDIM/4-NUM))
	temp = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + dp + d + NUM];

      if (idx<M7W*M7H/M7R/NRX) {
	for (int d=0;d<NUM;d++) {
	  float4 v1[NRX];
#pragma unroll
	  for (int i=0;i<NRX;i++) 
	    v1[i] = buffer1[(((M7W/NRX)*i + ix)<<5) + ((dp + d + (M7W/NRX)*i + ix)&31)];
	  //v1[i] = buffer1[((M7W/NRX)*i + ix)*NDIM/4 + (dp + d + (M7W/NRX)*i + ix)%(NDIM/4)];
#pragma unroll
	  for (int dy=0;dy<M7R;dy++) {
	    float4 v2 = buffer2[(M7R*iy + dy)*NUM + d];    
#pragma unroll
	    for (int i=0;i<NRX;i++) {
	      score[dy][i] += v1[i].x*v2.x;
	      score[dy][i] += v1[i].y*v2.y;
	      score[dy][i] += v1[i].z*v2.z;
	      score[dy][i] += v1[i].w*v2.w;
	    }
	  }
	}
      }
      __syncthreads();

      if (dp<(NDIM/4-NUM)) {
	buffer2[j*NUM + d] = temp;
	__syncthreads();
      }
    }
    for (int dy=0;dy<M7R;dy++) {
      for (int i=0;i<NRX;i++) {
	if (score[dy][i]>max_score[i]) {
	  max_score[i] = score[dy][i];     
	  index[i] = bp2 + M7R*iy + dy;
	}
      }
    }
    __syncthreads();
  }

  float *scores = (float*)buffer1;
  int *indices = (int*)&scores[M7W*M7H/M7R];
  if (idx<M7W*M7H/M7R/NRX) {
    for (int i=0;i<NRX;i++) {
      scores[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];  
      indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
    }
  }
  __syncthreads();
  
  if (ty==0) {
    float max_score = scores[tx];
    int index = indices[tx];
    for (int y=0;y<M7H/M7R;y++)
      if (scores[y*M7W + tx]>max_score) {
	max_score = scores[y*M7W + tx]; 
	index = indices[y*M7W + tx];    
      }
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}

__global__ void Match8blocked2(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
#define NRX 2
#define NUM (NRX*M7R)                       // 32*8 threads
  __shared__ float4 buffer1[M7W*NDIM/4];    // 32*32
  __shared__ float4 buffer2[M7H*NUM];       // 32*8
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bp1 = M7W*blockIdx.x;
  for (int d=tx;d<NDIM/4;d+=M7W)
    for (int j=ty;j<M7W;j+=M7H/M7R)     
      buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];

  float max_score[NRX];
  int index[NRX];
  for (int i=0;i<NRX;i++) {
    max_score[i] = 0.0f;
    index[i] = -1;
  }
  int idx = ty*M7W + tx;
  int ix = idx%(M7W/NRX);
  int iy = idx/(M7W/NRX);
  for (int bp2=0;bp2<NPTS;bp2+=M7H) {
    float score[M7R][NRX];                                    
    for (int dy=0;dy<M7R;dy++)
      for (int i=0;i<NRX;i++)
	score[dy][i] = 0.0f;
    for (int dp=0;dp<NDIM/4;dp+=NUM) {
      int d = (idx%NUM);
      int j = (idx/NUM);
      buffer2[j*NUM + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + dp + d];
      __syncthreads();

      if (idx<M7W*M7H/M7R/NRX) {
	for (int d=0;d<NUM;d++) {
	  float4 v1[NRX];
	  for (int i=0;i<NRX;i++) 
	    v1[i] = buffer1[((M7W/NRX)*i + ix)*NDIM/4 + (dp + d + (M7W/NRX)*i + ix)%(NDIM/4)];
	  for (int dy=0;dy<M7R;dy++) {
	    float4 v2 = buffer2[(M7R*iy + dy)*NUM + d];    
	    for (int i=0;i<NRX;i++) {
	      score[dy][i] += v1[i].x*v2.x;
	      score[dy][i] += v1[i].y*v2.y;
	      score[dy][i] += v1[i].z*v2.z;
	      score[dy][i] += v1[i].w*v2.w;
	    }
	  }
	}
      }
      __syncthreads();
    }
    for (int dy=0;dy<M7R;dy++) {
      for (int i=0;i<NRX;i++) {
	if (score[dy][i]>max_score[i]) {
	  max_score[i] = score[dy][i];     
	  index[i] = bp2 + M7R*iy + dy;
	}
      }
    }
    __syncthreads();
  }

  float *scores = (float*)buffer1;
  int *indices = (int*)&scores[M7W*M7H/M7R];
  if (idx<M7W*M7H/M7R/NRX) {
    for (int i=0;i<NRX;i++) {
      scores[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];  
      indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
    }
  }
  __syncthreads();
  
  if (ty==0) {
    float max_score = scores[tx];
    int index = indices[tx];
    for (int y=0;y<M7H/M7R;y++)
      if (scores[y*M7W + tx]>max_score) {
	max_score = scores[y*M7W + tx]; 
	index = indices[y*M7W + tx];    
      }
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}

__global__ void Match9(float *d_pts1, float *d_pts2, float *d_score, int *d_index)
{
#define NRX 2
#define NUM 8
  __shared__ float4 buffer1[M7W*NDIM/4]; 
  __shared__ float4 buffer2[M7H*NUM];       
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bp1 = M7W*blockIdx.x;
  for (int d=tx;d<NDIM/4;d+=M7W)
    for (int j=ty;j<M7W;j+=M7H/M7R)     
      buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];

  float max_score[NRX];
  int index[NRX];
  for (int i=0;i<NRX;i++) {
    max_score[i] = 0.0f;
    index[i] = -1;
  }
  int idx = ty*M7W + tx;
  int ix = idx%(M7W/NRX);
  int iy = idx/(M7W/NRX);
  for (int bp2=0;bp2<NPTS;bp2+=M7H) {
    
      float score[M7R][NRX];                                    
      if (idx<M7W*M7H/M7R/NRX) {    // 128
	for (int dy=0;dy<M7R;dy++)
	  for (int i=0;i<NRX;i++)
	    score[dy][i] = 0.0f;
      }

      for (int d=0;d<NDIM/4;d+=NUM) {
	if (idx<M7H*NUM)            // 256
	  buffer2[idx] = ((float4*)d_pts2)[(bp2 + (idx/NUM))*(NDIM/4) + d + (idx%NUM)];
	__syncthreads();

	if (idx<M7W*M7H/M7R/NRX) {  // 128
	  for (int j=0;j<NUM;j++) {
	    float4 v1[NRX];
	    for (int i=0;i<NRX;i++) 
	      v1[i] = buffer1[((M7W/NRX)*i + ix)*NDIM/4 + (d + j + (M7W/NRX)*i + ix)%(NDIM/4)];
	    for (int dy=0;dy<M7R;dy++) {
	      float4 v2 = buffer2[(M7R*ty + dy)*NUM + j];
	      for (int i=0;i<NRX;i++) {
		score[dy][i] += v1[i].x*v2.x;
		score[dy][i] += v1[i].y*v2.y;
		score[dy][i] += v1[i].z*v2.z;
		score[dy][i] += v1[i].w*v2.w;
	      }	      
	    }
	  }
	}
	__syncthreads();
      }
      
      if (idx<M7W*M7H/M7R/NRX) {  // 128
	for (int dy=0;dy<M7R;dy++) {
	  for (int i=0;i<NRX;i++) {
	    if (score[dy][i]>max_score[i]) {
	      max_score[i] = score[dy][i];     
	      index[i] = bp2 + M7R*iy + dy;
	    }
	  }
	}
      }
      __syncthreads();
  }

  float *scores = (float*)buffer1;
  int *indices = (int*)&scores[M7W*M7H/M7R];
  if (idx<M7W*M7H/M7R/NRX) {
    for (int i=0;i<NRX;i++) {
      scores[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];  
      indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
    }
  }
  __syncthreads();
  
  if (ty==0) {
    float max_score = scores[tx];
    int index = indices[tx];
    for (int y=0;y<M7H/M7R;y++)
      if (scores[y*M7W + tx]>max_score) {
	max_score = scores[y*M7W + tx]; 
	index = indices[y*M7W + tx];    
      }
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}


int main(int argc, char *argv[])
{
  safeCall(cudaSetDevice(0));

  size_t space = sizeof(float)*NPTS*NDIM*2 + 8;
  std::vector<float> data(NPTS*NDIM*2 + 8);
  void *ptr = (void*)&data[0];
  float *h_pts1 = (float*)std::align(32, sizeof(float)*NPTS*NDIM, ptr, space);
  ptr = (void*)&data[NPTS*NDIM];
  float *h_pts2 = (float*)std::align(32, sizeof(float)*NPTS*NDIM, ptr, space);
  std::vector<int> h_index(NPTS);
  std::vector<float> h_score(NPTS);
  std::vector<int> h_index2(NPTS);
  std::vector<float> h_score2(NPTS);
  
  float *d_pts1, *d_pts2, *d_score;
  int *d_index;
  std::cout << std::endl;
  int psize = sizeof(float)*NPTS;
  std::cout << "Data size:   " << 2.0*psize*NDIM/1024/1024 << " MB" << std::endl;
  TimerGPU time;
  float ltime = time.read();

  safeCall(cudaMalloc((void **)&d_pts1, psize*NDIM));
  safeCall(cudaMalloc((void **)&d_pts2, psize*NDIM));
  safeCall(cudaMalloc((void **)&d_index, psize));
  safeCall(cudaMalloc((void **)&d_score, psize));
  std::cout << "Allocate:    " << time.read() - ltime << " ms" << std::endl;

  for (int i=0;i<NPTS;i++) {
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int d=0;d<NDIM;d++) {
      sum1 += h_pts1[i*NDIM + d] = (float)rand()/RAND_MAX;
      sum2 += h_pts2[i*NDIM + d] = (float)rand()/RAND_MAX;
    }
    sum1 = sqrt(NDIM)/sum1;
    sum2 = sqrt(NDIM)/sum2;
    for (int d=0;d<NDIM;d++) {
      h_pts1[i*NDIM + d] *= sum1;
      h_pts2[i*NDIM + d] *= sum2;
    }
  }
  ltime = time.read();
  safeCall(cudaMemcpy(d_pts1, h_pts1, psize*NDIM, cudaMemcpyHostToDevice));
  safeCall(cudaMemcpy(d_pts2, h_pts2, psize*NDIM, cudaMemcpyHostToDevice));
  float delay = time.read() - ltime;
  std::cout << "Upload:      " << delay << " ms  " << 2*psize*NDIM/delay/1024/1024 << " MB/ms" << std::endl;

  if (RUNCPU) {
#if 0
    ltime = time.read();
    MatchC1(h_pts1, h_pts2, h_score.data(), h_index.data());
    delay = time.read() - ltime;
    std::cout << "MatchCPU1:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;

    ltime = time.read();
    MatchC2(h_pts1, h_pts2, h_score.data(), h_index.data());
    delay = time.read() - ltime;
    std::cout << "MatchCPU2:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
#endif

    ltime = time.read();
    MatchC3(h_pts1, h_pts2, h_score.data(), h_index.data());
    delay = time.read() - ltime;
    std::cout << "MatchCPU3:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  }
  dim3 blocks, threads;
#if 0
  blocks = dim3(NPTS/M1W);
  threads = dim3(M1W);
  ltime = time.read();
  Match1<<<blocks,threads>>>(d_pts1, d_pts2, d_score, d_index);
  delay = time.read() - ltime;
  checkMsg("Match1 error");
  std::cout << "MatchGPU1:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;

  blocks = dim3(NPTS/M2W);
  threads = dim3(M2W, M2H);
  ltime = time.read();
  Match2<<<blocks,threads>>>(d_pts1, d_pts2, d_score, d_index);
  delay = time.read() - ltime;
  checkMsg("Match2 error");
  std::cout << "MatchGPU2:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
#endif  

  blocks = dim3(NPTS/M2W);
  threads = dim3(M2W, M2H);
  ltime = time.read();
  Match3<<<blocks,threads>>>(d_pts1, d_pts2, d_score, d_index);
  delay = time.read() - ltime;
  checkMsg("Match3 error");
  std::cout << "MatchGPU3:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  
  blocks = dim3(NPTS/M2W);
  threads = dim3(M2W, M2H);
  ltime = time.read();
  Match4<<<blocks,threads>>>(d_pts1, d_pts2, d_score, d_index);
  delay = time.read() - ltime;
  checkMsg("Match4 error");
  std::cout << "MatchGPU4:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  
  blocks = dim3(NPTS/M5W);
  threads = dim3(M5W, M5H);
  ltime = time.read();
  Match5<<<blocks,threads>>>(d_pts1, d_pts2, d_score, d_index);
  delay = time.read() - ltime;
  checkMsg("Match5 error");
  std::cout << "MatchGPU5:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  
  blocks = dim3(NPTS/M5W);
  threads = dim3(M5W, M5H);
  ltime = time.read();
  Match6<<<blocks,threads>>>(d_pts1, d_pts2, d_score, d_index);
  delay = time.read() - ltime;
  checkMsg("Match6 error");
  std::cout << "MatchGPU6:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;

  blocks = dim3(NPTS/M7W);
  threads = dim3(M7W, M7H/M7R);
  ltime = time.read();
  Match7<<<blocks,threads>>>(d_pts1, d_pts2, d_score, d_index);
  delay = time.read() - ltime;
  checkMsg("Match7 error");
  std::cout << "MatchGPU7:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;

  blocks = dim3(NPTS/M7W);
  threads = dim3(M7W, M7H/M7R);
  ltime = time.read();
  Match8<<<blocks,threads>>>(d_pts1, d_pts2, d_score, d_index);
  delay = time.read() - ltime;
  checkMsg("Match8 error");
  std::cout << "MatchGPU8:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  #if 1
  blocks = dim3(NPTS/M7W);
  threads = dim3(M7W, M7H/M7R/2);
  ltime = time.read();
  Match8small<<<blocks,threads>>>(d_pts1, d_pts2, d_score, d_index);
  delay = time.read() - ltime;
  checkMsg("Match8small error");
  std::cout << "Match8small:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  #endif
  #if 1
  blocks = dim3(NPTS/M7W);
  threads = dim3(M7W, M7H/M7R);
  ltime = time.read();
  Match8blocked<<<blocks,threads>>>(d_pts1, d_pts2, d_score, d_index);
  delay = time.read() - ltime;
  checkMsg("Match8blocked error");
  std::cout << "MatchGPU8blocked:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  #endif
  ltime = time.read();
  safeCall(cudaMemcpy(h_index2.data(), d_index, psize, cudaMemcpyDeviceToHost));
  safeCall(cudaMemcpy(h_score2.data(), d_score, psize, cudaMemcpyDeviceToHost));
  delay = time.read() - ltime;
  std::cout << "Download:    " << delay << " ms  " << 2*psize/delay/1024/1024 << " MB/ms" << std::endl;
  ltime = time.read();

  if (CHECK)
    CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

  std::cout << std::endl;
  safeCall(cudaFree(d_pts1));
  safeCall(cudaFree(d_pts2));
  safeCall(cudaFree(d_index));
  safeCall(cudaFree(d_score));
}
