//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <cudautils.h>
#include "cudaSiftD_old.h"
#include "cudaSift.h"

///////////////////////////////////////////////////////////////////////////////
// Multiply an image to a constant and add another constant
///////////////////////////////////////////////////////////////////////////////
__global__ void MultiplyAdd(float *d_Result, float *d_Data, int width, int pitch, int height)
{
  const int x = blockIdx.x*16 + threadIdx.x;
  const int y = blockIdx.y*16 + threadIdx.y;
  int p = y*pitch + x;
  if (x<width && y<height)
    d_Result[p] = d_ConstantA[0]*d_Data[p] + d_ConstantB[0];
  __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
// Find minimum and maximum value of an image
///////////////////////////////////////////////////////////////////////////////
__global__ void FindMinMax(float *d_MinMax, float *d_Data, int width, int pitch, int height)
{
  __shared__ float minvals[128];
  __shared__ float maxvals[128];
  const int tx = threadIdx.x;
  const int x = blockIdx.x*128 + tx;
  const int y = blockIdx.y*16;
  const int b = blockDim.x;
  int p = y*pitch + x;
  if (x<width) {
    float val = d_Data[p];
    minvals[tx] = val;
    maxvals[tx] = val;
  } else {
    float val = d_Data[p-x];
    minvals[tx] = val;
    maxvals[tx] = val;
  }
  for (int ty=1;ty<16;ty++) {
    p += pitch;
    if (tx<width) {
      float val = d_Data[p];
      if (val<minvals[tx])
	minvals[tx] = val;
      if (val>maxvals[tx])
	maxvals[tx] = val;
    }
  }
  __syncthreads();
  int mod = 1;
  for (int d=1;d<b;d<<=1) {
    if ((tx&mod)==0) {
      if (minvals[tx+d]<minvals[tx+0])
	minvals[tx+0] = minvals[tx+d];
      if (maxvals[tx+d]>maxvals[tx+0])
	maxvals[tx+0] = maxvals[tx+d];
    }
    mod = 2*mod + 1;
    __syncthreads();
  }
  if (tx==0) {
    int ptr = 2*(gridDim.x*blockIdx.y + blockIdx.x);
    d_MinMax[ptr+0] = minvals[0];
    d_MinMax[ptr+1] = maxvals[0];
  }
  __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
// Find maximum in xpos, ypos and scale
///////////////////////////////////////////////////////////////////////////////

__global__ void Find3DMinMax(int *d_Result, float *d_Data1, float *d_Data2, float *d_Data3, int width, int pitch, int height)
{
  // Data cache
  __shared__ float data1[3*(MINMAX_W + 2)];
  __shared__ float data2[3*(MINMAX_W + 2)];
  __shared__ float data3[3*(MINMAX_W + 2)];
  __shared__ float ymin1[(MINMAX_W + 2)];
  __shared__ float ymin2[(MINMAX_W + 2)];
  __shared__ float ymin3[(MINMAX_W + 2)];
  __shared__ float ymax1[(MINMAX_W + 2)];
  __shared__ float ymax2[(MINMAX_W + 2)];
  __shared__ float ymax3[(MINMAX_W + 2)];

  // Current tile and apron limits, relative to row start
  const int tx = threadIdx.x;
  const int xStart = blockIdx.x*MINMAX_W;
  const int xEnd = xStart + MINMAX_W - 1;
  const int xReadPos = xStart + tx - WARP_SIZE;
  const int xWritePos = xStart + tx;
  const int xEndClamped = min(xEnd, width - 1);
  int memWid = MINMAX_W + 2;

  int memPos0 = (tx - WARP_SIZE + 1);
  int memPos1 = (tx - WARP_SIZE + 1);
  int yq = 0;
  unsigned int output = 0;
  for (int y=0;y<32+2;y++) {

    output >>= 1;
    int memPos =  yq*memWid + (tx - WARP_SIZE + 1);
    int yp = 32*blockIdx.y + y - 1;
    yp = max(yp, 0);
    yp = min(yp, height-1);
    int readStart = yp*pitch;

    // Set the entire data cache contents
    if (tx>=(WARP_SIZE-1)) {
      if (xReadPos<0) {
	data1[memPos] = 0;
	data2[memPos] = 0;
	data3[memPos] = 0;
      } else if (xReadPos>=width) {
	data1[memPos] = 0;
	data2[memPos] = 0;
	data3[memPos] = 0;
      } else {
	data1[memPos] = d_Data1[readStart + xReadPos];
	data2[memPos] = d_Data2[readStart + xReadPos];
	data3[memPos] = d_Data3[readStart + xReadPos];
      }
    }
    __syncthreads();
  
    int memPos2 = yq*memWid + tx;
    if (y>1) {
      if (tx<memWid) {
	float min1 = fminf(fminf(data1[memPos0], data1[memPos1]), data1[memPos2]);
	float min2 = fminf(fminf(data2[memPos0], data2[memPos1]), data2[memPos2]);
	float min3 = fminf(fminf(data3[memPos0], data3[memPos1]), data3[memPos2]);
	float max1 = fmaxf(fmaxf(data1[memPos0], data1[memPos1]), data1[memPos2]);
	float max2 = fmaxf(fmaxf(data2[memPos0], data2[memPos1]), data2[memPos2]);
	float max3 = fmaxf(fmaxf(data3[memPos0], data3[memPos1]), data3[memPos2]);
	ymin1[tx] = min1;
	ymin2[tx] = fminf(fminf(min1, min2), min3);
	ymin3[tx] = min3;
	ymax1[tx] = max1;
	ymax2[tx] = fmaxf(fmaxf(max1, max2), max3);
	ymax3[tx] = max3;
      }
    }
    __syncthreads();

    if (y>1) {
      if (tx<MINMAX_W) {
	if (xWritePos<=xEndClamped) {
	  float minv = fminf(fminf(fminf(fminf(fminf(ymin2[tx], ymin2[tx+2]), ymin1[tx+1]), ymin3[tx+1]), data2[memPos0+1]), data2[memPos2+1]);
	  minv = fminf(minv, d_Threshold[1]);
	  float maxv = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(ymax2[tx], ymax2[tx+2]), ymax1[tx+1]), ymax3[tx+1]), data2[memPos0+1]), data2[memPos2+1]);
	  maxv = fmaxf(maxv, d_Threshold[0]);
	  if (data2[memPos1+1]<minv || data2[memPos1+1]>maxv)
	    output |= 0x80000000;
	}
      }
    }
    __syncthreads();

    memPos0 = memPos1;
    memPos1 = memPos2;
    yq = (yq<2 ? yq+1 : 0);
  }
  if (tx<MINMAX_W && xWritePos<width) {
    int writeStart = blockIdx.y*pitch + xWritePos;
    d_Result[writeStart] = output;
  }
}

__global__ void UnpackPointers(int *minmax, int *ptrs, int w, int h, 
  int maxPts)
{
  const int tx = threadIdx.x;
  int numPts = 0;
  for (int y=0;y<h/32;y++) {
    for (int x=0;x<w;x+=16) {
      unsigned int val = minmax[y*w+x+tx];
      if (val) {
	for (int k=0;k<32;k++) {
	  if (val&0x1 && numPts<maxPts) {
	    ptrs[16*numPts+tx] = (y*32+k)*w + x+tx;
	    numPts++;
	  }
	  val >>= 1;
	}
      }
    }
  } 
}

///////////////////////////////////////////////////////////////////////////////
// Compute precise positions in xpos, ypos and scale
///////////////////////////////////////////////////////////////////////////////

#define POSBLK_SIZE    32

__global__ void ComputePositions(float *g_Data1, float *g_Data2, float *g_Data3, int *d_Ptrs, float *d_Sift, int numPts, int maxPts, int w, int h)
{
  int i = blockIdx.x*POSBLK_SIZE + threadIdx.x;
  if (i>=numPts) 
    return;
  int pos = d_Ptrs[i];
  float val[7];
  val[0] = g_Data2[pos];
  val[1] = g_Data2[pos-1];
  val[2] = g_Data2[pos+1];
  float dx = 0.5f*(val[2] - val[1]);
  float dxx = 2.0f*val[0] - val[1] - val[2];
  val[3] = g_Data2[pos-w];
  val[4] = g_Data2[pos+w];
  float dy = 0.5f*(val[4] - val[3]); 
  float dyy = 2.0f*val[0] - val[3] - val[4];
  val[5] = g_Data3[pos];
  val[6] = g_Data1[pos];
  float ds = 0.5f*(val[6] - val[5]); 
  float dss = 2.0f*val[0] - val[5] - val[6];
  float dxy = 0.25f*(g_Data2[pos+w+1] + g_Data2[pos-w-1] - g_Data2[pos-w+1] - g_Data2[pos+w-1]);
  float dxs = 0.25f*(g_Data3[pos+1] + g_Data1[pos-1] - g_Data1[pos+1] - g_Data3[pos-1]);
  float dys = 0.25f*(g_Data3[pos+w] + g_Data1[pos-w] - g_Data3[pos-w] - g_Data1[pos+w]);
  float idxx = dyy*dss - dys*dys;
  float idxy = dys*dxs - dxy*dss;  
  float idxs = dxy*dys - dyy*dxs;
  float idyy = dxx*dss - dxs*dxs;
  float idys = dxy*dxs - dxx*dys;
  float idss = dxx*dyy - dxy*dxy;
  float det = idxx*dxx + idxy*dxy + idxs*dxs;
  float idet = 1.0f / det;
  float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
  float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
  float pds = idet*(idxs*dx + idys*dy + idss*ds);
  if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f){
    pdx = __fdividef(dx, dxx);
    pdy = __fdividef(dy, dyy);
    pds = __fdividef(ds, dss);
  }
  float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
  d_Sift[i+0*maxPts] = (pos%w) + pdx;
  d_Sift[i+1*maxPts] = (pos/w) + pdy;
  d_Sift[i+2*maxPts] = d_Scale * exp2f(pds*d_Factor);
  d_Sift[i+3*maxPts] = val[0] + dval;
  float tra = dxx + dyy;
  det = dxx*dyy - dxy*dxy;
  d_Sift[i+4*maxPts] = __fdividef(tra*tra, det);
}

///////////////////////////////////////////////////////////////////////////////
// Compute two dominating orientations in xpos and ypos
///////////////////////////////////////////////////////////////////////////////
__global__ void ComputeOrientations(float *g_Data, int *d_Ptrs, float *d_Orient, int maxPts, int w, int h)
{
  __shared__ float data[16*15];
  __shared__ float hist[32*13];
  __shared__ float gauss[16];
  const int tx = threadIdx.x;
  const int bx = blockIdx.x;
  for (int i=0;i<13;i++)
    hist[i*32+tx] = 0.0f;
  __syncthreads();
  float i2sigma2 = -1.0f/(2.0f*3.0f*3.0f);
  if (tx<15) 
    gauss[tx] = exp(i2sigma2*(tx-7)*(tx-7));
  int p = d_Ptrs[bx];
  int yp = p/w - 7;
  int xp = p%w - 7;
  int px = xp & 15;
  int x = tx - px;

  for (int y=0;y<15;y++) {
    int memPos = 16*y + x;
    int xi = xp + x;
    int yi = yp + y;
    if (xi<0) xi = 0;
    if (xi>=w) xi = w-1;
    if (yi<0) yi = 0;
    if (yi>=h) yi = h-1;
    if (x>=0 && x<15) 
      data[memPos] = g_Data[yi*w+xi];
  }
  __syncthreads();
  for (int y=1;y<14;y++) {
    int memPos = 16*y + x;
    if (x>=1 && x<14) {
      float dy = data[memPos+16] - data[memPos-16];
      float dx = data[memPos+1]  - data[memPos-1];
      int bin = 16.0f*atan2f(dy, dx)/3.1416f + 16.5f;
      if (bin==32)
	bin = 0;
      float grad = sqrtf(dx*dx + dy*dy);
      hist[32*(x-1)+bin] += grad*gauss[x]*gauss[y];
    }
  }
  __syncthreads();
  for (int y=0;y<5;y++)
    hist[y*32+tx] += hist[(y+8)*32+tx];
  __syncthreads();
  for (int y=0;y<4;y++)
    hist[y*32+tx] += hist[(y+4)*32+tx];
  __syncthreads();
  for (int y=0;y<2;y++)
    hist[y*32+tx] += hist[(y+2)*32+tx];
  __syncthreads();
  hist[tx] += hist[32+tx];
  __syncthreads();
  if (tx==0) 
    hist[32] = 6*hist[0] + 4*(hist[1]+hist[31]) + (hist[2]+hist[30]);
  if (tx==1)
    hist[33] = 6*hist[1] + 4*(hist[2]+hist[0]) + (hist[3]+hist[31]);
  if (tx>=2 && tx<=29)
    hist[tx+32] = 6*hist[tx] + 4*(hist[tx+1]+hist[tx-1]) + 
      (hist[tx+2]+hist[tx-2]);
  if (tx==30)
    hist[62] = 6*hist[30] + 4*(hist[31]+hist[29]) + (hist[0]+hist[28]);
  if (tx==31)
    hist[63] = 6*hist[31] + 4*(hist[0]+hist[30]) + (hist[1]+hist[29]);
  __syncthreads();
  float v = hist[32+tx];
  hist[tx] = (v>hist[32+((tx+1)&31)] && v>=hist[32+((tx+31)&31)] ? v : 0.0f);
  __syncthreads();
  if (tx==0) {
    float maxval1 = 0.0;
    float maxval2 = 0.0;
    int i1 = -1;
    int i2 = -1;
    for (int i=0;i<32;i++) {
      float v = hist[i];
      if (v>maxval1) {
	maxval2 = maxval1;
	maxval1 = v;
	i2 = i1;
	i1 = i;
      } else if (v>maxval2) {
	maxval2 = v;
	i2 = i;
      }
    }
    float val1 = hist[32+((i1+1)&31)];
    float val2 = hist[32+((i1+31)&31)];
    float peak = i1 + 0.5f*(val1-val2) / (2.0f*maxval1-val1-val2);
    d_Orient[bx] = 11.25f*(peak<0.0f ? peak+32.0f : peak);
    if (maxval2<0.8f*maxval1) 
      i2 = -1;
    if (i2>=0) {
      float val1 = hist[32+((i2+1)&31)];
      float val2 = hist[32+((i2+31)&31)];
      float peak = i2 + 0.5f*(val1-val2) / (2.0f*maxval2-val1-val2);
      d_Orient[bx+maxPts] = 11.25f*(peak<0.0f ? peak+32.0f : peak);;
    } else 
      d_Orient[bx+maxPts] = i2;
  }
} 

