//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <cudautils.h>
#include "cudaSiftD.h"
#include "cudaSift.h"

///////////////////////////////////////////////////////////////////////////////
// Kernel configuration
///////////////////////////////////////////////////////////////////////////////

__device__ __constant__ float d_ConstantA[8]; 
__device__ __constant__ float d_ConstantB[8];

texture<float, 2, cudaReadModeElementType> tex;

///////////////////////////////////////////////////////////////////////////////
// Lowpass filter an image
///////////////////////////////////////////////////////////////////////////////
__global__ void LowPass5(float *d_Result, float *d_Data, int width, int height)
{
  __shared__ float inrow[LOWPASS5_DX+4];
  __shared__ float brow[5*LOWPASS5_DX];
  __shared__ int yRead[LOWPASS5_DY+4], yWrite[LOWPASS5_DY+4];
  const int tx = threadIdx.x;
  const int tx0 = tx+0*LOWPASS5_DX;
  const int tx1 = tx+1*LOWPASS5_DX;
  const int tx2 = tx+2*LOWPASS5_DX;
  const int tx3 = tx+3*LOWPASS5_DX;
  const int tx4 = tx+4*LOWPASS5_DX;
  const int xStart = __mul24(blockIdx.x, LOWPASS5_DX);
  const int yStart = __mul24(blockIdx.y, LOWPASS5_DY);
  const int xWrite = xStart + tx;
  const float *k = d_Kernel;
  if (tx<LOWPASS5_DY+4) {
    int y = yStart + tx - 1;
    y = (y<0 ? 0 : y);
    y = (y>=height ? height-1 : y);
    yRead[tx] = __mul24(y, width);
    yWrite[tx] = __mul24(yStart + tx - 4, width);
  }
  __syncthreads();
  int xRead = xStart + tx - WARP_SIZE;
  xRead = (xRead<0 ? 0 : xRead);
  xRead = (xRead>=width ? width-1 : xRead);
  for (int dy=0;dy<LOWPASS5_DY+4;dy+=5) {
    if (tx>=WARP_SIZE-2)
      inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+0] + xRead];
    __syncthreads();
    if (tx<LOWPASS5_DX) 
      brow[tx0] = __fmul_rz(k[0],(inrow[tx]+inrow[tx+4])) + 
	__fmul_rz(k[1],(inrow[tx+1]+inrow[tx+3])) + 
	__fmul_rz(k[2],inrow[tx+2]);
    __syncthreads();
    if (tx<LOWPASS5_DX && dy>=4) {
      d_Result[yWrite[dy+0] + xWrite] = __fmul_rz(k[2],brow[tx2]) +
	__fmul_rz(k[0],brow[tx0]+brow[tx4]) + 
	__fmul_rz(k[1],brow[tx1]+brow[tx3]);
    }
    if (dy<(LOWPASS5_DY+3)) {
      if (tx>=WARP_SIZE-2)
	inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+1] + xRead];
      __syncthreads();
      if (tx<LOWPASS5_DX)
	brow[tx1] = __fmul_rz(k[0],(inrow[tx]+inrow[tx+4])) + 
	  __fmul_rz(k[1],(inrow[tx+1]+inrow[tx+3])) + 
	  __fmul_rz(k[2],inrow[tx+2]);
      __syncthreads();
      if (tx<LOWPASS5_DX && dy>=3) {
	d_Result[yWrite[dy+1] + xWrite] = __fmul_rz(k[2],brow[tx3]) + 
	  __fmul_rz(k[0],brow[tx1]+brow[tx0]) + 
	  __fmul_rz(k[1],brow[tx2]+brow[tx4]); 
      }
    }
    if (dy<(LOWPASS5_DY+2)) {
      if (tx>=WARP_SIZE-2)
	inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+2] + xRead];
      __syncthreads();
      if (tx<LOWPASS5_DX)
	brow[tx2] = __fmul_rz(k[0],(inrow[tx]+inrow[tx+4])) + 
	  __fmul_rz(k[1],(inrow[tx+1]+inrow[tx+3])) + 
	  __fmul_rz(k[2],inrow[tx+2]);
      __syncthreads();
      if (tx<LOWPASS5_DX && dy>=2) {
	d_Result[yWrite[dy+2] + xWrite] = __fmul_rz(k[2],brow[tx4]) + 
	  __fmul_rz(k[0],brow[tx2]+brow[tx1]) + 
	  __fmul_rz(k[1],brow[tx3]+brow[tx0]); 
      }
    }
    if (dy<(LOWPASS5_DY+1)) {
      if (tx>=WARP_SIZE-2)
	inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+3] + xRead];
      __syncthreads();
      if (tx<LOWPASS5_DX)
	brow[tx3] = __fmul_rz(k[0],(inrow[tx]+inrow[tx+4])) + 
	  __fmul_rz(k[1],(inrow[tx+1]+inrow[tx+3])) + 
	  __fmul_rz(k[2],inrow[tx+2]);
      __syncthreads();
      if (tx<LOWPASS5_DX && dy>=1) {
	d_Result[yWrite[dy+3] + xWrite] = __fmul_rz(k[2],brow[tx0]) + 
	  __fmul_rz(k[0],brow[tx3]+brow[tx2]) + 
	  __fmul_rz(k[1],brow[tx4]+brow[tx1]); 
      }
    }
    if (dy<LOWPASS5_DY) {
      if (tx>=WARP_SIZE-2)
	inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+4] + xRead];
      __syncthreads();
      if (tx<LOWPASS5_DX)
	brow[tx4] = __fmul_rz(k[0],(inrow[tx]+inrow[tx+4])) + 
	  __fmul_rz(k[1],(inrow[tx+1]+inrow[tx+3])) + 
	  __fmul_rz(k[2],inrow[tx+2]);
      __syncthreads();
      if (tx<LOWPASS5_DX) {
	d_Result[yWrite[dy+4] + xWrite] = __fmul_rz(k[2],brow[tx1]) + 
	  __fmul_rz(k[0],brow[tx4]+brow[tx3]) + 
	  __fmul_rz(k[1],brow[tx0]+brow[tx2]); 
      }
    }
    __syncthreads();
  }
}

///////////////////////////////////////////////////////////////////////////////
// Lowpass filter an subsample image
///////////////////////////////////////////////////////////////////////////////
__global__ void ScaleDown(float *d_Result, float *d_Data, int width, int height)
{
  __shared__ float inrow[LOWPASS5_DX+4]; 
  __shared__ float brow[5*(LOWPASS5_DX/2)];
  __shared__ int yRead[LOWPASS5_DY+4], yWrite[LOWPASS5_DY+4];
  #define dx2 (LOWPASS5_DX/2)
  const int tx = threadIdx.x;
  const int tx0 = tx+0*dx2;
  const int tx1 = tx+1*dx2;
  const int tx2 = tx+2*dx2;
  const int tx3 = tx+3*dx2;
  const int tx4 = tx+4*dx2;
  const int xStart = __mul24(blockIdx.x, LOWPASS5_DX);
  const int yStart = __mul24(blockIdx.y, LOWPASS5_DY);
  const int xWrite = xStart/2 + tx;
  const float *k = d_Kernel;
  if (tx<LOWPASS5_DY+4) {
    int y = yStart + tx - 1;
    y = (y<0 ? 0 : y);
    y = (y>=height ? height-1 : y);
    yRead[tx] = __mul24(y, width);
    yWrite[tx] = __mul24((yStart + tx - 4)/2, (width/2));
  }
  __syncthreads();
  int xRead = xStart + tx - WARP_SIZE;
  xRead = (xRead<0 ? 0 : xRead);
  xRead = (xRead>=width ? width-1 : xRead);
  for (int dy=0;dy<LOWPASS5_DY+4;dy+=5) {
    if (tx>=WARP_SIZE-2) {
      //if ((yRead[dy+0] + xRead)>=(width*height))
      //printf("ScaleDown read error\n");
      inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+0] + xRead];
    }
    __syncthreads();
    if (tx<dx2) 
      brow[tx0] = __fmul_rz(k[0],(inrow[2*tx]+inrow[2*tx+4])) + 
	__fmul_rz(k[1],(inrow[2*tx+1]+inrow[2*tx+3])) + 
	__fmul_rz(k[2],inrow[2*tx+2]);
    __syncthreads();
    if (tx<dx2 && dy>=4 && !(dy&1)) {
      //if ((yWrite[dy+0] + xWrite)>=((width/2)*(height/2)))
      //printf("ScaleDown write error\n");
      d_Result[yWrite[dy+0] + xWrite] = __fmul_rz(k[2],brow[tx2]) +
	__fmul_rz(k[0],brow[tx0]+brow[tx4]) + 
	__fmul_rz(k[1],brow[tx1]+brow[tx3]);
    }
    if (dy<(LOWPASS5_DY+3)) {
      if (tx>=WARP_SIZE-2) {
	//if ((yRead[dy+1] + xRead)>=(width*height))
	//  printf("ScaleDown read error\n");
	inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+1] + xRead];
      }
      __syncthreads();
      if (tx<dx2)
	brow[tx1] = __fmul_rz(k[0],(inrow[2*tx]+inrow[2*tx+4])) + 
	  __fmul_rz(k[1],(inrow[2*tx+1]+inrow[2*tx+3])) + 
	  __fmul_rz(k[2],inrow[2*tx+2]);
      __syncthreads();
      if (tx<dx2 && dy>=3 && (dy&1)) {
	//if ((yWrite[dy+1] + xWrite)>=((width/2)*(height/2)))
	//  printf("ScaleDown write error\n");
	d_Result[yWrite[dy+1] + xWrite] = __fmul_rz(k[2],brow[tx3]) + 
	  __fmul_rz(k[0],brow[tx1]+brow[tx0]) + 
	  __fmul_rz(k[1],brow[tx2]+brow[tx4]); 
      }
    }
    if (dy<(LOWPASS5_DY+2)) {
      if (tx>=WARP_SIZE-2) {
	//if ((yRead[dy+2] + xRead)>=(width*height))
	//  printf("ScaleDown read error\n");
	inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+2] + xRead];
      }
      __syncthreads();
      if (tx<dx2)
	brow[tx2] = __fmul_rz(k[0],(inrow[2*tx]+inrow[2*tx+4])) + 
	  __fmul_rz(k[1],(inrow[2*tx+1]+inrow[2*tx+3])) + 
	  __fmul_rz(k[2],inrow[2*tx+2]);
      __syncthreads();
      if (tx<dx2 && dy>=2 && !(dy&1)) {
	//if ((yWrite[dy+2] + xWrite)>=((width/2)*(height/2)))
	//  printf("ScaleDown write error\n");
	d_Result[yWrite[dy+2] + xWrite] = __fmul_rz(k[2],brow[tx4]) + 
	  __fmul_rz(k[0],brow[tx2]+brow[tx1]) + 
	  __fmul_rz(k[1],brow[tx3]+brow[tx0]); 
      }
    }
    if (dy<(LOWPASS5_DY+1)) {
      if (tx>=WARP_SIZE-2) {
	//if ((yRead[dy+3] + xRead)>=(width*height))
	//  printf("ScaleDown read error\n");
	inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+3] + xRead];
      }
      __syncthreads();
      if (tx<dx2)
	brow[tx3] = __fmul_rz(k[0],(inrow[2*tx]+inrow[2*tx+4])) + 
	  __fmul_rz(k[1],(inrow[2*tx+1]+inrow[2*tx+3])) + 
	  __fmul_rz(k[2],inrow[2*tx+2]);
      __syncthreads();
      if (tx<dx2 && dy>=1 && (dy&1)) {
	//if ((yWrite[dy+3] + xWrite)>=((width/2)*(height/2)))
	//  printf("ScaleDown write error\n");
	d_Result[yWrite[dy+3] + xWrite] = __fmul_rz(k[2],brow[tx0]) + 
	  __fmul_rz(k[0],brow[tx3]+brow[tx2]) + 
	  __fmul_rz(k[1],brow[tx4]+brow[tx1]); 
      }
    }
    if (dy<LOWPASS5_DY) {
      if (tx>=WARP_SIZE-2) {
	//if ((yRead[dy+4] + xRead)>=(width*height))
	//  printf("ScaleDown read error\n");
	inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+4] + xRead];
      }
      __syncthreads();
      if (tx<dx2)
	brow[tx4] = __fmul_rz(k[0],(inrow[2*tx]+inrow[2*tx+4])) + 
	  __fmul_rz(k[1],(inrow[2*tx+1]+inrow[2*tx+3])) + 
	  __fmul_rz(k[2],inrow[2*tx+2]);
      __syncthreads();
      if (tx<dx2 && !(dy&1)) {
	//if ((yWrite[dy+4] + xWrite)>=((width/2)*(height/2)))
	//  printf("ScaleDown write error\n");
	d_Result[yWrite[dy+4] + xWrite] = __fmul_rz(k[2],brow[tx1]) + 
	  __fmul_rz(k[0],brow[tx4]+brow[tx3]) + 
	  __fmul_rz(k[1],brow[tx0]+brow[tx2]); 
      }
    }
    __syncthreads();
  }
}

///////////////////////////////////////////////////////////////////////////////
// Subtract two images
///////////////////////////////////////////////////////////////////////////////
__global__ void Subtract(float *d_Result, float *d_Data1, float *d_Data2,
  int width, int height)
{
  const int x = __mul24(blockIdx.x, 16) + threadIdx.x;
  const int y = __mul24(blockIdx.y, 16) + threadIdx.y;
  int p = __mul24(y, width) + x;
  if (x<width && y<height)
    d_Result[p] = d_Data1[p] - d_Data2[p];
  __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
// Multiply an image to a constant and add another constant
///////////////////////////////////////////////////////////////////////////////
__global__ void MultiplyAdd(float *d_Result, float *d_Data, 
  int width, int height)
{
  const int x = __mul24(blockIdx.x, 16) + threadIdx.x;
  const int y = __mul24(blockIdx.y, 16) + threadIdx.y;
  int p = __mul24(y, width) + x;
  if (x<width && y<height)
    d_Result[p] = d_ConstantA[0]*d_Data[p] + d_ConstantB[0];
  __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
// Find minimum and maximum value of an image
///////////////////////////////////////////////////////////////////////////////
__global__ void FindMinMax(float *d_MinMax, float *d_Data, int width, 
  int height)
{
  __shared__ float minvals[128];
  __shared__ float maxvals[128];
  const int tx = threadIdx.x;
  const int x = __mul24(blockIdx.x, 128) + tx;
  const int y = __mul24(blockIdx.y, 16);
  const int b = blockDim.x;
  int p = __mul24(y, width) + x;
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
    p += width;
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
    int ptr = 2*(__mul24(gridDim.x,blockIdx.y) + blockIdx.x);
    d_MinMax[ptr+0] = minvals[0];
    d_MinMax[ptr+1] = maxvals[0];
  }
  __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
// Find maximum in xpos, ypos and scale
///////////////////////////////////////////////////////////////////////////////

__global__ void Find3DMinMax(int *d_Result, float *d_Data1, float *d_Data2, 
  float *d_Data3, int width, int height)
{
  //Data cache
  __shared__ float data1[3*(MINMAX_SIZE + 2)];
  __shared__ float data2[3*(MINMAX_SIZE + 2)];
  __shared__ float data3[3*(MINMAX_SIZE + 2)];
  __shared__ float ymin1[(MINMAX_SIZE + 2)];
  __shared__ float ymin2[(MINMAX_SIZE + 2)];
  __shared__ float ymin3[(MINMAX_SIZE + 2)];
  __shared__ float ymax1[(MINMAX_SIZE + 2)];
  __shared__ float ymax2[(MINMAX_SIZE + 2)];
  __shared__ float ymax3[(MINMAX_SIZE + 2)];

  //Current tile and apron limits, relative to row start
  const int tx = threadIdx.x;
  const int xStart = __mul24(blockIdx.x, MINMAX_SIZE);
  const int xEnd = xStart + MINMAX_SIZE - 1;
  const int xReadPos = xStart + tx - WARP_SIZE;
  const int xWritePos = xStart + tx;
  const int xEndClamped = min(xEnd, width - 1);
  int memWid = MINMAX_SIZE + 2;

  int memPos0 = (tx - WARP_SIZE + 1);
  int memPos1 = (tx - WARP_SIZE + 1);
  int yq = 0;
  unsigned int output = 0;
  for (int y=0;y<34;y++) {

    output >>= 1;
    int memPos =  yq*memWid + (tx - WARP_SIZE + 1);
    int yp = 32*blockIdx.y + y - 1;
    yp = max(yp, 0);
    yp = min(yp, height-1);
    int readStart = __mul24(yp, width);

    //Set the entire data cache contents
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
	//if ((readStart + xReadPos)<0 || (readStart + xReadPos)>=width*height)
	//  printf("Find3DMinMax: read error\n");
      }
    }
    __syncthreads();
  
    int memPos2 = yq*memWid + tx;
    if (y>1) {
      if (tx<memWid) {
	float min1 = fminf(fminf(data1[memPos0], data1[memPos1]), 
			   data1[memPos2]);
	float min2 = fminf(fminf(data2[memPos0], data2[memPos1]), 
			   data2[memPos2]);
	float min3 = fminf(fminf(data3[memPos0], data3[memPos1]), 
			   data3[memPos2]);
	float max1 = fmaxf(fmaxf(data1[memPos0], data1[memPos1]), 
			   data1[memPos2]);
	float max2 = fmaxf(fmaxf(data2[memPos0], data2[memPos1]), 
			   data2[memPos2]);
	float max3 = fmaxf(fmaxf(data3[memPos0], data3[memPos1]), 
			   data3[memPos2]);
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
      if (tx<MINMAX_SIZE) {
	if (xWritePos<=xEndClamped) {
	  float minv = fminf(fminf(fminf(fminf(fminf(ymin2[tx], ymin2[tx+2]), 
	    ymin1[tx+1]), ymin3[tx+1]), data2[memPos0+1]), data2[memPos2+1]);
	  minv = fminf(minv, d_ConstantA[1]);
	  float maxv = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(ymax2[tx], ymax2[tx+2]), 
	    ymax1[tx+1]), ymax3[tx+1]), data2[memPos0+1]), data2[memPos2+1]);
	  maxv = fmaxf(maxv, d_ConstantA[0]);
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
  if (tx<MINMAX_SIZE && xWritePos<width) {
    int writeStart = __mul24(blockIdx.y, width) + xWritePos;
    d_Result[writeStart] = output;
    //if (writeStart<0 || writeStart>=width*iDivUp(height,32))
    //  printf("Find3DMinMax: write error\n");
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
__global__ void ComputePositions(float *g_Data1, float *g_Data2, 
  float *g_Data3, int *d_Ptrs, float *d_Sift, int numPts, int maxPts, 
  int w, int h)
{
  int i = __mul24(blockIdx.x, POSBLK_SIZE) + threadIdx.x;
  if (i>=numPts) 
    return;
  int p = d_Ptrs[i];
  //if (p<w+1 || p>=(w*h-w-1))
  //  printf("ComputePositions: read error\n");
  float val[7];
  val[0] = g_Data2[p];
  val[1] = g_Data2[p-1];
  val[2] = g_Data2[p+1];
  float dx = 0.5f*(val[2] - val[1]);
  float dxx = 2.0f*val[0] - val[1] - val[2];
  val[3] = g_Data2[p-w];
  val[4] = g_Data2[p+w];
  float dy = 0.5f*(val[4] - val[3]); 
  float dyy = 2.0f*val[0] - val[3] - val[4];
  val[5] = g_Data3[p];
  val[6] = g_Data1[p];
  float ds = 0.5f*(val[6] - val[5]); 
  float dss = 2.0f*val[0] - val[5] - val[6];
  float dxy = 0.25f*
    (g_Data2[p+w+1] + g_Data2[p-w-1] - g_Data2[p-w+1] - g_Data2[p+w-1]);
  float dxs = 0.25f*
    (g_Data3[p+1] + g_Data1[p-1] - g_Data1[p+1] - g_Data3[p-1]);
  float dys = 0.25f*
    (g_Data3[p+w] + g_Data1[p-w] - g_Data3[p-w] - g_Data1[p+w]);
  float idxx = dyy*dss - dys*dys;
  float idxy = dys*dxs - dxy*dss;  
  float idxs = dxy*dys - dyy*dxs;
  float idyy = dxx*dss - dxs*dxs;
  float idys = dxy*dxs - dxx*dys;
  float idss = dxx*dyy - dxy*dxy;
  float det = idxx*dxx + idxy*dxy + idxs*dxs;
  float idet = 1.0f / det;
  float pdx = idet*
    (idxx*dx + idxy*dy + idxs*ds);
  float pdy = idet*
    (idxy*dx + idyy*dy + idys*ds);
  float pds = idet*
    (idxs*dx + idys*dy + idss*ds);
  if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f){
    pdx = __fdividef(dx, dxx);
    pdy = __fdividef(dy, dyy);
    pds = __fdividef(ds, dss);
  }
  float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
  d_Sift[i+0*maxPts] = (p%w) + pdx;
  d_Sift[i+1*maxPts] = (p/w) + pdy;
  d_Sift[i+2*maxPts] = d_ConstantA[0] * exp2f(pds*d_ConstantB[0]);
  d_Sift[i+3*maxPts] = val[0] + dval;
  float tra = dxx + dyy;
  det = dxx*dyy - dxy*dxy;
  d_Sift[i+4*maxPts] = __fdividef(tra*tra, det);
}

///////////////////////////////////////////////////////////////////////////////
// Compute two dominating orientations in xpos and ypos
///////////////////////////////////////////////////////////////////////////////
__global__ void ComputeOrientations(float *g_Data, int *d_Ptrs, 
  float *d_Orient, int maxPts, int w, int h)
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

#define NUMDESCBUFS 4

///////////////////////////////////////////////////////////////////////////////
// Extract Sift descriptors
///////////////////////////////////////////////////////////////////////////////
__global__ void ExtractSiftDescriptors(float *g_Data, float *d_sift, 
  float *d_desc, int maxPts)
{
  __shared__ float buffer[NUMDESCBUFS*128];
  __shared__ float gauss[16];
  __shared__ float gradients[256];
  __shared__ float angles[256];
  const int tx = threadIdx.x;
  const int bx = blockIdx.x;
  gauss[tx] = exp(-(tx-7.5f)*(tx-7.5f)/128.0f);
  float theta = 2.0f*3.1415f/360.0f*d_sift[5*maxPts+bx];
  float sina = sinf(theta);           // cosa -sina
  float cosa = cosf(theta);           // sina  cosa
  float scale = 12.0f/16.0f*d_sift[2*maxPts+bx];
  float ssina = scale*sina;
  float scosa = scale*cosa;
  // Compute angles and gradients
  float xpos = d_sift[0*maxPts+bx] + (tx-7.5f)*scosa + 7.5f*ssina;
  float ypos = d_sift[1*maxPts+bx] + (tx-7.5f)*ssina - 7.5f*scosa;
  for (int i=0;i<128*NUMDESCBUFS/16;i++)
    buffer[16*i+tx] = 0.0f;
  for (int y=0;y<16;y++) {
    float dx = tex2D(tex, xpos+cosa, ypos+sina) - 
      tex2D(tex, xpos-cosa, ypos-sina);
    float dy = tex2D(tex, xpos-sina, ypos+cosa) - 
      tex2D(tex, xpos+sina, ypos-cosa);
    gradients[16*y+tx] = __fmul_rz(__fmul_rz(gauss[y],gauss[tx]),
      sqrtf(__fmul_rz(dx,dx) + __fmul_rz(dy,dy)));
    angles[16*y+tx] = 4.0f/3.1415f*atan2f(dy, dx) + 4.0f;
    xpos -= ssina;
    ypos += scosa;
  }
  __syncthreads();
  if (tx<NUMDESCBUFS) {
    for (int txi=tx;txi<16;txi+=NUMDESCBUFS) {
      int hori = (txi + 2)/4 - 1;
      float horf = (txi - 1.5f)/4.0f - hori;
      float ihorf = 1.0f - horf;
      int veri = -1;
      float verf = 1.0f - 1.5f/4.0f;
      for (int y=0;y<16;y++) {
	int i = 16*y + txi;
	float grad = gradients[i];
	float angf = angles[i];
	int angi = angf;
	int angp = (angi<7 ? angi+1 : 0);
	angf -= angi;
	float iangf = __fadd_rz(1.0f, -angf);
	float iverf = __fadd_rz(1.0f, -verf);
	int hist = 8*(4*veri + hori);
	int p1 = tx + NUMDESCBUFS*(angi+hist);
	int p2 = tx + NUMDESCBUFS*(angp+hist);
	if (txi>=2) { 
	  float grad1 = __fmul_rz(ihorf,grad);
	  if (y>=2) {
	    float grad2 = __fmul_rz(iverf,grad1);
	    buffer[p1+0] += __fmul_rz(iangf,grad2);
	    buffer[p2+0] += __fmul_rz( angf,grad2);
	  }
	  if (y<=14) {
	    float grad2 = __fmul_rz(verf,grad1);
	    buffer[p1+32*NUMDESCBUFS] += __fmul_rz(iangf,grad2); 
	    buffer[p2+32*NUMDESCBUFS] += __fmul_rz( angf,grad2);
	  }
	}
	if (txi<=14) { 
	  float grad1 = __fmul_rz(horf,grad);
	  if (y>=2) {
	    float grad2 = __fmul_rz(iverf,grad1);
	    buffer[p1+8*NUMDESCBUFS] += __fmul_rz(iangf,grad2);
	    buffer[p2+8*NUMDESCBUFS] += __fmul_rz( angf,grad2);
	  }
	  if (y<=14) {
	    float grad2 = __fmul_rz(verf,grad1);
	    buffer[p1+40*NUMDESCBUFS] += __fmul_rz(iangf,grad2);
	    buffer[p2+40*NUMDESCBUFS] += __fmul_rz( angf,grad2);
	  }
	}
	verf += 0.25f;
	if (verf>1.0f) {
	  verf -= 1.0f;
	  veri ++;
	}
      }
    }
  }
  __syncthreads();
  const int t8 = (tx&8)*8;
  const int tx8 = (tx&7);
  if (NUMDESCBUFS>8) {
    for (int i=0;i<64;i++) 
      buffer[NUMDESCBUFS*(i+t8)+tx8] += buffer[NUMDESCBUFS*(i+t8)+tx8+8];
  }
  __syncthreads();
  const int t4 = (tx&12)*8;
  const int tx4 = (tx&3);
  if (NUMDESCBUFS>4) {
    for (int i=0;i<32;i++) 
      buffer[NUMDESCBUFS*(i+t4)+tx4] += buffer[NUMDESCBUFS*(i+t4)+tx4+4];
  }
  __syncthreads();
  const int t2 = (tx&14)*8;
  const int tx2 = (tx&1);
  for (int i=0;i<16;i++) 
    buffer[NUMDESCBUFS*(i+t2)+tx2] += buffer[NUMDESCBUFS*(i+t2)+tx2+2];
  __syncthreads();
  const int t1 = tx*8;
  const int bptr = NUMDESCBUFS*tx+1;
  buffer[bptr] = 0.0f;
  for (int i=0;i<8;i++) {
    int p = NUMDESCBUFS*(i+t1);
    buffer[p] += buffer[p+1];
    buffer[bptr] += __fmul_rz(buffer[p],buffer[p]);
  }
  __syncthreads();
  if (tx<8) buffer[bptr] += buffer[bptr+8*NUMDESCBUFS];
  __syncthreads();
  if (tx<4) buffer[bptr] += buffer[bptr+4*NUMDESCBUFS];
  __syncthreads();
  if (tx<2) buffer[bptr] += buffer[bptr+2*NUMDESCBUFS];
  __syncthreads();
  float isum = 1.0f / sqrt(buffer[1] + buffer[NUMDESCBUFS+1]);
  buffer[bptr] = 0.0f;
  for (int i=0;i<8;i++) {
    int p = NUMDESCBUFS*(i+t1);
    buffer[p] = isum*buffer[p];
    if (buffer[p]>0.2f)
      buffer[p] = 0.2f;
    buffer[bptr] += __fmul_rz(buffer[p],buffer[p]);
  }
  __syncthreads();
  if (tx<8) buffer[bptr] += buffer[bptr+8*NUMDESCBUFS];
  __syncthreads();
  if (tx<4) buffer[bptr] += buffer[bptr+4*NUMDESCBUFS];
  __syncthreads();
  if (tx<2) buffer[bptr] += buffer[bptr+2*NUMDESCBUFS];
  __syncthreads();
  isum = 1.0f / sqrt(buffer[1] + buffer[NUMDESCBUFS+1]);
  for (int i=0;i<8;i++) {
    int p = NUMDESCBUFS*(i+t1);
    d_desc[128*bx+(i+t1)] = __fmul_rz(isum,buffer[p]);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Match two sets of Sift features
///////////////////////////////////////////////////////////////////////////////
__global__ void MatchSiftPoints(SiftPoint *sift1, SiftPoint *sift2, 
  float *corrData, int numPts1, int numPts2)
{
  __shared__ float siftPoint[128];
  __shared__ float sums[16*16];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1 = blockIdx.x;
  const int p2 = blockIdx.y*16 + ty;
  const float *ptr1 = sift1[p1].data;
  const float *ptr2 = sift2[p2].data;
  const int i = ty*16 + tx;
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
  if (tx<2)
    sums[i] += sums[i+2];
  __syncthreads();
  if (tx<1)
    sums[i] += sums[i+1];
  __syncthreads();
  if (ty==0) {
    corrData[p1*gridDim.y*16 + blockIdx.y*16 + tx] = sums[16*tx];
    //printf("corr = %.2f\n", sums[16*tx]);
  }
  __syncthreads();
}


__global__ void FindMaxCorr(float *corrData, SiftPoint *sift1, 
  SiftPoint *sift2, int numPts1, int corrWidth, int siftSize)
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
  //if (p1==1)
  //  printf("tx = %d, score = %.2f, scor2 = %.2f, index = %d\n", 
  //	   tx, maxScore[idx], maxScor2[idx], maxIndex[idx]);
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
    //if (p1==1 && tx<len) 
    //  printf("tx = %d, score = %.2f, scor2 = %.2f, index = %d\n", 
    //	     tx, maxScore[idx], maxScor2[idx], maxIndex[idx]);
  }
  if (tx==6)
    sift1[p1].score = maxScore[ty*16];
  if (tx==7)
    sift1[p1].ambiguity = maxScor2[ty*16] / (maxScore[ty*16] + 1e-6);
  if (tx==8)
    sift1[p1].match = maxIndex[ty*16];
  if (tx==9)
    sift1[p1].match_xpos = sift2[maxIndex[ty*16]].xpos;
  if (tx==10)
    sift1[p1].match_ypos = sift2[maxIndex[ty*16]].ypos;
  __syncthreads();
  //if (tx==0)
  //  printf("index = %d/%d, score = %.2f, ambiguity = %.2f, match = %d\n", 
  //	p1, numPts1, sift1[p1].score, sift1[p1].ambiguity, sift1[p1].match);
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
