//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#ifndef CUDASIFTD_H
#define CUDASIFTD_H

#define WARP_SIZE      16
#define NUM_SCALES      5

#define CONVROW_W     160
#define CONVCOL_W      32
#define CONVCOL_H      40
#define CONVCOL_S       8

#define SUBTRACT_W     32
#define SUBTRACT_H     16
#define SUBTRACTM_W    32
#define SUBTRACTM_H     1

#define SCALEDOWN_W   160
#define SCALEDOWN_H    16

#define MINMAX_W      126 // 126
#define MINMAX_S       64
#define MINMAX_H        8
#define NUMDESCBUFS     4


__device__ __constant__ float d_Kernel[12*16]; // NOTE: Maximum radius 

///////////////////////////////////////////////////////////////////////////////
// Row convolution filter
///////////////////////////////////////////////////////////////////////////////
template<int RADIUS>
__global__ void ConvRowGPU(float *d_Result, float *d_Data, int width, int pitch, int height)
{
  __shared__ float data[CONVROW_W + 2*RADIUS];
  const int tx = threadIdx.x;
  const int minx = blockIdx.x*CONVROW_W;
  const int maxx = min(minx + CONVROW_W, width);
  const int yptr = blockIdx.y*pitch;
  const int loadPos = minx + tx - RADIUS; 
  const int writePos = minx + tx;

  if (loadPos<0) 
    data[tx] = d_Data[yptr];
  else if (loadPos>=width) 
    data[tx] = d_Data[yptr + width-1];
  else
    data[tx] = d_Data[yptr + loadPos];
  __syncthreads();
  if (writePos<maxx && tx<CONVROW_W) {
    float sum = 0.0f;
    for (int i=0;i<=(2*RADIUS);i++) 
      sum += data[tx + i]*d_Kernel[i];
    d_Result[yptr + writePos] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Column convolution filter
///////////////////////////////////////////////////////////////////////////////
template<int RADIUS>
__global__ void ConvColGPU(float *d_Result, float *d_Data, int width, int pitch, int height)
{
  __shared__ float data[CONVCOL_W*(CONVCOL_H + 2*RADIUS)];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int miny = blockIdx.y*CONVCOL_H;
  const int maxy = min(miny + CONVCOL_H, height) - 1;
  const int totStart = miny - RADIUS;
  const int totEnd = maxy + RADIUS;
  const int colStart = blockIdx.x*CONVCOL_W + tx;
  const int colEnd = colStart + (height-1)*pitch;
  const int smemStep = CONVCOL_W*CONVCOL_S;
  const int gmemStep = pitch*CONVCOL_S;
 
  if (colStart<width) {
    int smemPos = ty*CONVCOL_W + tx;
    int gmemPos = colStart + (totStart + ty)*pitch;
    for (int y = totStart+ty;y<=totEnd;y+=blockDim.y){
      if (y<0) 
	data[smemPos] = d_Data[colStart];
      else if (y>=height) 
	data[smemPos] = d_Data[colEnd];
      else 
	data[smemPos] = d_Data[gmemPos];  
      smemPos += smemStep;
      gmemPos += gmemStep;
    }
  }
  __syncthreads();
  if (colStart<width) {
    int smemPos = ty*CONVCOL_W + tx;
    int gmemPos = colStart + (miny + ty)*pitch;
    for (int y=miny+ty;y<=maxy;y+=blockDim.y) {
      float sum = 0.0f;
      for (int i=0;i<=2*RADIUS;i++)
	sum += data[smemPos + i*CONVCOL_W]*d_Kernel[i];
      d_Result[gmemPos] = sum;
      smemPos += smemStep;
      gmemPos += gmemStep;
    }
  }
}

#endif
