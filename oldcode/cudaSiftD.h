//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#ifndef CUDASIFTD_H
#define CUDASIFTD_H

#define WARP_SIZE     16

#define ROW_TILE_W    160
#define COLUMN_TILE_W 16
#define COLUMN_TILE_H 48

#define MINMAX_SIZE   128
#define POSBLK_SIZE   32
#define LOWPASS5_DX 160
#define LOWPASS5_DY 16

__device__ __constant__ float d_Kernel[17]; // NOTE: Maximum radius 

///////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
///////////////////////////////////////////////////////////////////////////////
template<int i> 
__device__ float ConvRow(float *data)
{
    return data[i]*d_Kernel[i] + ConvRow<i-1>(data);
}

template<> 
__device__ float ConvRow<-1>(float *data)
{
    return 0;
}

template<int i> 
__device__ float ConvCol(float *data)
{
    return data[i*COLUMN_TILE_W]*d_Kernel[i] + ConvCol<i-1>(data);
}

template<> 
__device__ float ConvCol<-1>(float *data)
{
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Row convolution filter
///////////////////////////////////////////////////////////////////////////////
template<int RADIUS>
__global__ void ConvRowGPU(float *d_Result, float *d_Data,
  int width, int height)
{
  //Data cache
  __shared__ float data[RADIUS+ROW_TILE_W+RADIUS+1];
  //Current tile and apron limits, relative to row start
  const int tileStart = __mul24(blockIdx.x, ROW_TILE_W);

  //Row start index in d_Data[]
  const int rowStart = __mul24(blockIdx.y, width);
  const int rowEnd = rowStart + width - 1;
  const int loadPos = threadIdx.x - WARP_SIZE + tileStart;
  const int smemPos =  threadIdx.x - WARP_SIZE + RADIUS;

  //Set the entire data cache contents
  if (smemPos>=0) {
    if (loadPos<0)
      data[smemPos] = d_Data[rowStart];
    else if (loadPos>=width) 
      data[smemPos] = d_Data[rowEnd];
    else
      data[smemPos] = d_Data[rowStart + loadPos];
#if 0
    if (rowStart+loadPos<0 || (rowStart+loadPos>= width*height || 
        smemPos<0 || (smemPos >=(RADIUS+ROW_TILE_W+RADIUS+1))
      printf("smemPos: %d\n", smemPos);
#endif
  }
  __syncthreads();
  
  //Clamp tile and apron limits by image borders
  const int tileEnd = tileStart + ROW_TILE_W - 1;
  const int tileEndClamped = min(tileEnd, width - 1);
  const int writePos = tileStart + threadIdx.x;
  
  if (writePos <= tileEndClamped){ 
    const int smemPos = threadIdx.x + RADIUS;
#if 0
    if (rowStart+writePos<0 || rowStart+writePos>=width*height ||
	(smemPos - RADIUS*COLUMN_TILE_W)<0 || 
	(smemPos + RADIUS*COLUMN_TILE_W)>=(RADIUS+ROW_TILE_W+RADIUS+1))
      printf("gmemPos: %d\n", rowStart+writePos);
#endif
    d_Result[rowStart + writePos] = 
      ConvRow<2 * RADIUS>(data + smemPos - RADIUS);;
  }
  __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
// Column convolution filter
///////////////////////////////////////////////////////////////////////////////
template<int RADIUS>
__global__ void ConvColGPU(float *d_Result, float *d_Data, int width,
  int height, int smemStride, int gmemStride)
{
  // Data cache
  __shared__ float data[COLUMN_TILE_W*(RADIUS + COLUMN_TILE_H + RADIUS+1)];

  // Current tile and apron limits, in rows
  const int tileStart = __mul24(blockIdx.y, COLUMN_TILE_H);
  const int tileEnd = tileStart + COLUMN_TILE_H - 1;
  const int apronStart = tileStart - RADIUS;
  const int apronEnd = tileEnd + RADIUS;
  
  // Current column index
  const int columnStart = __mul24(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;
  const int columnEnd = columnStart + __mul24(height-1, width);
    
  if (columnStart<width) {
    // Shared and global memory indices for current column
    int smemPos = __mul24(threadIdx.y, COLUMN_TILE_W) + threadIdx.x;
    int gmemPos = __mul24(apronStart + threadIdx.y, width) + columnStart;
    // Cycle through the entire data cache
    for (int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y){
      if (y<0) 
	data[smemPos] = d_Data[columnStart];
      else if (y>=height) 
	data[smemPos] = d_Data[columnEnd];
      else 
	data[smemPos] = d_Data[gmemPos];  
#if 0
      if (columnStart<0 || columnEnd>= width*height || smemPos<0 || 
	  smemPos>=COLUMN_TILE_W*(RADIUS + COLUMN_TILE_H + RADIUS+1) || 
	  gmemPos<0 || gmemPos>=width*height) 
	printf("pos: %d %d\n", smemPos, gmemPos);
#endif
      smemPos += smemStride;
      gmemPos += gmemStride;
    }
  }
  __syncthreads();

  if (columnStart<width) {
    // Shared and global memory indices for current column
    // Clamp tile and apron limits by image borders
    const int tileEndClamped = min(tileEnd, height - 1);
    int smemPos = __mul24(threadIdx.y + RADIUS, COLUMN_TILE_W) + threadIdx.x;
    int gmemPos = __mul24(tileStart + threadIdx.y , width) + columnStart;
    // Cycle through the tile body, clamped by image borders
    // Calculate and output the results
    for (int y=tileStart+threadIdx.y;y<=tileEndClamped;y+=blockDim.y){
#if 0
      if (gmemPos<0 || gmemPos>=width*height || 
	  (smemPos - RADIUS*COLUMN_TILE_W)<0 || 
	  (smemPos + RADIUS*COLUMN_TILE_W)>=
	  (COLUMN_TILE_W*(RADIUS + COLUMN_TILE_H + RADIUS+1)))
	printf("pos = %d, smemPos = %d, width = %d, height = %d\n", 
	       gmemPos, smemPos, width, height);
#endif
      d_Result[gmemPos] = 
	ConvCol<2*RADIUS>(data + smemPos - RADIUS*COLUMN_TILE_W);;
      smemPos += smemStride;
      gmemPos += gmemStride;
    }
  }
  __syncthreads();
}

#endif
