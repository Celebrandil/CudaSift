//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <cudautils.h>
#include "cudaSiftD.h"
#include "cudaSift.h"

///////////////////////////////////////////////////////////////////////////////
// Kernel configuration
///////////////////////////////////////////////////////////////////////////////

__constant__ float d_Threshold[2];
__constant__ float d_Scales[8], d_Factor;
__constant__ float d_EdgeLimit;
__constant__ int d_MaxNumPoints;

__device__ unsigned int d_PointCounter[1];

texture<float, 2, cudaReadModeElementType> tex;

///////////////////////////////////////////////////////////////////////////////
// Lowpass filter an subsample image
///////////////////////////////////////////////////////////////////////////////
__global__ void ScaleDown(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
  __shared__ float inrow[SCALEDOWN_W+4]; 
  __shared__ float brow[5*(SCALEDOWN_W/2)];
  __shared__ int yRead[SCALEDOWN_H+4], yWrite[SCALEDOWN_H+4];
  #define dx2 (SCALEDOWN_W/2)
  const int tx = threadIdx.x;
  const int tx0 = tx + 0*dx2;
  const int tx1 = tx + 1*dx2;
  const int tx2 = tx + 2*dx2;
  const int tx3 = tx + 3*dx2;
  const int tx4 = tx + 4*dx2;
  const int xStart = blockIdx.x*SCALEDOWN_W;
  const int yStart = blockIdx.y*SCALEDOWN_H;
  const int xWrite = xStart/2 + tx;
  const float *k = d_Kernel;
  if (tx<SCALEDOWN_H+4) {
    int y = yStart + tx - 1;
    y = (y<0 ? 0 : y);
    y = (y>=height ? height-1 : y);
    yRead[tx] = y*pitch;
    yWrite[tx] = (yStart + tx - 4)/2 * newpitch;
  }
  __syncthreads();
  int xRead = xStart + tx - WARP_SIZE;
  xRead = (xRead<0 ? 0 : xRead);
  xRead = (xRead>=width ? width-1 : xRead);
  for (int dy=0;dy<SCALEDOWN_H+4;dy+=5) {
    if (tx>=WARP_SIZE-2) 
      inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+0] + xRead];
    __syncthreads();
    if (tx<dx2) 
      brow[tx0] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
    __syncthreads();
    if (tx<dx2 && dy>=4 && !(dy&1)) 
      d_Result[yWrite[dy+0] + xWrite] = k[2]*brow[tx2] + k[0]*(brow[tx0]+brow[tx4]) + k[1]*(brow[tx1]+brow[tx3]);
    if (dy<(SCALEDOWN_H+3)) {
      if (tx>=WARP_SIZE-2) 
	inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+1] + xRead];
      __syncthreads();
      if (tx<dx2)
	brow[tx1] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();
      if (tx<dx2 && dy>=3 && (dy&1)) 
	d_Result[yWrite[dy+1] + xWrite] = k[2]*brow[tx3] + k[0]*(brow[tx1]+brow[tx0]) + k[1]*(brow[tx2]+brow[tx4]); 
    }
    if (dy<(SCALEDOWN_H+2)) {
      if (tx>=WARP_SIZE-2) 
	inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+2] + xRead];
      __syncthreads();
      if (tx<dx2)
	brow[tx2] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();
      if (tx<dx2 && dy>=2 && !(dy&1)) 
	d_Result[yWrite[dy+2] + xWrite] = k[2]*brow[tx4] + k[0]*(brow[tx2]+brow[tx1]) + k[1]*(brow[tx3]+brow[tx0]); 
    }
    if (dy<(SCALEDOWN_H+1)) {
      if (tx>=WARP_SIZE-2) 
	inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+3] + xRead];
      __syncthreads();
      if (tx<dx2)
	brow[tx3] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();
      if (tx<dx2 && dy>=1 && (dy&1)) 
	d_Result[yWrite[dy+3] + xWrite] = k[2]*brow[tx0] + k[0]*(brow[tx3]+brow[tx2]) + k[1]*(brow[tx4]+brow[tx1]); 
    }
    if (dy<SCALEDOWN_H) {
      if (tx>=WARP_SIZE-2) 
	inrow[tx-WARP_SIZE+2] = d_Data[yRead[dy+4] + xRead];
      __syncthreads();
      if (tx<dx2)
	brow[tx4] = k[0]*(inrow[2*tx]+inrow[2*tx+4]) + k[1]*(inrow[2*tx+1]+inrow[2*tx+3]) + k[2]*inrow[2*tx+2];
      __syncthreads();
      if (tx<dx2 && !(dy&1)) 
	d_Result[yWrite[dy+4] + xWrite] = k[2]*brow[tx1] + k[0]*(brow[tx4]+brow[tx3]) + k[1]*(brow[tx0]+brow[tx2]); 
    }
    __syncthreads();
  }
}

///////////////////////////////////////////////////////////////////////////////
// Subtract two images
///////////////////////////////////////////////////////////////////////////////
__global__ void Subtract(float *d_Result, float *d_Data1, float *d_Data2, int width, int pitch, int height)
{
  const int x = blockIdx.x*SUBTRACT_W + threadIdx.x;
  const int y = blockIdx.y*SUBTRACT_H + threadIdx.y;
  int p = y*pitch + x;
  if (x<width && y<height)
    d_Result[p] = d_Data1[p] - d_Data2[p];
  __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
// Extract Sift descriptors
///////////////////////////////////////////////////////////////////////////////
 __global__ void ExtractSiftDescriptors(float *g_Data, SiftPoint *d_sift, int fstPts, float subsampling)
{
  __shared__ float buffer[NUMDESCBUFS*128];
  __shared__ float gauss[16];
  __shared__ float gradients[256];
  __shared__ float angles[256];
  const int tx = threadIdx.x; // 0 -> 16
  const int bx = blockIdx.x + fstPts;  // 0 -> numPts
  gauss[tx] = exp(-(tx-7.5f)*(tx-7.5f)/128.0f);
  __syncthreads();
  float theta = 2.0f*3.1415f/360.0f*d_sift[bx].orientation;
  float sina = sinf(theta);           // cosa -sina
  float cosa = cosf(theta);           // sina  cosa
  float scale = 12.0f/16.0f*d_sift[bx].scale;
  float ssina = scale*sina; 
  float scosa = scale*cosa;
  // Compute angles and gradients
  float xpos = d_sift[bx].xpos + (tx-7.5f)*scosa + 7.5f*ssina;
  float ypos = d_sift[bx].ypos + (tx-7.5f)*ssina - 7.5f*scosa;
  for (int i=0;i<128*NUMDESCBUFS/16;i++)
    buffer[16*i+tx] = 0.0f;
  for (int y=0;y<16;y++) {
    float dx = tex2D(tex, xpos+cosa, ypos+sina) - tex2D(tex, xpos-cosa, ypos-sina);
    float dy = tex2D(tex, xpos-sina, ypos+cosa) - tex2D(tex, xpos+sina, ypos-cosa);
    gradients[16*y+tx] = gauss[y]*gauss[tx] * sqrtf(dx*dx + dy*dy);
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
	float iangf = 1.0f - angf;
	float iverf = 1.0f - verf;
	int hist = 8*(4*veri + hori);
	//printf("%d\n", hist);
	int p1 = tx + NUMDESCBUFS*(angi+hist);
	int p2 = tx + NUMDESCBUFS*(angp+hist);
	if (txi>=2) { 
	  float grad1 = ihorf*grad;
	  if (y>=2) {
	    float grad2 = iverf*grad1;
	    buffer[p1+0] += iangf*grad2;
	    buffer[p2+0] +=  angf*grad2;
	  }
	  if (y<=14) {
	    float grad2 = verf*grad1;
	    buffer[p1+32*NUMDESCBUFS] += iangf*grad2; 
	    buffer[p2+32*NUMDESCBUFS] +=  angf*grad2;
	  }
	}
	if (txi<=14) { 
	  float grad1 = horf*grad;
	  if (y>=2) {
	    float grad2 = iverf*grad1;
	    buffer[p1+8*NUMDESCBUFS] += iangf*grad2;
	    buffer[p2+8*NUMDESCBUFS] +=  angf*grad2;
	  }
	  if (y<=14) {
	    float grad2 = verf*grad1;
	    buffer[p1+40*NUMDESCBUFS] += iangf*grad2;
	    buffer[p2+40*NUMDESCBUFS] +=  angf*grad2;
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
  const int t2 = (tx&14)*8;
  const int tx2 = (tx&1);
  for (int i=0;i<16;i++) 
    buffer[NUMDESCBUFS*(i+t2)+tx2] += buffer[NUMDESCBUFS*(i+t2)+tx2+2];
  __syncthreads();

  const int t1 = tx*8;                 
  const int bptr = NUMDESCBUFS*tx + 2;   
  buffer[bptr] = 0.0f;
  for (int i=0;i<8;i++) {
    int p = NUMDESCBUFS*(i+t1);  
    buffer[p] += buffer[p+1];
    buffer[bptr] += buffer[p]*buffer[p];
  }
  __syncthreads();

  if (tx<8) 
    buffer[bptr] += buffer[bptr+8*NUMDESCBUFS];
  __syncthreads();
  if (tx<4) 
    buffer[bptr] += buffer[bptr+4*NUMDESCBUFS];
  __syncthreads();
  if (tx<2) 
    buffer[bptr] += buffer[bptr+2*NUMDESCBUFS];
  __syncthreads();
  float isum = 1.0f / sqrt(buffer[2] + buffer[NUMDESCBUFS+2]);

  buffer[bptr] = 0.0f;
  for (int i=0;i<8;i++) {
    int p = NUMDESCBUFS*(i+t1);
    buffer[p] = isum*buffer[p];
    if (buffer[p]>0.2f)
      buffer[p] = 0.2f;
    buffer[bptr] += buffer[p]*buffer[p];
  }
  __syncthreads();

  if (tx<8) 
    buffer[bptr] += buffer[bptr+8*NUMDESCBUFS];
  __syncthreads();
  if (tx<4) 
    buffer[bptr] += buffer[bptr+4*NUMDESCBUFS];
  __syncthreads();
  if (tx<2) 
    buffer[bptr] += buffer[bptr+2*NUMDESCBUFS];
  __syncthreads();
  isum = 1.0f / sqrt(buffer[2] + buffer[NUMDESCBUFS+2]);

  float *desc = d_sift[bx].data;
  for (int i=0;i<8;i++) {
    int p = NUMDESCBUFS*(i+t1);
    desc[i+t1] = isum*buffer[p];
  }
  d_sift[bx].xpos *= subsampling;
  d_sift[bx].ypos *= subsampling;
  d_sift[bx].scale *= subsampling;
}
 
 __global__ void ExtractSiftDescriptors2(float *g_Data, SiftPoint *d_sift, int fstPts, float subsampling)
{
  __shared__ float gauss[16];
  __shared__ float buffer[128];
  __shared__ float tsum, isum;

  const int tx = threadIdx.x; // 0 -> 16
  const int ty = threadIdx.y; // 0 -> 16
  const int idx = ty*16 + tx;
  const int bx = blockIdx.x + fstPts;  // 0 -> numPts
  if (ty==8)
    gauss[tx] = exp(-(tx-7.5f)*(tx-7.5f)/128.0f);
  if (idx<128)
    buffer[idx] = 0.0f;
  __syncthreads();

  // Compute angles and gradients
  float theta = 2.0f*3.1415f/360.0f*d_sift[bx].orientation;
  float sina = sinf(theta);           // cosa -sina
  float cosa = cosf(theta);           // sina  cosa
  float scale = 12.0f/16.0f*d_sift[bx].scale;
  float ssina = scale*sina; 
  float scosa = scale*cosa;
  float xpos = d_sift[bx].xpos + (tx-7.5f)*scosa - (ty-7.5f)*ssina;
  float ypos = d_sift[bx].ypos + (tx-7.5f)*ssina + (ty-7.5f)*scosa;
  float dx = tex2D(tex, xpos+cosa, ypos+sina) - tex2D(tex, xpos-cosa, ypos-sina);
  float dy = tex2D(tex, xpos-sina, ypos+cosa) - tex2D(tex, xpos+sina, ypos-cosa);
  float grad = gauss[ty]*gauss[tx] * sqrtf(dx*dx + dy*dy);
  float angf = 4.0f/3.1415f*atan2f(dy, dx) + 4.0f;

  int hori = (tx + 2)/4 - 1;      // Convert from (tx,ty,angle) to bins      
  float horf = (tx - 1.5f)/4.0f - hori;  
  float ihorf = 1.0f - horf;           
  int veri = (ty + 2)/4 - 1;
  float verf = (ty - 1.5f)/4.0f - veri;
  float iverf = 1.0f - verf;
  int angi = angf;
  int angp = (angi<7 ? angi+1 : 0);
  angf -= angi;
  float iangf = 1.0f - angf;

  int hist = 8*(4*veri + hori);   // Each gradient measure is interpolated 
  int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
  int p2 = angp + hist;
  if (tx>=2) { 
    float grad1 = ihorf*grad;
    if (ty>=2) {   // Upper left
      float grad2 = iverf*grad1;
      atomicAdd(buffer + p1, iangf*grad2);
      atomicAdd(buffer + p2,  angf*grad2);
    }
    if (ty<=14) {  // Lower left
      float grad2 = verf*grad1;
      atomicAdd(buffer + p1+32, iangf*grad2); 
      atomicAdd(buffer + p2+32,  angf*grad2);
    }
  }
  if (tx<=14) { 
    float grad1 = horf*grad;
    if (ty>=2) {    // Upper right
      float grad2 = iverf*grad1;
      atomicAdd(buffer + p1+8, iangf*grad2);
      atomicAdd(buffer + p2+8,  angf*grad2);
    }
    if (ty<=14) {   // Lower right
      float grad2 = verf*grad1;
      atomicAdd(buffer + p1+40, iangf*grad2);
      atomicAdd(buffer + p2+40,  angf*grad2);
    }
  }
  __syncthreads();

  if (idx==0)                       // Normalize twice and suppress 
    tsum = 0.0f;                    // peaks first time
  __syncthreads();
  if (idx<128) 
    atomicAdd(&tsum, buffer[idx]*buffer[idx]);
  __syncthreads();
  if (idx==0) {    // Normalize first time
    isum = 1.0f / sqrt(tsum);
    tsum = 0.0f;
  }
  __syncthreads();
  if (idx<128) {
    buffer[idx] = isum*buffer[idx];
    if (buffer[idx]>0.2f)
      buffer[idx] = 0.2f;
    atomicAdd(&tsum, buffer[idx]*buffer[idx]);
  }
  __syncthreads();
  if (idx==0)      // Normalize second time
    isum = 1.0f / sqrt(tsum);
  __syncthreads();

  float *desc = d_sift[bx].data;
  if (idx<128)
    desc[idx] = isum*buffer[idx];
  if (idx==0) {
    d_sift[bx].xpos *= subsampling;
    d_sift[bx].ypos *= subsampling;
    d_sift[bx].scale *= subsampling;
  }
}
 
///============= New functions

#if 1

__global__ void FindPoints(float *d_Data1, float *d_Data2, float *d_Data3, float *d_Sift, int width, int pitch, int height)
{
  #define MEMWID (MINMAX_W + 2)
  __shared__ float data1[3*MEMWID], data2[3*MEMWID], data3[3*MEMWID];
  __shared__ float ymin1[MEMWID],   ymin2[MEMWID],   ymin3[MEMWID];
  __shared__ float ymax1[MEMWID],   ymax2[MEMWID],   ymax3[MEMWID];

  const int tx = threadIdx.x;
  const int minx = blockIdx.x*MINMAX_W;
  const int maxx = min(minx + MINMAX_W, width);
  const int xpos = minx + tx;

  int ptr0 = tx;
  int ptr1 = tx;
  int yq = 0;
  for (int y=0;y<MINMAX_H+2;y++) {

    int ypos = MINMAX_H*blockIdx.y + y - 1;
    int yptr = min(max(ypos, 0), height - 1)*pitch;
    int xposr = xpos - 1;
    int ptr2 = yq*MEMWID + tx;

    if (xposr<0) {
      data1[ptr2] = 0;
      data2[ptr2] = 0;
      data3[ptr2] = 0;
    } else if (xposr>=width) {
      data1[ptr2] = 0;
      data2[ptr2] = 0;
      data3[ptr2] = 0;
    } else {
      data1[ptr2] = d_Data1[yptr + xposr];
      data2[ptr2] = d_Data2[yptr + xposr];
      data3[ptr2] = d_Data3[yptr + xposr];
    }
    //__syncthreads();
  
    if (y>1) {
      float min1 = fminf(fminf(data1[ptr0], data1[ptr1]), data1[ptr2]);
      float min2 = fminf(fminf(data2[ptr0], data2[ptr1]), data2[ptr2]);
      float min3 = fminf(fminf(data3[ptr0], data3[ptr1]), data3[ptr2]);
      float max1 = fmaxf(fmaxf(data1[ptr0], data1[ptr1]), data1[ptr2]);
      float max2 = fmaxf(fmaxf(data2[ptr0], data2[ptr1]), data2[ptr2]);
      float max3 = fmaxf(fmaxf(data3[ptr0], data3[ptr1]), data3[ptr2]);
      ymin1[tx] = min1;
      ymin2[tx] = fminf(fminf(min1, min2), min3);
      ymin3[tx] = min3;
      ymax1[tx] = max1;
      ymax2[tx] = fmaxf(fmaxf(max1, max2), max3);
      ymax3[tx] = max3;
    }
    //__syncthreads();

    if (y>1) {
      if (tx<MINMAX_W && xpos<maxx) {
	float minv = fminf(fminf(fminf(fminf(fminf(ymin2[tx], ymin2[tx+2]), ymin1[tx+1]), ymin3[tx+1]), data2[ptr0+1]), data2[ptr2+1]);
	minv = fminf(minv, d_Threshold[1]);
	float maxv = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(ymax2[tx], ymax2[tx+2]), ymax1[tx+1]), ymax3[tx+1]), data2[ptr0+1]), data2[ptr2+1]);
	maxv = fmaxf(maxv, d_Threshold[0]);
	float val = data2[ptr1+1];
	if (val<minv || val>maxv) {
	  float dxx = 2.0f*val - data2[ptr1+0] - data2[ptr1+2];
	  float dyy = 2.0f*val - data2[ptr0+1] - data2[ptr2+1];
	  float dxy = 0.25f*(data2[ptr2+2] + data2[ptr0+0] - data2[ptr0+2] - data2[ptr2+0]);
	  float tra = dxx + dyy;
	  float det = dxx*dyy - dxy*dxy;
	  if (tra*tra<d_EdgeLimit*det) {
	    float edge = __fdividef(tra*tra, det);
	    float dx = 0.5f*(data2[ptr1+2] - data2[ptr1+0]);
	    float dy = 0.5f*(data2[ptr2+1] - data2[ptr0+1]); 
	    float ds = 0.5f*(data1[ptr1+1] - data3[ptr1+1]); 
	    float dss = 2.0f*val - data3[ptr1+1] - data1[ptr1+1];
	    float dxs = 0.25f*(data3[ptr1+2] + data1[ptr1+0] - data1[ptr1+2] - data3[ptr1+0]);
	    float dys = 0.25f*(data3[ptr2+1] + data1[ptr0+1] - data3[ptr0+1] - data1[ptr2+1]);
	    float idxx = dyy*dss - dys*dys;
	    float idxy = dys*dxs - dxy*dss;  
	    float idxs = dxy*dys - dyy*dxs;
	    float idyy = dxx*dss - dxs*dxs;
	    float idys = dxy*dxs - dxx*dys;
	    float idss = dxx*dyy - dxy*dxy;
	    float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
	    float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
	    float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
	    float pds = idet*(idxs*dx + idys*dy + idss*ds);
	    if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f) {
	      pdx = __fdividef(dx, dxx);
	      pdy = __fdividef(dy, dyy);
	      pds = __fdividef(ds, dss);
	    }
	    float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
	    int maxPts = d_MaxNumPoints;
	    unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
	    idx = (idx>=maxPts ? maxPts-1 : idx);
	    d_Sift[idx + 0*maxPts] = xpos + pdx;
	    d_Sift[idx + 1*maxPts] = ypos - 1 + pdy;
	    d_Sift[idx + 2*maxPts] = d_Scales[0] * exp2f(pds*d_Factor);
	    d_Sift[idx + 3*maxPts] = val + dval;
	    d_Sift[idx + 4*maxPts] = edge;
	    //printf("idx: %d %.1f %.1f %.2f\n", idx, d_Sift[idx + 0*maxPts], d_Sift[idx + 1*maxPts], edge);
	  }
	}
      }
    }
    __syncthreads();

    ptr0 = ptr1;
    ptr1 = ptr2;
    yq = (yq<2 ? yq+1 : 0);
  }
}

#else

__global__ void FindPoints(float *d_Data1, float *d_Data2, float *d_Data3, float *d_Sift, int width, int pitch, int height)
{
  #define MEMWID (MINMAX_W + 2)
  __shared__ float data1[3*MEMWID], data2[3*MEMWID], data3[3*MEMWID];
  __shared__ float ymin1[MEMWID],   ymin2[MEMWID],   ymin3[MEMWID];
  __shared__ float ymax1[MEMWID],   ymax2[MEMWID],   ymax3[MEMWID];

  const int tx = threadIdx.x;
  const int minx = blockIdx.x*MINMAX_W;
  const int maxx = min(minx + MINMAX_W, width);

  int ptr0 = 0;
  int ptr1 = 0;
  int yq = 0;
  for (int y=0;y<MINMAX_H+2;y++) {
 
    int ypos = MINMAX_H*blockIdx.y + y - 1;
    int yptr = min(max(ypos, 0), height - 1)*pitch;
    int ptr2 = yq*MEMWID;

    for (int idx=tx;idx<MEMWID;idx+=MINMAX_S) {
      int xpos = minx - 1 + idx;
      int p0 = ptr0 + idx;
      int p1 = ptr1 + idx;
      int p2 = ptr2 + idx;
      if (xpos<0) {
	data1[p2] = 0;
	data2[p2] = 0;
	data3[p2] = 0;
      } else if (xpos>=width) {
	data1[p2] = 0;
	data2[p2] = 0;
	data3[p2] = 0;
      } else {
	data1[p2] = d_Data1[yptr + xpos];
	data2[p2] = d_Data2[yptr + xpos];
	data3[p2] = d_Data3[yptr + xpos];
      }
      if (y>1) {
	float min1 = fminf(fminf(data1[p0], data1[p1]), data1[p2]);
	float max1 = fmaxf(fmaxf(data1[p0], data1[p1]), data1[p2]);
	float min2 = fminf(fminf(data2[p0], data2[p1]), data2[p2]);
	float max2 = fmaxf(fmaxf(data2[p0], data2[p1]), data2[p2]);
	float min3 = fminf(fminf(data3[p0], data3[p1]), data3[p2]);
	float max3 = fmaxf(fmaxf(data3[p0], data3[p1]), data3[p2]);
	ymin1[idx] = min1;
	ymin2[idx] = fminf(fminf(min1, min2), min3);
	ymin3[idx] = min3;
	ymax1[idx] = max1;
	ymax2[idx] = fmaxf(fmaxf(max1, max2), max3);
	ymax3[idx] = max3;
      }
    }
    __syncthreads();

    if (y>1) {
      for (int idx=tx;idx<MINMAX_W && (minx+idx)<maxx;idx+=MINMAX_S) {
	const int xpos = minx + idx;
	int p0 = ptr0 + idx;
	int p1 = ptr1 + idx;
	int p2 = ptr2 + idx;
	float minv = fminf(fminf(fminf(fminf(fminf(ymin2[idx], ymin2[idx+2]), ymin1[idx+1]), ymin3[idx+1]), data2[p0+1]), data2[p2+1]);
	minv = fminf(minv, d_Threshold[1]);
	float maxv = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(ymax2[idx], ymax2[idx+2]), ymax1[idx+1]), ymax3[idx+1]), data2[p0+1]), data2[p2+1]);
	maxv = fmaxf(maxv, d_Threshold[0]);
	float val = data2[p1+1];
	if (val<minv || val>maxv) {
	  float dxx = 2.0f*val - data2[p1+0] - data2[p1+2];
	  float dyy = 2.0f*val - data2[p0+1] - data2[p2+1];
	  float dxy = 0.25f*(data2[p2+2] + data2[p0+0] - data2[p0+2] - data2[p2+0]);
	  float tra = dxx + dyy;
	  float det = dxx*dyy - dxy*dxy;
	  if (tra*tra<d_EdgeLimit*det) {
	    float edge = __fdividef(tra*tra, det);
	    float dx = 0.5f*(data2[p1+2] - data2[p1+0]);
	    float dy = 0.5f*(data2[p2+1] - data2[p0+1]); 
	    float ds = 0.5f*(data1[p1+1] - data3[p1+1]); 
	    float dss = 2.0f*val - data3[p1+1] - data1[p1+1];
	    float dxs = 0.25f*(data3[p1+2] + data1[p1+0] - data1[p1+2] - data3[p1+0]);
	    float dys = 0.25f*(data3[p2+1] + data1[p0+1] - data3[p0+1] - data1[p2+1]);
	    float idxx = dyy*dss - dys*dys;
	    float idxy = dys*dxs - dxy*dss;  
	    float idxs = dxy*dys - dyy*dxs;
	    float idyy = dxx*dss - dxs*dxs;
	    float idys = dxy*dxs - dxx*dys;
	    float idss = dxx*dyy - dxy*dxy;
	    float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
	    float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
	    float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
	    float pds = idet*(idxs*dx + idys*dy + idss*ds);
	    if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f) {
	      pdx = __fdividef(dx, dxx);
	      pdy = __fdividef(dy, dyy);
	      pds = __fdividef(ds, dss);
	    }
	    float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
	    int maxPts = d_MaxNumPoints;
	    unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
	    idx = (idx>=maxPts ? maxPts-1 : idx);
	    d_Sift[i + 0*maxPts] = xpos + pdx;
	    d_Sift[i + 1*maxPts] = ypos - 1 + pdy;
	    d_Sift[i + 2*maxPts] = d_Scales[0] * exp2f(pds*d_Factor);
	    d_Sift[i + 3*maxPts] = val + dval;
	    d_Sift[i + 4*maxPts] = edge;
	    //printf("i: %d %.1f %.1f %.2f\n", i, d_Sift[i + 0*maxPts], d_Sift[i + 1*maxPts], edge);
	  }
	}
      }
    }
    __syncthreads();

    ptr0 = ptr1;
    ptr1 = ptr2;
    yq = (yq<2 ? yq+1 : 0);
  }
}

#endif

__global__ void ComputeOrientations(float *g_Data, SiftPoint *d_Sift, int fstPts, int w, int h)
{
  __shared__ float data[16*15];
  __shared__ float hist[32*13];
  __shared__ float gauss[16];
  const int tx = threadIdx.x;
  const int bx = blockIdx.x + fstPts;
  for (int i=0;i<13;i++)
    hist[i*32+tx] = 0.0f;
  __syncthreads();
  float i2sigma2 = -1.0f/(2.0f*3.0f*3.0f);
  if (tx<15) 
    gauss[tx] = exp(i2sigma2*(tx-7)*(tx-7));
  int xp = (int)(d_Sift[bx].xpos - 6.5f);
  int yp = (int)(d_Sift[bx].ypos - 6.5f);
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
    d_Sift[bx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);
    if (maxval2>0.8f*maxval1) {
      float val1 = hist[32+((i2+1)&31)];
      float val2 = hist[32+((i2+31)&31)];
      float peak = i2 + 0.5f*(val1-val2) / (2.0f*maxval2-val1-val2);
      unsigned int idx = atomicInc(d_PointCounter, d_MaxNumPoints-1);
      d_Sift[idx].xpos = d_Sift[bx].xpos;
      d_Sift[idx].ypos = d_Sift[bx].ypos;
      d_Sift[idx].scale = d_Sift[bx].scale;
      d_Sift[idx].sharpness = d_Sift[bx].sharpness;
      d_Sift[idx].edgeness = d_Sift[bx].edgeness;
      d_Sift[idx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);;
    } 
  }
} 

__global__ void ComputeOrientations2(float *g_Data, SiftPoint *d_Sift, int fstPts)
{
  __shared__ float hist[64];
  __shared__ float gauss[11];
  const int tx = threadIdx.x;
  const int bx = blockIdx.x + fstPts;
  float i2sigma2 = -1.0f/(4.5f*d_Sift[bx].scale*d_Sift[bx].scale);
  if (tx<11) 
    gauss[tx] = exp(i2sigma2*(tx-5)*(tx-5));
  if (tx<64)
    hist[tx] = 0.0f;
  __syncthreads();
  float xp = d_Sift[bx].xpos - 5.0f;
  float yp = d_Sift[bx].ypos - 5.0f;
  int yd = tx/11;
  int xd = tx - yd*11;
  float xf = xp + xd;
  float yf = yp + yd;
  if (yd<11) {
    float dx = tex2D(tex, xf+1.0, yf) - tex2D(tex, xf-1.0, yf); 
    float dy = tex2D(tex, xf, yf+1.0) - tex2D(tex, xf, yf-1.0); 
    int bin = 16.0f*atan2f(dy, dx)/3.1416f + 16.5f;
    if (bin>31)
      bin = 0;
    float grad = sqrtf(dx*dx + dy*dy);
    atomicAdd(&hist[bin], grad*gauss[xd]*gauss[yd]);
  }
  __syncthreads();
  int x1m = (tx>=1 ? tx-1 : tx+31);
  int x1p = (tx<=30 ? tx+1 : tx-31);
  if (tx<32) {
    int x2m = (tx>=2 ? tx-2 : tx+30);
    int x2p = (tx<=29 ? tx+2 : tx-30);
    hist[tx+32] = 6.0f*hist[tx] + 4.0f*(hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
  }
  __syncthreads();
  if (tx<32) {
    float v = hist[32+tx];
    hist[tx] = (v>hist[32+x1m] && v>=hist[32+x1p] ? v : 0.0f);
  }
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
    d_Sift[bx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);
    if (maxval2>0.8f*maxval1) {
      float val1 = hist[32+((i2+1)&31)];
      float val2 = hist[32+((i2+31)&31)];
      float peak = i2 + 0.5f*(val1-val2) / (2.0f*maxval2-val1-val2);
      unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
      if (idx<d_MaxNumPoints) {
	d_Sift[idx].xpos = d_Sift[bx].xpos;
	d_Sift[idx].ypos = d_Sift[bx].ypos;
	d_Sift[idx].scale = d_Sift[bx].scale;
	d_Sift[idx].sharpness = d_Sift[bx].sharpness;
	d_Sift[idx].edgeness = d_Sift[bx].edgeness;
	d_Sift[idx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);;
      }
    } 
  }
} 

///////////////////////////////////////////////////////////////////////////////
// Subtract two images (multi-scale version)
///////////////////////////////////////////////////////////////////////////////

__global__ void SubtractMulti(float *d_Result, float *d_Data, int width, int pitch, int height)
{
  const int x = blockIdx.x*SUBTRACTM_W + threadIdx.x;
  const int y = blockIdx.y*SUBTRACTM_H + threadIdx.y;
  int sz = height*pitch;
  int p = threadIdx.z*sz + y*pitch + x;
  if (x<width && y<height)
    d_Result[p] = d_Data[p] - d_Data[p + sz];
  __syncthreads();
}

__global__ void FindPointsMulti(float *d_Data0, float *d_Sift, int width, int pitch, int height, int nScales)
{
  #define MEMWID (MINMAX_W + 2)
  __shared__ float data1[3*MEMWID], data2[3*MEMWID], data3[3*MEMWID];
  __shared__ float ymin1[MEMWID],   ymin2[MEMWID],   ymin3[MEMWID];
  __shared__ float ymax1[MEMWID],   ymax2[MEMWID],   ymax3[MEMWID];

  const int tx = threadIdx.x;
  const int block = blockIdx.x/nScales; 
  const int scale = blockIdx.x - nScales*block;
  const int minx = block*MINMAX_W;
  const int maxx = min(minx + MINMAX_W, width);
  const int xpos = minx + tx;
  const int size = pitch*height;
  const float *d_Data1 = d_Data0 + size*scale;
  const float *d_Data2 = d_Data1 + size;
  const float *d_Data3 = d_Data2 + size;
  //if (block==0 && blockIdx.y==0 && tx==0)
  //  printf("%08x %08x %08x %d %d\n", d_Data1, d_Data2, d_Data3, pitch, height);

  int ptr0 = tx;
  int ptr1 = tx;
  int yq = 0;
  for (int y=0;y<MINMAX_H+2;y++) {

    int xposr = xpos - 1;
    int ypos = MINMAX_H*blockIdx.y + y - 1;
    int yptr = min(max(ypos, 0), height - 1)*pitch;

    int ptr2 = yq*MEMWID + tx;
    if (xposr<0) {
      data1[ptr2] = 0;
      data2[ptr2] = 0;
      data3[ptr2] = 0;
    } else if (xposr>=width) {
      data1[ptr2] = 0;
      data2[ptr2] = 0;
      data3[ptr2] = 0;
    } else {
      data1[ptr2] = d_Data1[yptr + xposr];
      data2[ptr2] = d_Data2[yptr + xposr];
      data3[ptr2] = d_Data3[yptr + xposr];
    }
    __syncthreads();
    if (y>1) {
      float min1 = fminf(fminf(data1[ptr0], data1[ptr1]), data1[ptr2]);
      float min2 = fminf(fminf(data2[ptr0], data2[ptr1]), data2[ptr2]);
      float min3 = fminf(fminf(data3[ptr0], data3[ptr1]), data3[ptr2]);
      float max1 = fmaxf(fmaxf(data1[ptr0], data1[ptr1]), data1[ptr2]);
      float max2 = fmaxf(fmaxf(data2[ptr0], data2[ptr1]), data2[ptr2]);
      float max3 = fmaxf(fmaxf(data3[ptr0], data3[ptr1]), data3[ptr2]);
      ymin1[tx] = min1;
      ymin2[tx] = fminf(fminf(min1, min2), min3);
      ymin3[tx] = min3;
      ymax1[tx] = max1;
      ymax2[tx] = fmaxf(fmaxf(max1, max2), max3);
      ymax3[tx] = max3;
    }
    __syncthreads();
    if (y>1) {
      if (tx<MINMAX_W && xpos<maxx) {
	float minv = fminf(fminf(fminf(fminf(fminf(ymin2[tx], ymin2[tx+2]), ymin1[tx+1]), ymin3[tx+1]), data2[ptr0+1]), data2[ptr2+1]);
	minv = fminf(minv, d_Threshold[1]);
	float maxv = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(ymax2[tx], ymax2[tx+2]), ymax1[tx+1]), ymax3[tx+1]), data2[ptr0+1]), data2[ptr2+1]);
	maxv = fmaxf(maxv, d_Threshold[0]);
	float val = data2[ptr1+1];
	if (val<minv || val>maxv) {
	  float dxx = 2.0f*val - data2[ptr1+0] - data2[ptr1+2];
	  float dyy = 2.0f*val - data2[ptr0+1] - data2[ptr2+1];
	  float dxy = 0.25f*(data2[ptr2+2] + data2[ptr0+0] - data2[ptr0+2] - data2[ptr2+0]);
	  float tra = dxx + dyy;
	  float det = dxx*dyy - dxy*dxy;
	  if (tra*tra<d_EdgeLimit*det) {
	    float edge = __fdividef(tra*tra, det);
	    float dx = 0.5f*(data2[ptr1+2] - data2[ptr1+0]);
	    float dy = 0.5f*(data2[ptr2+1] - data2[ptr0+1]); 
	    float ds = 0.5f*(data1[ptr1+1] - data3[ptr1+1]); 
	    float dss = 2.0f*val - data3[ptr1+1] - data1[ptr1+1];
	    float dxs = 0.25f*(data3[ptr1+2] + data1[ptr1+0] - data1[ptr1+2] - data3[ptr1+0]);
	    float dys = 0.25f*(data3[ptr2+1] + data1[ptr0+1] - data3[ptr0+1] - data1[ptr2+1]);
	    float idxx = dyy*dss - dys*dys;
	    float idxy = dys*dxs - dxy*dss;  
	    float idxs = dxy*dys - dyy*dxs;
	    float idyy = dxx*dss - dxs*dxs;
	    float idys = dxy*dxs - dxx*dys;
	    float idss = dxx*dyy - dxy*dxy;
	    float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
	    float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
	    float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
	    float pds = idet*(idxs*dx + idys*dy + idss*ds);
	    if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f) {
	      pdx = __fdividef(dx, dxx);
	      pdy = __fdividef(dy, dyy);
	      pds = __fdividef(ds, dss);
	    }
	    float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
	    int maxPts = d_MaxNumPoints;
	    unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
	    idx = (idx>=maxPts ? maxPts-1 : idx);
	    d_Sift[idx + 0*maxPts] = xpos + pdx;
	    d_Sift[idx + 1*maxPts] = ypos - 1 + pdy;
	    d_Sift[idx + 2*maxPts] = d_Scales[scale] * exp2f(pds*d_Factor);
	    d_Sift[idx + 3*maxPts] = val + dval;
	    d_Sift[idx + 4*maxPts] = edge;
	  }
	}
      }
    }
    __syncthreads();
    ptr0 = ptr1;
    ptr1 = ptr2;
    yq = (yq<2 ? yq+1 : 0);
  }
}

__global__ void FindPointsMulti2(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, int nScales)
{
  #define MEMWID (MINMAX_W + 2)
  __shared__ float ymin1[MEMWID], ymin2[MEMWID], ymin3[MEMWID];
  __shared__ float ymax1[MEMWID], ymax2[MEMWID], ymax3[MEMWID];
  __shared__ unsigned int cnt;
  __shared__ unsigned short points[96];

  int tx = threadIdx.x;
  int block = blockIdx.x/nScales; 
  int scale = blockIdx.x - nScales*block;
  int minx = block*MINMAX_W;
  int maxx = min(minx + MINMAX_W, width);
  int xpos = minx + tx;
  int size = pitch*height;
  int ptr = size*scale + max(min(xpos-1, width-1), 0);

  if (tx==0)
    cnt = 0; 
  for (int y=0;y<MINMAX_H;y++) {

    int ypos = MINMAX_H*blockIdx.y + y;
    int yptr0 = ptr + max(0,ypos-1)*pitch;
    int yptr1 = ptr + ypos*pitch;
    int yptr2 = ptr + min(height-1,ypos+1)*pitch;
    {
      float d10 = d_Data0[yptr0];
      float d11 = d_Data0[yptr1];
      float d12 = d_Data0[yptr2];
      ymin1[tx] = fminf(fminf(d10, d11), d12);
      ymax1[tx] = fmaxf(fmaxf(d10, d11), d12);
    }
    {
      float d30 = d_Data0[yptr0 + 2*size];
      float d31 = d_Data0[yptr1 + 2*size];
      float d32 = d_Data0[yptr2 + 2*size]; 
      ymin3[tx] = fminf(fminf(d30, d31), d32);
      ymax3[tx] = fmaxf(fmaxf(d30, d31), d32);
    }
    float d20 = d_Data0[yptr0 + 1*size];
    float d21 = d_Data0[yptr1 + 1*size];
    float d22 = d_Data0[yptr2 + 1*size];
    ymin2[tx] = fminf(fminf(ymin1[tx], fminf(fminf(d20, d21), d22)), ymin3[tx]);
    ymax2[tx] = fmaxf(fmaxf(ymax1[tx], fmaxf(fmaxf(d20, d21), d22)), ymax3[tx]);
    __syncthreads(); 
    if (tx>0 && tx<MINMAX_W+1 && xpos<=maxx) {
      if (d21<d_Threshold[1]) {
	float minv = fminf(fminf(fminf(ymin2[tx-1], ymin2[tx+1]), ymin1[tx]), ymin3[tx]);
	minv = fminf(fminf(minv, d20), d22);
	if (d21<minv) { 
	  int pos = atomicInc(&cnt, 31);
	  points[3*pos+0] = xpos - 1;
	  points[3*pos+1] = ypos;
	  points[3*pos+2] = scale;
	}
      } 
      if (d21>d_Threshold[0]) {
	float maxv = fmaxf(fmaxf(fmaxf(ymax2[tx-1], ymax2[tx+1]), ymax1[tx]), ymax3[tx]);
	maxv = fmaxf(fmaxf(maxv, d20), d22);
	if (d21>maxv) { 
	  int pos = atomicInc(&cnt, 31);
	  points[3*pos+0] = xpos - 1;
	  points[3*pos+1] = ypos;
	  points[3*pos+2] = scale;
	}
      }
    }
    __syncthreads();
  }
  if (tx<cnt) {
    int xpos = points[3*tx+0];
    int ypos = points[3*tx+1];
    int scale = points[3*tx+2];
    int ptr = xpos + (ypos + (scale+1)*height)*pitch;
    float val = d_Data0[ptr];
    float *data1 = &d_Data0[ptr];
    float dxx = 2.0f*val - data1[-1] - data1[1];
    float dyy = 2.0f*val - data1[-pitch] - data1[pitch];
    float dxy = 0.25f*(data1[+pitch+1] + data1[-pitch-1] - data1[-pitch+1] - data1[+pitch-1]);
    float tra = dxx + dyy;
    float det = dxx*dyy - dxy*dxy;
    if (tra*tra<d_EdgeLimit*det) {
      float edge = __fdividef(tra*tra, det);
      float dx = 0.5f*(data1[1] - data1[-1]);
      float dy = 0.5f*(data1[pitch] - data1[-pitch]); 
      float *data0 = d_Data0 + ptr - height*pitch;
      float *data2 = d_Data0 + ptr + height*pitch;
      float ds = 0.5f*(data0[0] - data2[0]); 
      float dss = 2.0f*val - data2[0] - data0[0];
      float dxs = 0.25f*(data2[1] + data0[-1] - data0[1] - data2[-1]);
      float dys = 0.25f*(data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
      float idxx = dyy*dss - dys*dys;
      float idxy = dys*dxs - dxy*dss;   
      float idxs = dxy*dys - dyy*dxs;
      float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
      float idyy = dxx*dss - dxs*dxs;
      float idys = dxy*dxs - dxx*dys;
      float idss = dxx*dyy - dxy*dxy;
      float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
      float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
      float pds = idet*(idxs*dx + idys*dy + idss*ds);
      if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f) {
	pdx = __fdividef(dx, dxx);
	pdy = __fdividef(dy, dyy);
	pds = __fdividef(ds, dss);
      }
      float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
      int maxPts = d_MaxNumPoints;
      unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
      idx = (idx>=maxPts ? maxPts-1 : idx);
      d_Sift[idx].xpos = xpos + pdx;
      d_Sift[idx].ypos = ypos + pdy;
      d_Sift[idx].scale = d_Scales[scale] * exp2f(pds*d_Factor);
      d_Sift[idx].sharpness = val + dval;
      d_Sift[idx].edgeness = edge;
    }
  }
}

#define RADIUS 4

__global__ void LowPassRowMulti(float *d_Result, float *d_Data, int width, int pitch, int height)
{
  __shared__ float data[CONVROW_W + 2*RADIUS];
  const int tx = threadIdx.x;
  const int block = blockIdx.x/(NUM_SCALES+3); 
  const int scale = blockIdx.x - (NUM_SCALES+3)*block;
  const int xout = block*CONVROW_W + tx;
  const int loadPos = xout - RADIUS; 
  const int yptr = blockIdx.y*pitch;
  const int writePos = yptr + height*pitch*scale + xout;
  float *kernel = d_Kernel + scale*16;

  if (loadPos<0) 
    data[tx] = d_Data[yptr];
  else if (loadPos>=width) 
    data[tx] = d_Data[yptr + width-1];
  else
    data[tx] = d_Data[yptr + loadPos];
  __syncthreads();
  if (xout<width && tx<CONVROW_W) 
    d_Result[writePos] = 
      (data[tx+0] + data[tx+8])*kernel[0] + 
      (data[tx+1] + data[tx+7])*kernel[1] + 
      (data[tx+2] + data[tx+6])*kernel[2] + 
      (data[tx+3] + data[tx+5])*kernel[3] + 
      data[tx+4]*kernel[4]; 
  __syncthreads();
}

__global__ void LowPassColMulti(float *d_Result, float *d_Data, int width, int pitch, int height)
{
  __shared__ float data[CONVCOL_W*(CONVCOL_H + 2*RADIUS)];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int block = blockIdx.x/(NUM_SCALES+3); 
  const int scale = blockIdx.x - (NUM_SCALES+3)*block;
  const int miny = blockIdx.y*CONVCOL_H;
  const int maxy = min(miny + CONVCOL_H, height) - 1;
  const int totStart = miny - RADIUS;
  const int totEnd = maxy + RADIUS;
  const int colStart = block*CONVCOL_W + tx;
  const int colEnd = colStart + (height-1)*pitch;
  const int sStep = CONVCOL_W*CONVCOL_S;
  const int gStep = pitch*CONVCOL_S;
  float *kernel = d_Kernel + scale*16;
  const int size = pitch*height*scale;
  d_Result += size;
  d_Data += size;
 
  if (colStart<width) {
    float *sdata = data + ty*CONVCOL_W + tx;
    int gPos = colStart + (totStart + ty)*pitch;
    for (int y = totStart+ty;y<=totEnd;y+=blockDim.y){
      if (y<0) 
	sdata[0] = d_Data[colStart];
      else if (y>=height) 
	sdata[0] = d_Data[colEnd];
      else 
	sdata[0] = d_Data[gPos];  
      sdata += sStep;
      gPos += gStep;
    }
  }
  __syncthreads();
  if (colStart<width) {
    float *sdata = data + ty*CONVCOL_W + tx;
    int gPos = colStart + (miny + ty)*pitch;
    for (int y=miny+ty;y<=maxy;y+=blockDim.y) {
      d_Result[gPos] = 
	(sdata[0*CONVCOL_W] + sdata[8*CONVCOL_W])*kernel[0] + 
	(sdata[1*CONVCOL_W] + sdata[7*CONVCOL_W])*kernel[1] + 
	(sdata[2*CONVCOL_W] + sdata[6*CONVCOL_W])*kernel[2] + 
	(sdata[3*CONVCOL_W] + sdata[5*CONVCOL_W])*kernel[3] + 
	sdata[4*CONVCOL_W]*kernel[4]; 
      sdata += sStep;
      gPos += gStep;
    }
  }
}

 __global__ void LaplaceMulti(float *d_Result, float *d_Data, int width, int pitch, int height)
{
  __shared__ float data1[(LAPLACE_W + 2*RADIUS)*LAPLACE_S];
  __shared__ float data2[LAPLACE_W*LAPLACE_S];
  const int tx = threadIdx.x;
  const int xp = blockIdx.x*LAPLACE_W + tx;
  const int yp = blockIdx.y;
  const int scale = threadIdx.y;
  float *kernel = d_Kernel + scale*16;
  float *sdata1 = data1 + (LAPLACE_W + 2*RADIUS)*scale; 
  float x = xp-3.5;
  float y = yp+0.5;
  sdata1[tx] = kernel[4]*tex2D(tex, x, y) + 
    kernel[3]*(tex2D(tex, x, y-1.0) + tex2D(tex, x, y+1.0)) + 
    kernel[2]*(tex2D(tex, x, y-2.0) + tex2D(tex, x, y+2.0)) + 
    kernel[1]*(tex2D(tex, x, y-3.0) + tex2D(tex, x, y+3.0)) + 
    kernel[0]*(tex2D(tex, x, y-4.0) + tex2D(tex, x, y+4.0));
  __syncthreads();
  float *sdata2 = data2 + LAPLACE_W*scale; 
  if (tx<LAPLACE_W) {
    sdata2[tx] = kernel[4]*sdata1[tx+4] + 
      kernel[3]*(sdata1[tx+3] + sdata1[tx+5]) + 
      kernel[2]*(sdata1[tx+2] + sdata1[tx+6]) + 
      kernel[1]*(sdata1[tx+1] + sdata1[tx+7]) + 
      kernel[0]*(sdata1[tx+0] + sdata1[tx+8]);
  }
  __syncthreads(); 
  if (tx<LAPLACE_W && scale<LAPLACE_S-1 && xp<width) 
    d_Result[scale*height*pitch + yp*pitch + xp] = sdata2[tx] - sdata2[tx+LAPLACE_W];
}


