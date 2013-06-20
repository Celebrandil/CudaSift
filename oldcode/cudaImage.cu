//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <cstdio>

#include "cudautils.h"
#include "cudaImage.h"

#undef VERBOSE

int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

int iDivDown(int a, int b) {
  return a / b;
}

int iAlignUp(int a, int b) {
  return (a % b != 0) ?  (a - a % b + b) : a;
}

int iAlignDown(int a, int b) {
  return a - a % b;
}

double AllocCudaImage(CudaImage *img, int w, int h, int p, bool host, bool dev)
{
  TimerGPU timer(0);
  int sz = sizeof(float)*p*h;
  img->width = w;
  img->height = h;
  img->pitch = p;
  img->h_data = NULL;
  if (host) {
    //printf("Allocating host data...\n");
    img->h_data = (float *)malloc(sz);
    //safeCall(cudaMallocHost((void **)&img->h_data, sz));
  }
  img->d_data = NULL;
  if (dev) {
    //printf("Allocating device data...\n");
    safeCall(cudaMalloc((void **)&img->d_data, sz));
    if (img->d_data==NULL) 
      printf("Failed to allocate device data\n");
  }
  img->t_data = NULL;
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("AllocCudaImage time =         %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double FreeCudaImage(CudaImage *img)
{
  TimerGPU timer(0);
  if (img->d_data!=NULL) {
    //printf("Freeing device data...\n");
    safeCall(cudaFree(img->d_data));
  }
  img->d_data = NULL;
  if (img->h_data!=NULL) {
    //printf("Freeing host data...\n");
    free(img->h_data);
    //safeCall(cudaFreeHost(img->h_data));
  }
  img->h_data = NULL;
  if (img->t_data!=NULL) {
    //printf("Freeing texture data...\n");
    safeCall(cudaFreeArray((cudaArray *)img->t_data));
  }
  img->t_data = NULL;
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("FreeCudaImage time =          %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double Download(CudaImage *img)
{
  TimerGPU timer(0);
  if (img->d_data!=NULL && img->h_data!=NULL) 
    safeCall(cudaMemcpy(img->d_data, img->h_data, 
      sizeof(float)*img->pitch*img->height, cudaMemcpyHostToDevice));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Download time =               %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double Readback(CudaImage *img, int w, int h)
{
  TimerGPU timer(0);
  int p = sizeof(float)*img->pitch;
  w = sizeof(float)*(w<0 ? img->width : w);
  h = (h<0 ? img->height : h); 
  safeCall(cudaMemcpy2D(img->h_data, p, img->d_data, p, 
    w, h, cudaMemcpyDeviceToHost));
  //safeCall(cudaMemcpy(img->h_data, img->d_data, 
  //  sizeof(float)*img->pitch*img->height, cudaMemcpyDeviceToHost));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Readback time =               %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double InitTexture(CudaImage *img)
{
  TimerGPU timer(0);
  cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>(); 
  safeCall(cudaMallocArray((cudaArray **)&img->t_data, &t_desc, 
    img->pitch, img->height)); 
  //printf("InitTexture(%d, %d)\n", img->pitch, img->height); 
  if (img->t_data==NULL)
    printf("Failed to allocated texture data\n");
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("InitTexture time =            %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}
 
double CopyToTexture(CudaImage *src, CudaImage *dst, bool host)
{
  if (dst->t_data==NULL) {
    printf("Error CopyToTexture: No texture data\n");
    return 0.0;
  }
  if ((!host || src->h_data==NULL) && (host || src->d_data==NULL)) {
    printf("Error CopyToTexture: No source data\n");
    return 0.0;
  }
  TimerGPU timer(0);
  if (host)
    safeCall(cudaMemcpyToArray((cudaArray *)dst->t_data, 0, 0, 
      src->h_data, sizeof(float)*src->pitch*dst->height, 
      cudaMemcpyHostToDevice));
  else
    safeCall(cudaMemcpyToArray((cudaArray *)dst->t_data, 0, 0, 
      src->d_data, sizeof(float)*src->pitch*dst->height, 
      cudaMemcpyDeviceToDevice));
  safeCall(cudaThreadSynchronize());
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("CopyToTexture time =          %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}
