#include "cudaSiftH_old.h"

double MultiplyAdd(CudaImage *res, CudaImage *data, float constA, float constB)
{
  int w = res->width;
  int p = res->pitch;
  int h = res->height;
  TimerGPU timer(0);
  if (res->d_data==NULL || data->d_data==NULL) {
    printf("MultiplyAdd: missing data\n");
    return 0.0;
  }
  safeCall(cudaMemcpyToSymbol(d_ConstantA, &constA, sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_ConstantB, &constB, sizeof(float)));

  dim3 blocks(iDivUp(w, 16), iDivUp(h, 16));
  dim3 threads(16, 16);
  MultiplyAdd<<<blocks, threads>>>(res->d_data, data->d_data, w, p, h);
  checkMsg("MultiplyAdd() execution failed\n");
  safeCall(cudaThreadSynchronize());

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("MultiplyAdd time =            %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double FindMinMax(CudaImage *img, float &minval, float &maxval)
{
  int w = img->width;
  int p = img->pitch;
  int h = img->height;
  TimerGPU timer(0);

  int dx = iDivUp(w, 128);
  int dy = iDivUp(h, 16);
  int sz = 2*dx*dy*sizeof(float);
  float *d_minmax = NULL; 
  float *h_minmax = (float *)malloc(sz); 
  for (int i=0;i<2*dx*dy;i+=2) {
    h_minmax[i+0] = 1e6;
    h_minmax[i+1] = -1e6; 
  }
  safeCall(cudaMalloc((void **)&d_minmax, sz));
  safeCall(cudaMemcpy(d_minmax, h_minmax, sz, cudaMemcpyHostToDevice));

  dim3 blocks(dx, dy); 
  dim3 threads(128, 1);
  FindMinMax<<<blocks, threads>>>(d_minmax, img->d_data, w, p, h);
  checkMsg("FindMinMax() execution failed\n");
  safeCall(cudaThreadSynchronize());

  safeCall(cudaMemcpy(h_minmax, d_minmax, sz, cudaMemcpyDeviceToHost));
  minval = 1e6;
  maxval = -1e6;
  for (int i=0;i<2*dx*dy;i+=2) {
    if (h_minmax[i+0]<minval) 
      minval = h_minmax[i];
    if (h_minmax[i+1]>maxval)  
      maxval = h_minmax[i+1];
  }

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("FindMinMax time =             %.2f msec\n", gpuTime);
#endif
  safeCall(cudaFree(d_minmax));
  free(h_minmax);
  return gpuTime; 
}
 
double Find3DMinMax(CudaArray *minmax, CudaImage *data1, CudaImage *data2, CudaImage *data3, float thresh, int maxPts)
{
  int *h_res = (int *)minmax->h_data;
  int *d_res = (int *)minmax->d_data;
  if (data1->d_data==NULL || data2->d_data==NULL || data3->d_data==NULL || h_res==NULL || d_res==NULL) {
    std::cout << "Find3DMinMax: missing data " << std::endl;
    printf("Find3DMinMax: missing data %08x %08x %08x %08x %08x\n", data1->d_data, data2->d_data, data3->d_data, h_res, d_res);
    return 0.0;
  }
  int w = data1->width;
  int p = data1->pitch;
  int h = data1->height;
  TimerGPU timer(0);
  float threshs[2] = { thresh, -thresh };
  safeCall(cudaMemcpyToSymbol(d_Threshold, &threshs, 2*sizeof(float)));

  dim3 blocks(iDivUp(w, MINMAX_SIZE), iDivUp(h, 32));
  dim3 threads(WARP_SIZE + MINMAX_SIZE + 1);
  Find3DMinMax<<<blocks, threads>>>(d_res, data1->d_data, data2->d_data, data3->d_data, w, p, h); 
  checkMsg("Find3DMinMax() execution failed\n");
  safeCall(cudaThreadSynchronize());
  safeCall(cudaMemcpy(h_res, d_res, sizeof(float)*minmax->pitch*minmax->height, cudaMemcpyDeviceToHost));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Find3DMinMax time =           %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double UnpackPointers(CudaArray *minmax, int maxPts, int *ptrs, int &numPts)
{
  unsigned int *minmax_data = (unsigned int *)minmax->h_data;
  if (minmax_data==NULL || ptrs==NULL) {
    printf("UnpackPointers: missing data %08x %08x\n", minmax_data, ptrs);
    return 0.0;
  }
  int p = minmax->pitch;
  int w = minmax->width;
  int h = 32*minmax->height;
  TimerGPU timer(0);
  int num = 0;
  for (int y=0;y<h/32;y++) {
    for (int x=0;x<w;x++) {
      unsigned int val = minmax_data[y*p + x];
      if (val) {
	//printf("%d %d %08x\n", x, y, val);
	for (int k=0;k<32;k++) {
	  if (val&0x1 && num<maxPts)
	    ptrs[num++] = (y*32+k)*p + x;
	  val >>= 1;
	}
      }
    }
  }
  numPts = num;
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("UnpackPointers time =         %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double ComputePositions(CudaImage *data1, CudaImage *data2, CudaImage *data3, int *h_ptrs, CudaArray *sift, int numPts, int maxPts, float scale, float factor)
{
  int p = data1->pitch;
  int h = data1->height;
  TimerGPU timer(0);
  int *d_ptrs = NULL;
  float *d_sift = sift->d_data;
  safeCall(cudaMalloc((void **)&d_ptrs, sizeof(int)*numPts));
  safeCall(cudaMemcpy(d_ptrs, h_ptrs, sizeof(int)*numPts, cudaMemcpyHostToDevice));
  safeCall(cudaMemcpyToSymbol(d_Scale, &scale, sizeof(float)));
  safeCall(cudaMemcpyToSymbol(d_Factor, &factor, sizeof(float)));

  dim3 blocks(iDivUp(numPts, POSBLK_SIZE));
  dim3 threads(POSBLK_SIZE);
  ComputePositions<<<blocks, threads>>>(data1->d_data, data2->d_data, data3->d_data, d_ptrs, d_sift, numPts, maxPts, p, h);
  checkMsg("ComputePositions() execution failed\n");
  safeCall(cudaThreadSynchronize());

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("ComputePositions time =       %.2f msec\n", gpuTime);
#endif
  safeCall(cudaFree(d_ptrs));
  return gpuTime;
}

double RemoveEdgePoints(CudaArray *sift, int &initNumPts, int maxPts, float edgeLimit) 
{
  TimerGPU timer(0);
  int numPts = initNumPts;
  float *d_sift = sift->d_data;
  int bw = sizeof(float)*numPts;
  float *h_sift = (float *)malloc(5*bw);
  safeCall(cudaMemcpy2D(h_sift, bw, d_sift, sizeof(float)*maxPts,  bw, 5, cudaMemcpyDeviceToHost));
  int num = 0;
  for (int i=0;i<numPts;i++) 
    if (h_sift[4*numPts+i]<edgeLimit) {
      for (int j=0;j<5;j++) 
	h_sift[j*numPts+num] = h_sift[j*numPts+i];
      num ++;
    }
  safeCall(cudaMemcpy2D(d_sift, sizeof(float)*maxPts, h_sift, bw, bw, 5, cudaMemcpyHostToDevice));
  free(h_sift);
  initNumPts = num;
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("RemoveEdgePoints time =       %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double ComputeOrientations(CudaImage *img, int *h_ptrs, CudaImage *sift, int numPts, int maxPts)
{
  int p = img->pitch;
  int h = img->height;
  TimerGPU timer(0);
  int *d_ptrs = NULL;
  float *d_orient = &sift->d_data[5*maxPts];
  safeCall(cudaMalloc((void **)&d_ptrs, sizeof(int)*numPts));
  safeCall(cudaMemcpy(d_ptrs, h_ptrs, sizeof(int)*numPts, cudaMemcpyHostToDevice));

  dim3 blocks(numPts);
  dim3 threads(32);
  ComputeOrientations<<<blocks, threads>>>(img->d_data, d_ptrs, d_orient, maxPts, p, h);
  checkMsg("ComputeOrientations() execution failed\n");
  safeCall(cudaThreadSynchronize());

  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("ComputeOrientations time =    %.2f msec\n", gpuTime);
#endif
  safeCall(cudaFree(d_ptrs));
  return gpuTime;
}

