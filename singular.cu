#include <cstdio>

#include "cudautils.h"
#include "cudaImage.h"

#define SINGULAR_BATCH 32

template<int n>
__device__ float Householder(float *a, float *v, int row)
{
  float sigma = 0.0f;
  float beta = 0.0f;
  for (int i=row+1;i<n;i++)
    sigma += a[i] * a[i];
  for (int i=0;i<row;i++)
    v[i] = 0.0;
  v[row] = 1.0;
  for (int i=row+1;i<n;i++)
    v[i] = a[i]; 
  if (sigma!=0.0) {
    float x1 = a[row];
    float v1 = v[row];
    float eig = sqrt(x1*x1 + sigma);
    if (x1<=0.0)
      v1 = x1 - eig;
    else
      v1 = -sigma / (x1 + eig);
    beta = 2*v1*v1 / (sigma + v1*v1);
    for (int i=row+1;i<n;i++) 
      v[i] /= v1;
  }
  return beta;
}

template <int n>
__device__ void SingularValues(float *A, float *a)
{
#define eps 1e-4f
  // Householder bidiagonalization A = U*B*V^T   5.4.2
  float vA[n];
  float v[n];
  for (int j=0;j<n;j++) {
    for (int k=j;k<n;k++) 
      a[k] = A[k*n+j];
    float betaU = Householder<n>(a, v, j);
    for (int k=j;k<n;k++) {
      float sum = 0.0f;
      for (int l=j;l<n;l++) 
	sum += v[l] * A[l*n+k];
      vA[k] = sum;
    }
    for (int k=j;k<n;k++) 
      for (int l=j;l<n;l++) 
	A[l*n+k] -= betaU*v[l]*vA[k];
    if (j<n-1) {
      for (int k=j+1;k<n;k++) 
	a[k] = A[j*n+k];
      float betaV = Householder<n>(a, v, j+1);
      for (int k=j;k<n;k++) {
	float sum = 0.0f;
	for (int l=j+1;l<n;l++) 
	  sum += A[k*n+l] * v[l];
	vA[k] = sum;
      }
      for (int k=j;k<n;k++) 
	for (int l=j+1;l<n;l++) 
	  A[k*n+l] -= betaV*vA[k]*v[l];
    }
  }
  // Golub-Kahan SVD Step B = U*D*V^T   8.6.2
  for (int i=0;i<n-1;i++) {
    a[i] = A[i*n+i];
    v[i] = A[i*n+i+1];
  }
  a[n-1] = A[n*n-1];
  v[n-1] = 0.0;
  int q = n-1;
  int cnt = 0;
  while (q>0 && cnt<10000) {
    for (int i=0;i<n-1;i++) 
      if (fabs(v[i])<eps*(fabs(a[i]) + fabs(a[i+1])))
	v[i] = 0.0f;
    q = n - 1;
    while (q>0 && fabs(v[q-1])<eps) 
      q--;
    if (q>0) {
      int p = q;
      while (p>0 && fabs(v[p-1])>eps) 
	p--;
      bool dogivens = true;
      for (int i=p;i<q;i++)
	if (a[i]*a[i]<eps*eps) {
	  v[i] = 0.0f;
	  dogivens = false;
	}
      if (dogivens) {
	float oldc = 1.0f;
	float olds = 0.0f;
	float y = a[p];
	float z = v[p];
	for (int k=p;k<q;k++) {
	  float sz = sqrt(y*y + z*z);
	  float c = y / sz;
	  float s = -z / sz;
	  if (k>p) 
	    v[k-1] = olds*sz;
	  y = oldc*sz;
	  z = a[k+1]*s;
	  float h = a[k+1]*c;
	  sz = sqrt(y*y + z*z);
	  c = y / sz;
	  s = -z / sz;
	  a[k] = sz;
	  y = h;
	  if (k<q-1)
	    z = v[k+1];
	  oldc = c;
	  olds = s;
	}
	v[q-1] = y*olds;
	a[q] = y*oldc;
      }
    }
    cnt ++;
  }
  for (int i=0;i<n;i++)
    a[i] = (a[i]<0.0f ? -a[i] : a[i]);
}

// 362

template <int n, int k>
__global__ void ComputeSingular(float *imgData, float *svdData, int svdWid)
{
#define SINGULAR_WIDTH ((SINGULAR_BATCH-1)*k + n)
  __shared__ float buffer[SINGULAR_WIDTH];
  float A[n*n];
  float a[n];
  const int tx = threadIdx.x;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  int imgWid = k*svdWid;
  int readPos = __mul24(by, imgWid) + __mul24(bx*k, SINGULAR_BATCH);
  for (int yp=0;yp<n;yp++) {
    float *imgd = &imgData[readPos+yp*imgWid];
    for (int xp=tx;xp<SINGULAR_WIDTH;xp+=SINGULAR_BATCH)
      buffer[xp] = imgd[xp];
    __syncthreads();
    for (int xp=0;xp<n;xp++)
      A[yp*n+xp] = buffer[tx*k+xp];
    __syncthreads();
  }
  SingularValues<n>(A, a);
  __syncthreads();
  int writePos = __mul24(by, svdWid) + __mul24(bx, SINGULAR_BATCH);
  for (int i=0;i<n-1;i++) {
    for (int j=i+1;j<n;j++) {
      if (a[i]<a[j]) {
	float t = a[i];
	a[i] = a[j];
	a[j] = t;
      }
    }
  }
  float sum = 1e-10f;
  for (int i=0;i<5*n/8;i++) 
    sum += a[i];
  float tot = sum;
  for (int i=5*n/8;i<n;i++) 
    tot += a[i];
  svdData[writePos+tx] = 1.0f - sum/tot;
}

double ComputeSingular(CudaImage *img, CudaImage *svd)
{
  int sw = svd->width;
  int sh = svd->height;
  TimerGPU timer(0);
  if (img->d_data==NULL || svd->d_data==NULL) {
    printf("ComputeSingular: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(sw, SINGULAR_BATCH), sh);
  dim3 threads(SINGULAR_BATCH);
  ComputeSingular<8,1><<<blocks,threads>>>(img->d_data, svd->d_data, sw);
  checkMsg("ComputeSingular() execution failed\n");
  safeCall(cudaThreadSynchronize());
 
  double gpuTime = timer.read();
  //#ifdef VERBOSE
  printf("ComputeSingular time =        %.2f ms\n", gpuTime);
  //#endif
  return gpuTime;
}
