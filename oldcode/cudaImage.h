//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

typedef struct {
  int width, height;
  int pitch;
  float *h_data;
  float *d_data;
  void *t_data; //cudaArray *t_data;
} CudaImage;

typedef CudaImage CudaArray;

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);

void StartTimer(unsigned int *hTimer);
double StopTimer(unsigned int hTimer);
double AllocCudaImage(CudaImage *img, int w, int h, int p, 
  bool host, bool dev);
double FreeCudaImage(CudaImage *img);
double Download(CudaImage *img);
double Readback(CudaImage *img, int w = -1, int h = -1);
double InitTexture(CudaImage *img);
double CopyToTexture(CudaImage *src, CudaImage *dst, bool host);

#endif // CUDAIMAGE_H
