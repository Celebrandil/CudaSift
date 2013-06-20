//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ nada.kth.se                       //
//********************************************************//  

#include <iostream>  
#include <cmath>

#include <pyra/matrix.h>
#include <pyra/vector.h>
#include <pyra/tpimage.h>
#include <pyra/tpimageutil.h>

#include "cudaImage.h"
#include "cudaSift.h"


///////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
///////////////////////////////////////////////////////////////////////////////
extern "C" void ConvRowCPU(float *h_Result, float *h_Data, float *h_Kernel, int w, int h, int kernelR);   
extern "C" void ConvColCPU(float *h_Result, float *h_Data, float *h_Kernel, int w, int h, int kernelR);
extern "C" double Find3DMinMaxCPU(CudaImage *res, CudaImage *data1, CudaImage *data2, CudaImage *data3);
int ImproveHomography(SiftData *data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
double ComputeSingular(CudaImage *img, CudaImage *svd);

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) 
{     
  CudaImage img1, img2;
  Image<float> limg(1280, 960);
  Image<float> rimg(1280, 960);

  limg.Load("data/left.pgm");
  rimg.Load("data/righ.pgm");
  ReScale(limg, 1.0/256.0f);
  ReScale(rimg, 1.0/256.0f);
  unsigned int w = limg.GetWidth();
  unsigned int h = limg.GetHeight();

  std::cout << "Image size = (" << w << "," << h << ")" << std::endl;
      
  InitCuda();
  std::cout << "Initializing data..." << std::endl;
  AllocCudaImage(&img1, w, h, w, false, true);
  img1.h_data = limg.GetData();
  AllocCudaImage(&img2, w, h, w, false, true);
  img2.h_data = rimg.GetData(); 
  Download(&img1);
  Download(&img2);

  SiftData siftData1, siftData2;
  for (int i=0;i<10;i++) {
    InitSiftData(&siftData1, 128, true, true); 
    ExtractSift(&siftData1, &img1, 3, 3, 0.3f, 0.03);
    FreeSiftData(&siftData1);
  }
  InitSiftData(&siftData1, 128, true, true); 
  ExtractSift(&siftData1, &img1, 3, 3, 0.3f, 0.03);
  InitSiftData(&siftData2, 128, true, true);
  ExtractSift(&siftData2, &img2, 3, 3, 0.3f, 0.03);
  std::cout << "Number of original features: " <<  siftData1.numPts << " " 
	    << siftData2.numPts << std::endl;
  MatchSiftData(&siftData1, &siftData2);
  float homography[9];
  int numMatches;
  FindHomography(&siftData1, homography, &numMatches, 1000, 0.85f, 0.95f, 5.0);
  int numFit = ImproveHomography(&siftData1, homography, 3, 0.80f, 0.95f, 3.0);
  std::cout << "Number of matching features: " << numFit << " " << numMatches << std::endl;
  //PrintSiftData(&siftData1);
#if 1
  int numPts = siftData1.numPts;
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
  float *h_img = img1.h_data; 
  for (int j=0;j<numPts;j++) { 
    int k = sift1[j].match;
    if (sift1[j].match_error<5.0) {
      float dx = sift2[k].xpos - sift1[j].xpos;
      float dy = sift2[k].ypos - sift1[j].ypos;
#if 0
      std::cout << "score = " << sift1[j].score << ", ambiguity = " << sift1[j].ambiguity << ", match = " << k << "  ";
      std::cout << "error = " << sift1[j].match_error << " ";
      std::cout << "pos1 = (" << (int)sift1[j].xpos << "," << (int)sift1[j].ypos << ")" << "  ";
      std::cout << "delta = (" << (int)dx << "," << (int)dy << ")" << std::endl;
#endif
      int len = (int)(fabs(dx)>fabs(dy) ? fabs(dx) : fabs(dy));
      for (int l=0;l<len;l++) {
	int x = (int)(sift1[j].xpos + dx*l/len);
	int y = (int)(sift1[j].ypos + dy*l/len);
	h_img[y*w+x] = 1.0f;
      }	
    }
    int p = (int)(sift1[j].ypos+0.5)*w + (int)(sift1[j].xpos+0.5);
    p += (w+1);
    for (int k=0;k<(int)(1.41*sift1[j].scale);k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 0.0f;
    p -= (w+1);
    for (int k=0;k<(int)(1.41*sift1[j].scale);k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 1.0f;
  }
#endif
  FreeSiftData(&siftData1);
  FreeSiftData(&siftData2);
  limg.Store("data/limg_pts.pgm", true, false);

  img1.h_data = NULL;
  FreeCudaImage(&img1);
  img2.h_data = NULL;
  FreeCudaImage(&img2);
}

bool SolveSymmetricSystem(pyra::Matrix &A, pyra::Vector &b, pyra::Vector &x)
{
  int n = A.q_rows();
  if (n!=A.q_cols() || n!=b.q_size() || n!=x.q_size()) {
    std::cout << "Error: SolveSymmetricSystem() incorrect sizes" << std::endl; 
    return false;
  }
  // Cholesky factorization A = C*C^T   4.2.1
  for (int j=0;j<n;j++) {    
    if (j>0) {
      for (int i=j;i<n;i++) {        
	for (int k=0;k<j;k++)       
	  A(i,j) -= A(i,k)*A(j,k);
      }
    }
    if (A(j,j)!=0.0) {
      double sqrtD = sqrt(A(j,j));
      for (int i=j;i<n;i++)
	A(i,j) = A(i,j) / sqrtD;  
    }
  }
  // Solve C*y = b                      5.3.1
  for (int i=0;i<n;i++) {
    double sum = b(i);
    for (int j=0;j<i;j++)
      sum -= A(i,j) * x(j);
    if (A(i,i)==0.0)
      x(i) = sum;
    else
      x(i) = sum / A(i,i);
  }
  // Solve C^t*x = y                    5.3.1
  for (int i=n-1;i>=0;i--) {
    double sum = x(i);
    for (int j=n-1;j>i;j--)
      sum -= A(j,i) * x(j);
    if (A(i,i)==0.0)
      x(i) = sum;
    else 
      x(i) = sum / A(i,i);
  }
  return true;
}

double Householder(pyra::Vector &a, pyra::Vector &v, int row)
{
  int m = a.q_size();
  double sigma = 0.0;
  double beta = 0.0;
  for (int i=row+1;i<m;i++)
    sigma += a(i) * a(i);
  for (int i=0;i<row;i++)
    v(i) = 0.0;
  v(row) = 1.0;
  for (int i=row+1;i<m;i++)
    v(i) = a(i);
  if (sigma!=0.0) {
    double x1 = a(row);
    double v1 = v(row);
    double eig = sqrt(x1*x1 + sigma);
    if (x1<=0.0)
      v1 = x1 - eig;
    else
      v1 = -sigma / (x1 + eig);
    beta = 2*v1*v1 / (sigma + v1*v1);
    for (int i=row+1;i<m;i++) 
      v(i) /= v1;
  }
  return beta;
}

bool SolveLinearSystem(pyra::Matrix &A, pyra::Vector &b, pyra::Vector &x)
{
  int m = A.q_rows();
  int n = A.q_cols();
  if (m<n || m!=b.q_size() || n!=x.q_size()) {
    std::cout << "Error: SolveLinearSystem() incorrect sizes" << std::endl; 
    return false;
  }
  // QR factorization A = Q*R           5.2.1
  pyra::Vector vA(n);
  pyra::Vector v(m);
  pyra::Vector beta(n);
  pyra::Vector a(m);
  for (int j=0;j<n;j++) {
    for (int k=j;k<m;k++) 
      a(k) = A(k,j);
    beta(j) = Householder(a, v, j);
    for (int k=j;k<n;k++) {
      double sum = 0.0;
      for (int l=j;l<m;l++) 
	sum += v(l) * A(l,k);
      vA(k) = sum;
    }
    for (int k=j;k<n;k++) 
      for (int l=j;l<m;l++) 
	A(l,k) -= beta(j)*v(l)*vA(k);
    if (j<m) {
      for (int k=j+1;k<m;k++)
	A(k,j) = v(k);
    }
  }
  // Compute c = Q^t*b                  5.3.2
  for (int j=0;j<n;j++) {
    v(j) = 1.0;
    for (int k=j+1;k<m;k++)
      v(k) = A(k,j);
    double vd = 0.0;
    for (int k=j;k<m;k++) {
      vd += v(k)*b(k);
    }
    for (int k=j;k<m;k++) 
      b(k) -= beta(j)*v(k)*vd;
  }
  // Solve R*x = c                      5.3.2
  for (int i=n-1;i>=0;i--) {
    double sum = b(i);
    for (int j=n-1;j>i;j--)
      sum -= A(i,j) * x(j);
    if (A(i,i)==0.0)
      x(i) = sum;
    else 
      x(i) = sum / A(i,i);
  }
  return true;
}

bool ComputeSingularValues(pyra::Matrix &C)
{
  int m = C.q_rows();
  int n = C.q_cols();
  if (m<n) {
    std::cout << "Error: ComputeSingularValues() incorrect sizes" <<std::endl; 
    return false;
  }
  // Householder bidiagonalization A = U*B*V^T   5.4.2
  pyra::Matrix A = C;
  pyra::Vector vA(m);
  pyra::Vector v(m);
  pyra::Vector a(m);
  for (int j=0;j<n;j++) {
    for (int k=j;k<m;k++) 
      a(k) = A(k,j);
    double betaU = Householder(a, v, j);
    for (int k=j;k<n;k++) {
      double sum = 0.0;
      for (int l=j;l<m;l++) 
	sum += v(l) * A(l,k);
      vA(k) = sum;
    }
    for (int k=j;k<n;k++) 
      for (int l=j;l<m;l++) 
	A(l,k) -= betaU*v(l)*vA(k);
    if (j<n-1) {
      for (int k=j+1;k<m;k++) 
	a(k) = A(j,k);
      double betaV = Householder(a, v, j+1);
      for (int k=j;k<m;k++) {
	double sum = 0.0;
	for (int l=j+1;l<n;l++) 
	  sum += A(k,l) * v(l);
	vA(k) = sum;
      }
      for (int k=j;k<m;k++) 
	for (int l=j+1;l<n;l++) 
	  A(k,l) -= betaV*vA(k)*v(l);
    }
  }
  // Golub-Kahan SVD Step B = U*D*V^T   8.6.2
  for (int i=0;i<n-1;i++) {
    a(i) = A(i,i);
    v(i) = A(i,i+1);
  }
  a(n-1) = A(n-1,n-1);
  v(n-1) = 0.0;
  const double eps = 1e-10;
  int q = n-1;
  while (q>0) {
    for (int i=0;i<n-1;i++) 
      if (fabs(v(i))<eps*(fabs(a(i)) + fabs(a(i+1))))
	v(i) = 0.0;
    q = n - 1;
    while (q>0 && fabs(v(q-1))<eps) q--;
    if (q>0) {
      int p = q;
      while (p>0 && fabs(v(p-1))>eps) p--;
      bool dogivens = true;
      for (int i=p;i<q;i++)
	if (a(i)*a(i)<eps*eps) {
	  v(i) = 0.0;
	  dogivens = false;
	}
      if (dogivens) {
	double oldc = 1.0;
	double olds = 0.0;
	double y = a(p);
	double z = v(p);
	for (int k=p;k<q;k++) {
	  double sz = sqrt(y*y + z*z);
	  double c = y / sz;
	  double s = -z / sz;
	  if (k>p) 
	    v(k-1) = olds*sz;
	  y = oldc*sz;
	  z = a(k+1)*s;
	  double h = a(k+1)*c;
	  sz = sqrt(y*y + z*z);
	  c = y / sz;
	  s = -z / sz;
	  a(k) = sz;
	  y = h;
	  if (k<q-1)
	    z = v(k+1);
	  oldc = c;
	  olds = s;
	}
	v(q-1) = y*olds;
	a(q) = y*oldc;
      }
    }
  }
  for (int i=0;i<n;i++)
    a(i) = (a(i)<0.0 ? -a(i) : a(i));
  std::cout << "svd: "; a.print();
  return true;
}

int ImproveHomography(SiftData *data, float *homography, int numLoops, 
  float minScore, float maxAmbiguity, float thresh)
{
  if (data->h_data==NULL)
    return 0;
  float limit = thresh*thresh;
  int numPts = data->numPts;
  SiftPoint *mpts = data->h_data;
  pyra::Matrix M(8, 8);
  pyra::Vector A(8), X(8), Y(8);
  for (int i=0;i<8;i++)
    A(i) = homography[i] / homography[8];
  A.print();
  for (int loop=0;loop<numLoops;loop++) {
    M = 0;
    X = 0;
    for (int i=0;i<numPts;i++) {
      SiftPoint &pt = mpts[i];
      if (pt.score<minScore || pt.ambiguity>maxAmbiguity)
	continue;
      float den = A(6)*pt.xpos + A(7)*pt.ypos + 1.0f;
      float dx = (A(0)*pt.xpos + A(1)*pt.ypos + A(2)) / den - pt.match_xpos;
      float dy = (A(3)*pt.xpos + A(4)*pt.ypos + A(5)) / den - pt.match_ypos;
      float err = dx*dx + dy*dy;
      float wei = limit / (err + limit);
      Y(0) = pt.xpos;
      Y(1) = pt.ypos;
      Y(2) = 1.0;
      Y(3) = Y(4) = Y(5) = 0.0;
      Y(6) = - pt.xpos * pt.match_xpos;
      Y(7) = - pt.ypos * pt.match_xpos;
      for (int c=0;c<8;c++) 
        for (int r=0;r<8;r++) 
          M(r,c) += (Y(c) * Y(r) * wei);
      X += (Y * pt.match_xpos * wei);
      Y(0) = Y(1) = Y(2) = 0.0;
      Y(3) = pt.xpos;
      Y(4) = pt.ypos; 
      Y(5) = 1.0;
      Y(6) = - pt.xpos * pt.match_ypos;
      Y(7) = - pt.ypos * pt.match_ypos;
      for (int c=0;c<8;c++) 
        for (int r=0;r<8;r++) 
          M(r,c) += (Y(c) * Y(r) * wei);
      X += (Y * pt.match_ypos * wei);
    }
    ComputeSingularValues(M);
    SolveLinearSystem(M, X, A);
    //SolveSymmetricSystem(M, X, A);
    //A = M.invert() * X;
    A.print();
  }
  int numfit = 0;
  for (int i=0;i<numPts;i++) {
    SiftPoint &pt = mpts[i];
    float den = A(6)*pt.xpos + A(7)*pt.ypos + 1.0;
    float dx = (A(0)*pt.xpos + A(1)*pt.ypos + A(2)) / den - pt.match_xpos;
    float dy = (A(3)*pt.xpos + A(4)*pt.ypos + A(5)) / den - pt.match_ypos;
    float err = dx*dx + dy*dy;
    if (err<limit) 
      numfit++;
    pt.match_error = sqrt(err);
  }
  A.print();
  for (int i=0;i<8;i++) 
    homography[i] = A(i);
  homography[8] = 1.0f;
  return numfit;
}
