//********************************************************//
// OpenCV- and VLFeat-compatible SIFT extraction          //
//              Apple Metal backend                       //
//********************************************************//

// Constants and types shared by the Metal host code of ExtractSiftCompat (cudaSiftCompatMetal.mm)
// and its kernels (cudaSiftCompat.metal, compiled at run time with this header prepended). They
// follow cudaSiftCompat.cu, with plain int/float members so that the layout is the same on both
// sides: Apple GPUs have no double precision, so the values that the CUDA code keeps in double
// are floats here, and the scale space is one buffer with the offsets of its levels.

#ifndef CUDASIFTCOMPATSHARED_H
#define CUDASIFTCOMPATSHARED_H

#define COMPAT_MAX_OCTAVES 14
#define COMPAT_MAX_LAYERS  6     // nOctaveLayers
#define COMPAT_MAX_LEVELS  (2*COMPAT_MAX_LAYERS + 3)   // with extendFirstOctave
#define COMPAT_MAX_RADIUS  64    // of the Gaussian kernels

// OpenCV's constants (sift.simd.hpp)
#define SIFT_DESCR_WIDTH      4
#define SIFT_DESCR_HIST_BINS  8
#define SIFT_INIT_SIGMA       0.5f
#define SIFT_IMG_BORDER       5
#define SIFT_MAX_INTERP_STEPS 5
#define SIFT_ORI_HIST_BINS    36
#define SIFT_ORI_SIG_FCTR     1.5f
#define SIFT_ORI_RADIUS       4.5f
#define SIFT_ORI_PEAK_RATIO   0.8f
#define SIFT_DESCR_SCL_FCTR   3.f
#define SIFT_DESCR_MAG_THR    0.2f
#define SIFT_INT_DESCR_FCTR   512.f

// VLFeat's constants (vl/sift.c)
#define VL_EXPN_SZ  256
#define VL_EXPN_MAX 25.0f
#define VL_EPSILON_F 1.19209290E-07F
#define VL_MAX_W 128                 // largest window radius of the orientations and descriptors
#define VL_MAX_ORIENTATION_PEAKS 4   // peaks scanned per orientation histogram
#define VL_MAX_REFINE_STEPS 5        // of the refinement of the extrema

// Kernel geometry, used by the kernels and for the dispatch sizes of the host code
#define COMPAT_BLOCK_W     32        // InputCompat, DownsampleCompat, FindExtremaCompat
#define COMPAT_BLOCK_H     8
#define ROW_W 128                    // BlurRowsTiled
#define ROW_H 4
#define COL_W 32                     // BlurColumnsTiled
#define COL_H 32
#define COL_THREADS_Y 8
#define TILED_MAX_RADIUS 16          // the blurs' radii with their own kernels (unrolled loops)
#define ORI_SG_OPENCV  4             // OrientationsCompat: SIMD groups (keypoints) per threadgroup
#define ORI_SG_VLFEAT  2
#define DESC_NT_OPENCV 128           // DescriptorsCompat: threads per threadgroup (keypoint)
#define DESC_NT_VLFEAT 128
#define DESC_GROUP     8             // DescriptorsCompat: lanes per histogram
// DescriptorsCompat computes the 128 values of a descriptor with a thread each
#if DESC_NT_OPENCV<128 || DESC_NT_VLFEAT<128
#error "DescriptorsCompat needs at least 128 threads per threadgroup"
#endif

// The scale space, as CompatPyramid in cudaSiftCompat.cu, with the levels as offsets (in floats)
// in the scale-space buffer
struct CompatPyramidM {
  int nOct, nLev;
  int w[COMPAT_MAX_OCTAVES], h[COMPAT_MAX_OCTAVES], p[COMPAT_MAX_OCTAVES];
  float pix[COMPAT_MAX_OCTAVES];   // pixel size of the octave in keypoint coordinates
  int lev[COMPAT_MAX_OCTAVES];     // levels below the mode's first one (the extension)
  unsigned int off[COMPAT_MAX_OCTAVES][COMPAT_MAX_LEVELS];
};

// Parameters of the extrema detection and refinement (CompatDetectParams), in float
struct CompatDetectParamsM {
  int S, s_min, s_max;     // levels per octave; VLFeat's first and last levels (-1, S + 1)
  float edgeThreshold;
  // OpenCV
  float preThreshold;      // OpenCV's integer threshold on the DoG value before refinement
  float contrastThreshold, sigma;
  // VLFeat
  float peakThreshold;     // largest float <= the peak threshold: v > it exactly when v is
                           // above the threshold (VLFeat's test in double)
  float peakPrefilter;     // smallest float >= 0.8*peakThreshold (the same for >=)
  float sigma0;
};

// The keypoints of both modes, as in cudaSiftCompat.cu (VLFeat's angle in float)
struct OpenCVKeypoint {
  float x, y, size, angle, response;
  int packedOctave;
  float contr, edge;
  int o, layer, r, c;
};

struct VLFeatKeypointM {
  float x, y, sigma;
  float angle;             // radians
  float val, score;
  int o, is;
  int ori;
  int raster;
};

// The parameters of a blur pass and of the per-pixel kernels
struct CompatImageArgs {
  unsigned int dst, src;   // offsets in the scale-space buffer
  int w, h, p;             // of dst (and src, but for InputCompat and DownsampleCompat)
  int sw, sh, sp;          // of src for InputCompat and DownsampleCompat
  int radius, upsample;
};

#endif
