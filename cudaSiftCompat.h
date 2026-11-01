//********************************************************//
// OpenCV- and VLFeat-compatible SIFT extraction          //
//********************************************************//

#ifndef CUDASIFTCOMPAT_H
#define CUDASIFTCOMPAT_H

#include <cstddef>   // NULL, used by cudaImage.h
#include "cudaImage.h"
#include "cudaSift.h"

// The two pipelines. COLMAP's features are VLFeat's with other parameters (SiftCompatParams::COLMAP).
enum SiftCompatMode {
  SIFT_COMPAT_OPENCV,   // cv::SIFT (OpenCV 4.8 and later, enable_precise_upscale = true)
  SIFT_COMPAT_VLFEAT    // vl_sift_* of VLFeat 0.9.21
};

// Parameters of ExtractSiftCompat. The default is OpenCV's SIFT with cv::SIFT::create's
// defaults; VLFeat() and COLMAP() give VLFeat's defaults and COLMAP's settings.
struct SiftCompatParams {
  SiftCompatMode mode = SIFT_COMPAT_OPENCV;
  int nOctaveLayers = 3;            // S, intervals per octave (VLFeat's nlevels), 1 to 6
  int firstOctave = -1;             // -1: start with the image upsampled by 2, or 0
  double edgeThreshold = 10.0;      // on the ratio of principal curvatures
  // Not in OpenCV, VLFeat or COLMAP: with firstOctave 0, add to the first octave the
  // Gaussian levels below sigma that the image upsampled by 2 would have (octave -1), at
  // sigma * 2^(-j/S), j = 1..S, blurred from the input at the original resolution. The first
  // octave then has 2S detection scales instead of S, down to half the scale; the other levels,
  // octaves and thresholds are unchanged, so the keypoints are those of firstOctave 0 plus
  // finer ones. extendFirstOctaveScales keeps only that many of the added scales, the
  // coarsest (and computes only the levels they need): 0 adds none, S or more (or -1) all S.
  bool extendFirstOctave = false;
  int extendFirstOctaveScales = -1;
  // Precision of the refinement, orientations and descriptors. Float by default, which
  // deviates from VLFeat's code (and COLMAP's), where they are double, and is faster on GPUs
  // with slow double precision; the features are essentially the same (on HPatches, 0 of
  // VLFeat's 789016 keypoints and 2 of COLMAP's 1555638 missing in float, 0 in double). With
  // OpenCV, which computes in float, double changes the refinement and the orientation
  // histograms (a deviation from OpenCV). Not available with the Metal backend (Apple GPUs have
  // no double precision), whose builds define CUDASIFT_COMPAT_NO_DOUBLE.
  bool doublePrecision = false;
  // OpenCV only
  int nfeatures = 0;                // keep the nfeatures strongest keypoints, ties included (0: all)
  double contrastThreshold = 0.04;  // on the interpolated DoG value, times nOctaveLayers
  double sigma = 1.6;               // blur of the first level of each octave
  // VLFeat only
  int numOctaves = -1;              // -1: VLFeat's default (OpenCV computes its own number)
  double peakThreshold = 0.0;       // on the interpolated DoG value, of the image divided by
                                    // 255 (x/255.0f, as COLMAP gives it to VLFeat)
  int maxOrientations = 4;          // orientations per keypoint, the first peaks in bin order
                                    // (VLFeat scans at most 4)
  int maxFeatures = 0;              // COLMAP: keep whole DoG levels, coarsest first, until
                                    // they have more than maxFeatures detections (0: all;
                                    // unlike nfeatures, by level rather than by response)
  bool rootSift = false;            // COLMAP: L1-root descriptors in OpenCV's bin order,
                                    // rounded to 8 bits as COLMAP stores them

  static SiftCompatParams OpenCV() { return SiftCompatParams(); }
  static SiftCompatParams VLFeat()
  {
    SiftCompatParams p;
    p.mode = SIFT_COMPAT_VLFEAT;
    p.firstOctave = 0;
    return p;
  }
  // COLMAP's SiftExtractionOptions (first_octave -1, 4 octaves, peak 0.02/3, edge 10, at most
  // 2 orientations and 8192 features, L1_ROOT normalization). The positions are not shifted:
  // COLMAP adds 0.5 to VLFeat's x and y (pixel centres at half-integer coordinates), so add
  // 0.5 to xpos and ypos to get COLMAP's keypoints
  static SiftCompatParams COLMAP()
  {
    SiftCompatParams p = VLFeat();
    p.firstOctave = -1;
    p.numOctaves = 4;
    p.peakThreshold = 0.02/3;
    p.maxOrientations = 2;
    p.maxFeatures = 8192;
    p.rootSift = true;
    return p;
  }
};

// Extracts SIFT keypoints and descriptors from img (values 0 to 255) as OpenCV's SIFT
// (cv::SIFT::detectAndCompute) or VLFeat's (vl_sift_detect, vl_sift_calc_keypoint_orientations
// and vl_sift_calc_keypoint_descriptor) do, with separate kernels from ExtractSift. Both modes
// run the same kernels; the mode and the parameters select each library's steps. The
// SiftPoint fields:
//   xpos, ypos    position in pixels of img, pixel centres at integer coordinates, in all
//                 modes (COLMAP's are 0.5 more, see SiftCompatParams::COLMAP)
//   scale         sigma of the keypoint (OpenCV's KeyPoint::size / 2)
//   orientation   degrees in [0, 360), clockwise in image coordinates (OpenCV's
//                 KeyPoint::angle; VLFeat's angle in degrees)
//   sharpness     interpolated DoG value (OpenCV: KeyPoint::response with its sign)
//   edgeness      trace^2 / determinant of the DoG Hessian
//   subsampling   pixel size of the keypoint's octave (0.5 for octave -1)
//   data          the descriptor, of unit L2 norm (for MatchSiftData): OpenCV's 8-bit
//                 descriptor scaled, VLFeat's (in its bin order), or COLMAP's 8-bit
//                 descriptor (in OpenCV's bin order) scaled
// Keypoints are in OpenCV's order (by x, then y; with nfeatures, OpenCV's own order is that of
// its std::nth_element selection, which isn't reproduced) or VLFeat's (by octave, level, y, x). If
// there are more than siftData.maxPts, the strongest (largest |sharpness|) are kept, with a
// message on stderr. Invalid parameters (nOctaveLayers outside 1..6, firstOctave other than -1
// or 0, extendFirstOctave without firstOctave 0, numOctaves 0 or below -1, or a sigma needing a
// Gaussian kernel of more than 129 taps) stop the program with a message. Thread-safe: each
// device has its own temporary memory, kept between calls, and calls on the same device run one
// at a time. With the Metal backend, not in a batch (cudaSiftBatch.h).
void ExtractSiftCompat(SiftData &siftData, CudaImage &img, const SiftCompatParams &params = SiftCompatParams());

// ExtractSiftCompat keeps its temporary memory between calls; this releases it, on the current
// device. Call it before cudaDeviceReset or before destroying the device's context.
void FreeSiftCompatMemory();

#endif
