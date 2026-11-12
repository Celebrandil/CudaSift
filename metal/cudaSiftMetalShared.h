//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//              Apple Metal backend                       //
//********************************************************//

// Types shared by the Metal host code (cudaSiftMetal.mm) and the Metal kernels
// (cudaSift.metal, which is compiled at run time with this header prepended).
// Only plain int/float members, so that the layout is the same on both sides.

#ifndef CUDASIFTMETALSHARED_H
#define CUDASIFTMETALSHARED_H

// Kernel geometry, used by the kernels and for the dispatch sizes of the host code
// (MAX_OCTAVES comes from cudaSiftD.h, which is included or prepended first)
#define NUM_COUNTERS   ((MAX_OCTAVES+1)*2+1)  // d_PointCounter, as in the CUDA code
#define FUSED_FY       4        // LaplaceFindPoints: FindPoints tiles (MINMAX_H rows) per threadgroup
#define FUSED_NT       288      // LaplaceFindPoints: threads per threadgroup
#define ORIENT_SG      8        // ComputeOrientationsSimd: SIMD groups (points) per threadgroup
#define ORIENT_GROUPS  128      // ComputeOrientationsSimd: threadgroups
#define DESC_GROUPS    512      // ExtractSiftDescriptorsGather: threadgroups (of 16x8 threads)
#define M12_T          64       // FindMaxCorr12: points per tile (sift1 and sift2)
#define M12_NT         256      // FindMaxCorr12: threads per threadgroup
#define M12_CHUNKS     64       // FindMaxCorr12: sift2 chunks
#define MERGE_NT       128      // MergeMaxCorr12: threads per threadgroup
#define TESTHOMO_TESTS 16       // TestHomographies: tests per threadgroup
#define TESTHOMO_LOOPS 16       // TestHomographies: hypotheses per threadgroup
#define TESTHOMO_PER   4        // TestHomographiesGPU: hypotheses per thread
#define HOMOGRAPHY_NT  1024     // HomographyPrepare, HomographyBest: threads (one threadgroup)
#define IMPROVE_GROUPS 16       // ImproveAccumulate, ImproveFinal: threadgroups
#define IMPROVE_NT     256      // ImproveAccumulate, ImproveFinal: threads per threadgroup
#define IMPROVE_NV     35       // 27 non-zero entries of the symmetric 8x8 matrix + 8 of the vector
// Size of ImproveState (cudaSift.metal): 8 + IMPROVE_GROUPS*IMPROVE_NV pairs of floats, IMPROVE_GROUPS ints
#define IMPROVE_STATE_SIZE ((8 + IMPROVE_GROUPS*IMPROVE_NV)*2*sizeof(float) + IMPROVE_GROUPS*sizeof(int))

struct ImageParams {     // LowPass, ScaleDown, ScaleUp
  int width, pitch, height, newpitch;
};

struct FindPointsParams { // LaplaceFindPoints
  int width, pitch, height, octave, maxPts;
  float subsampling, lowestScale, thresh, factor, edgeLimit;
};

struct OctaveParams {     // ComputeOrientations, ExtractSiftDescriptors
  int octave, maxPts;
  float subsampling;      // ExtractSiftDescriptors only
};

struct MatchParams {      // FindMaxCorr12, MergeMaxCorr12 (point counts are separate arguments)
  int numChunks;
};

struct HomographyParams { // ComputeHomographies, TestHomographies
  int numPts, numLoops;      // numPts: point stride of the coordinate arrays
  float thresh2;
  int count;                 // number of points tested by TestHomographies
};

struct HomographyGPUParams { // FindHomographyGPU and ImproveHomographyGPU kernels
  int numLoops, stride;      // RANSAC hypotheses; point stride of the coordinate arrays
  float minScore, maxAmbiguity, thresh2;
  unsigned int seed;
};

struct HomographyResult {    // homography and number of matches / inliers
  float h[9];
  int count;
};

#endif
