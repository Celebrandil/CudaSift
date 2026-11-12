//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

// Seeded random sampling for FindHomographySeeded and FindHomographyGPU, and the check of
// the samples, shared by the host (C++), CUDA and Metal code, so that all of them draw
// and keep the same samples for the same seed. The random numbers come from a
// counter-based generator (a splitmix64-style hash of the seed, the hypothesis index and
// a counter), so each RANSAC hypothesis has its own reproducible stream and can be drawn
// independently on the GPU.

#ifndef CUDASIFTRANDOM_H
#define CUDASIFTRANDOM_H

#if defined(__METAL_VERSION__)
#define SIFTRANDOM_FUNC static inline
#define SIFTRANDOM_THREAD thread
typedef uint siftrandom_u32;
typedef ulong siftrandom_u64;
typedef float siftrandom_real;     // no double precision
#elif defined(__CUDACC__)
#define SIFTRANDOM_FUNC static inline __host__ __device__
#define SIFTRANDOM_THREAD
typedef unsigned int siftrandom_u32;
typedef unsigned long long siftrandom_u64;
typedef double siftrandom_real;
#else
#define SIFTRANDOM_FUNC static inline
#define SIFTRANDOM_THREAD
typedef unsigned int siftrandom_u32;
typedef unsigned long long siftrandom_u64;
typedef double siftrandom_real;
#endif

// The 'counter'-th 32-bit random number of stream 'stream', for a given seed
SIFTRANDOM_FUNC siftrandom_u32 SiftRandom(siftrandom_u32 seed, siftrandom_u32 stream, siftrandom_u32 counter)
{
  siftrandom_u64 z = (((siftrandom_u64)seed << 32) | stream)*0x9E3779B97F4A7C15ull + (siftrandom_u64)counter*0xD1B54A32D192ED03ull + 0x632BE59BD9B4E019ull;
  z = (z ^ (z >> 30))*0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27))*0x94D049BB133111EBull;
  return (siftrandom_u32)((z ^ (z >> 31)) >> 32);
}

// Four distinct indices in [0, numValid) for hypothesis 'loop', drawn as in
// FindHomography (but from the seeded generator instead of rand()). numValid >= 4.
SIFTRANDOM_FUNC void SiftRandomSample4(siftrandom_u32 seed, siftrandom_u32 loop, int numValid, SIFTRANDOM_THREAD int *p)
{
  siftrandom_u32 c = 0;
  p[0] = SiftRandom(seed, loop, c++) % numValid;
  p[1] = SiftRandom(seed, loop, c++) % numValid;
  p[2] = SiftRandom(seed, loop, c++) % numValid;
  p[3] = SiftRandom(seed, loop, c++) % numValid;
  while (p[1]==p[0])
    p[1] = SiftRandom(seed, loop, c++) % numValid;
  while (p[2]==p[0] || p[2]==p[1])
    p[2] = SiftRandom(seed, loop, c++) % numValid;
  while (p[3]==p[0] || p[3]==p[1] || p[3]==p[2])
    p[3] = SiftRandom(seed, loop, c++) % numValid;
}

// Whether the 4 matches (x1[k], y1[k]) -> (x2[k], y2[k]) of a RANSAC sample give the
// triangles they form the same orientation in both images, as checked by OpenCV's
// findHomography, COLMAP and PoseLib. A homography through a sample that fails reverses
// the orientation or is degenerate: no camera viewing a plane produces it, so the sample
// gives no hypothesis. In double precision the signs are exact for float coordinates.
SIFTRANDOM_FUNC bool SiftSameOrientation(SIFTRANDOM_THREAD const float *x1, SIFTRANDOM_THREAD const float *y1,
                                         SIFTRANDOM_THREAD const float *x2, SIFTRANDOM_THREAD const float *y2)
{
  const int tri[4][3] = {{0, 1, 2}, {0, 1, 3}, {2, 3, 0}, {2, 3, 1}};
  for (int t=0;t<4;t++) {
    const int a = tri[t][0], b = tri[t][1], c = tri[t][2];
    siftrandom_real d1 = ((siftrandom_real)x1[b] - x1[a])*((siftrandom_real)y1[c] - y1[a]) - ((siftrandom_real)y1[b] - y1[a])*((siftrandom_real)x1[c] - x1[a]);
    siftrandom_real d2 = ((siftrandom_real)x2[b] - x2[a])*((siftrandom_real)y2[c] - y2[a]) - ((siftrandom_real)y2[b] - y2[a])*((siftrandom_real)x2[c] - x2[a]);
    if (d1*d2<0)
      return false;
  }
  return true;
}

#endif
