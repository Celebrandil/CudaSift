# CudaSift — GPU-Accelerated SIFT Feature Detection & Matching

**Branch: AdaLovelace** — Optimized for NVIDIA Ada Lovelace architecture (RTX 4060 Ti, sm_89)

A high-performance CUDA implementation of the Scale Invariant Feature Transform (SIFT) algorithm. This implementation runs the complete SIFT pipeline on the GPU, achieving sub-millisecond feature extraction on modern NVIDIA hardware.

Based on the original work by Mårten Björkman (Celebrandil), with Ada Lovelace architecture optimizations.

---

## Hardware Target

| Spec | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4060 Ti |
| Architecture | Ada Lovelace (sm_89) |
| CUDA Cores | 4352 |
| VRAM | 8 GB GDDR6 |
| Memory Bandwidth | 288 GB/s |
| FP32 Performance | ~22.1 TFLOPS |
| L2 Cache | 32 MB |
| Driver | 595.71 |

## Performance Benchmarks

### SIFT Extraction (5 octaves, threshold=3.0)

| Resolution | Size | Features | Extract (ms) | Match (ms) | Total (ms) | FPS |
|-----------|------|----------|-------------|------------|------------|-----|
| VGA | 640x480 | 653 | 0.77 | 0.12 | 1.04 | 965 |
| 720p | 1280x720 | 1155 | 0.91 | 0.20 | 1.47 | 681 |
| SXGA | 1280x960 | 1326 | 0.99 | 0.21 | 1.66 | 601 |
| 1080p | 1920x1080 | 1911 | 1.38 | 0.34 | 2.49 | 402 |
| 1440p | 2560x1440 | 2244 | 1.85 | 0.38 | 3.56 | 281 |
| 4K UHD | 3840x2160 | 2829 | 3.53 | 0.51 | 6.95 | 144 |

> Benchmarked on RTX 4060 Ti (Driver 595.71, CUDA 13.1). Compute Capability 8.9, 34 SMs, 8187 MB VRAM, 128-bit bus, 32768 KB L2 cache.
>
> These RTX 4060 Ti numbers were measured with the previous kernels (`LaplaceMulti` + `FindPointsMulti`, `FindMaxCorr10`), not the current ones.

### Feature Matching

| Features | Match Time (ms) |
|----------|----------------|
| 1911 (self-match) | 0.33 |

### Octave Comparison (1080p, threshold=3.0)

| Octaves | Features | Extract (ms) |
|---------|----------|-------------|
| 3 | 1741 | 1.24 |
| 4 | 1877 | 1.35 |
| 5 | 1911 | 1.60 |
| 6 | 1920 | 1.80 |

### Threshold Comparison (1080p, 5 octaves)

| Threshold | Features | Extract (ms) |
|-----------|----------|-------------|
| 1.0 | 7081 | 2.07 |
| 2.0 | 3700 | 1.79 |
| 3.0 | 1911 | 1.59 |
| 5.0 | 542 | 1.32 |
| 10.0 | 6 | 1.35 |

### Cross-Architecture Comparison

| Arch | GPU | Extract 1280x960 | Extract 1920x1080 | Match (ms) | GFLOPS | BW (GB/s) |
|------|-----|------------------|--------------------|------------|--------|-----------|
| Pascal | GTX 1080 Ti | 1.20* | 1.70* | 2.20* | 11340 | 484 |
| Turing | RTX 2080 Ti | 0.42* | 0.56* | 0.30* | 11750 | 616 |
| **Ada** | **RTX 4060 Ti** | **0.99** | **1.38** | **0.33** | **22060** | **288** |

> \* Values from original CudaSift benchmarks. Ada values measured with Driver 595.71, CUDA 13.1.

## Architecture Overview

```
Input Image (Host -> Device)
         |
         v
+--------------------------------------------------+
|          Gaussian Scale Space                     |
|  Octave 0 (full) -> Octave 1 (1/2) -> ... -> N   |
|         |                                         |
|         v                                         |
|  ScaleDownDirect: 2x downsampling per octave      |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|    DoG + Keypoint Detection (fused)               |
|  LaplaceFindPoints:                               |
|    - DoG layers in shared memory, per tile        |
|      (5 scales + 3 border per octave)             |
|    - 3D extrema detection (26 neighbors)          |
|    - Edge response rejection                      |
|    - Sub-pixel localization (Taylor expansion)    |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|        Orientation Assignment                     |
|  ComputeOrientationsCONST:                        |
|    - 32-bin gradient histogram                    |
|    - Gaussian-weighted 11x11 window               |
|    - Secondary peak -> duplicate feature          |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|       Descriptor Computation                      |
|  ExtractSiftDescriptorsGather:                    |
|    - 4x4 spatial bins x 8 orientations            |
|    - 128-D vector per feature                     |
|    - Two-pass normalization (clip + renorm)       |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|          Feature Matching                         |
|  FindMaxCorr11 + MergeMaxCorr11 (brute-force):    |
|    - 64x64 tile pairs over persistent blocks      |
|    - 8x8 correlations per thread                  |
|    - Best + second-best per tile, then merged     |
|                                                   |
|  FindHomography (RANSAC):                         |
|    - 4-point DLT on GPU                           |
|    - Parallel hypothesis testing                  |
|    - Iterative refinement (CPU, Cholesky)         |
|                                                   |
|  FindHomographyGPU + ImproveHomographyGPU:        |
|    - The same, entirely on the GPU                |
+--------------------------------------------------+
```

## CUDA Kernel Configuration

| Kernel | Block Size | Shared Mem | Description |
|--------|-----------|------------|-------------|
| ScaleDownDirect | 32x8 | - | 2x downsampling with 5-tap Gaussian |
| LaplaceFindPoints | 288x1 | 19 KB | Multi-scale DoG + 3D extrema detection + sub-pixel |
| ComputeOrientationsCONST | 121x1 | 1.2 KB | Gradient histogram, peak detection |
| ExtractSiftDescriptorsGather | 16x8 | 6.1 KB | 128-D descriptor with trilinear interp |
| FindMaxCorr11 + MergeMaxCorr11 | 64x1, 128x1 | 8.5 KB | Tiled brute-force matching |

## Building

### Prerequisites

- **CUDA Toolkit** 11.0+ (recommended 12.x for Ada Lovelace)
- **OpenCV** 4.x
- **CMake** 3.18+
- **C++17** compatible compiler

### Quick Build (Windows)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Quick Build (Linux)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Using the Build Script

```bash
bash scripts/build.sh Release
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | ON | Build test and benchmark programs |
| `BUILD_EXAMPLES` | ON | Build example programs |
| `CMAKE_CUDA_ARCHITECTURES` | 89 | Target GPU architecture(s), e.g. `75` for a Tesla T4 (also `CUDAARCHS`) |
| `USE_MANAGED_MEM` | OFF | Use CUDA managed memory |
| `VERBOSE_OUTPUT` | ON | Enable verbose timing output |

## Usage

### Main Demo

```bash
# Default (img1.png & img2.png)
./cudasift

# Specify GPU device and image set
./cudasift 0 1   # device 0, PGM image set
```

### Feature Extraction Demo

```bash
./demo_extract [image_path] [gpu_id] [threshold] [num_octaves]
./demo_extract data/img1.png 0 3.0 5
```

Output: `data/keypoints.png` with detected keypoints drawn.

### Feature Matching Demo

```bash
./demo_match [img1] [img2] [gpu_id]
./demo_match data/img1.png data/img2.png 0
```

Output: `data/matches.png` with match lines between images.

### Real-time Video Demo

```bash
./demo_video [source] [gpu_id] [threshold]
./demo_video 0           # Webcam
./demo_video video.mp4   # Video file
```

Keys: `q` quit, `+`/`-` adjust threshold.

### Performance Benchmark

```bash
./benchmark [gpu_id] [num_runs] [threshold]
./benchmark 0 200 3.0
```

Outputs performance tables at multiple resolutions: extraction, matching, upload and homography estimation times (`FindHomography` + `ImproveHomography`, and `FindHomographyGPU` + `ImproveHomographyGPU`).

## Running Tests

The tests read `data/` by relative path, so run them from the repository root:

```bash
# Individual tests
build/test_extract         # Feature extraction correctness
build/test_match           # Matching and quality tests
build/test_homography      # Geometric verification tests
build/test_homography_gpu  # FindHomographyGPU, ImproveHomographyGPU
build/test_batch           # Batches of calls (cudaSiftBatch.h)

# All built tests + benchmark (default build directory: build)
bash scripts/run_benchmark.sh [build directory]
```

Each test prints a `[PASS]`, `[FAIL]` or `[SKIP]` line per check and exits with a nonzero code on failure:

| Test | What it verifies |
|------|-----------------|
| test_extract | Extraction: features, thresholds, octaves, reproducibility, upsampling, and the regression tests of the bug fixes |
| test_match | Self-matching, cross-matching, RANSAC and refinement, matching time |
| test_homography | Translation, rotation, scale, stereo pair, and the regression tests of the homography fixes |
| test_homography_gpu | `FindHomographyGPU` and `ImproveHomographyGPU` against `FindHomography` and `ImproveHomography` |

## API Reference

### Core Functions

```cpp
// Initialize CUDA device
void InitCuda(int devNum = 0);

// Allocate/free temporary GPU memory for extraction
float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp = false);
void FreeSiftTempMemory(float *memoryTmp);

// Extract SIFT features from a GPU image
void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves,
                 double initBlur, float thresh, float lowestScale = 0.0f,
                 bool scaleUp = false, float *tempMemory = 0);

// Initialize/free SIFT data container
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data);

// Sort the points by position (optional, for reproducible results; see below)
void SortSiftData(SiftData &data);

// Match two sets of SIFT features on GPU
double MatchSiftData(SiftData &data1, SiftData &data2);

// Find homography using RANSAC, among the matches with score > minScore and
// ambiguity < maxAmbiguity; numMatches is the number of inliers among them
double FindHomography(SiftData &data, float *homography, int *numMatches,
                      int numLoops = 1000, float minScore = 0.85f,
                      float maxAmbiguity = 0.95f, float thresh = 5.0f);

// As FindHomography, with a seeded random number generator instead of rand()
double FindHomographySeeded(SiftData &data, float *homography, int *numMatches,
                            int numLoops = 1000, float minScore = 0.85f,
                            float maxAmbiguity = 0.95f, float thresh = 5.0f,
                            unsigned int seed = 42);

// Find homography using RANSAC entirely on the GPU, with the same seeded
// random number generator (see Homography Estimation on the GPU)
double FindHomographyGPU(SiftData &data, float *homography, int *numMatches,
                         int numLoops = 1000, float minScore = 0.85f,
                         float maxAmbiguity = 0.95f, float thresh = 5.0f,
                         unsigned int seed = 42);

// Refine a homography by least squares over the inliers, on the GPU
// (ImproveHomography in geomFuncs.cpp does it on the CPU)
double ImproveHomographyGPU(SiftData &data, float *homography, int *numFit,
                            int numLoops = 5, float minScore = 0.0f,
                            float maxAmbiguity = 0.80f, float thresh = 3.0f);

// Refine a homography on the CPU (geomFuncs.cpp). Not declared in any
// header: declare it yourself, as the tests do. Returns the number of inliers.
int ImproveHomography(SiftData &data, float *homography, int numLoops,
                      float minScore, float maxAmbiguity, float thresh);
```

- `FindHomographyGPU` and `ImproveHomographyGPU` return the GPU time in ms. Inside a batch (`cudaSiftBatch.h`), the functions that return a time return 0.0.
- `AllocSiftTempMemory` reserves no DoG layers, since `LaplaceFindPoints` keeps them in shared memory. With 5 octaves, it takes 1.7 MB at 640x480, 10.7 MB at 1920x1080 and 42.4 MB at 3840x2160 (74.0 MB at 1920x1080 with `scaleUp`). `ExtractSift` without `tempMemory` allocates it on every call.
- `InitSiftData` allocates `h_data` in pinned memory (`cudaMallocHost`), which makes the copies from the device faster. `FreeSiftData` frees it with `cudaFreeHost`, or with `free` if the caller allocated `h_data` with `malloc`.
- Threads: `ExtractSift`, `MatchSiftData`, `FindHomographyGPU` and `ImproveHomographyGPU` can be called from several host threads. Each holds a lock of its device until it completes, so calls on the same device run one at a time, and calls on different devices run concurrently. `FindHomographySeeded` is thread-safe too; `FindHomography` uses `rand()`, which is not.
- Cached constant uploads and scratch memory are not invalidated by `cudaDeviceReset()`. Don't reset the device while using CudaSift.

### Homography Estimation on the GPU

`FindHomography` selects and samples the matches on the CPU with `rand()`, and `ImproveHomography` (`geomFuncs.cpp`) runs on the CPU. `FindHomographyGPU` and `ImproveHomographyGPU` do the same entirely on the GPU, so the matches stay there.

- `FindHomographySeeded` is `FindHomography` with a counter-based random number generator seeded with `seed` (`cudaSiftRandom.h`) instead of `rand()`, so the samples are reproducible.
- `FindHomographyGPU` draws the same samples, and gives the same result as `FindHomographySeeded`.
- Both depend on the order of the points, as `FindHomography` does, and `ExtractSift` writes the points in an order that varies between runs. For reproducible results (tests, benchmarks), sort both point sets with `SortSiftData` after `ExtractSift` and before `MatchSiftData`: it orders them by position, then scale, orientation and descriptor, on the CPU (not in a batch). It is optional; without it, the results vary slightly between runs.
- `ImproveHomographyGPU` reproduces `ImproveHomography` in double precision. When an iteration has fewer than 4 inliers or the normal equations are not positive definite, it keeps the homography unchanged, as `ImproveHomography` does.

Compared with `FindHomography` and `ImproveHomography` on `data/img1.png` and `data/img2.png` (10000 hypotheses, `tests/test_homography_gpu`, Tesla T4):

| | CUDA (Tesla T4) |
|---|---|
| Matches: `FindHomography` / `FindHomographyGPU` (seeds 1–10) | 957 / 957–958 |
| Both refined with `ImproveHomography` | Same homography, 948 inliers |
| Inliers: `ImproveHomography` / `ImproveHomographyGPU` | 948 / 948 |
| Homography difference (max over the inliers) | 0.0003 px |
| `match_error` difference (max) | 0.00015 px |

### Batches of Calls

Outside a batch, each call returns after its GPU work has completed. `ExtractSift`, `MatchSiftData`, `FindHomographyGPU` and `ImproveHomographyGPU` calls made between `BeginSiftBatch()` and `EndSiftBatch()` (`cudaSiftBatch.h`) don't wait: `EndSiftBatch()` returns immediately, the CPU can do other work while the batch runs, and `WaitSiftBatch()` makes the results available. The results are identical to those of the same calls outside a batch.

```cpp
#include "cudaSiftBatch.h"

BeginSiftBatch();
ExtractSift(siftData2, img2, 5, 1.0f, 3.0f);
MatchSiftData(siftData1, siftData2);
EndSiftBatch();                     // returns immediately
...                                 // other CPU work, e.g. on the previous frame's results
double gpuTime = WaitSiftBatch();   // results are valid
```

The calls of a batch keep the point counts on the GPU and leave the copies to the host to the completion of the batch. `cudaSiftBatch.h` documents the rules. Frame loop of `tests/test_batch` on a Tesla T4 (a 1080p extraction and its matching with the previous frame): 1.39 ms per frame with separate calls and 1.38 ms in a batch; with 1 ms of other CPU work per frame, 2.29 ms and 1.38 ms.

### Data Structures

```cpp
struct SiftPoint {
  float xpos, ypos;       // Sub-pixel position
  float scale;            // Feature scale (sigma)
  float sharpness;        // DoG response value
  float edgeness;         // Edge response ratio
  float orientation;      // Dominant orientation (degrees)
  float score;            // Match correlation score
  float ambiguity;        // Second-best / best ratio
  int match;              // Index of best match
  float match_xpos, match_ypos;  // Matched point position
  float match_error;      // Reprojection error
  float subsampling;      // Octave subsampling factor
  float data[128];        // 128-D descriptor vector
};

struct SiftData {
  int numPts;             // Number of detected features
  int maxPts;             // Allocated capacity
  SiftPoint *h_data;      // Host pointer
  SiftPoint *d_data;      // Device pointer
};
```

## File Structure

```
CudaSift/
|-- CMakeLists.txt          # Modern CMake build (sm_89)
|-- README.md               # This file
|-- LICENSE                  # MIT License
|
|-- cudaSift.h              # Public API header
|-- cudaSiftBatch.h         # Batches of calls
|-- cudaSiftRandom.h        # Seeded random numbers (FindHomographySeeded, FindHomographyGPU)
|-- cudaSiftH.cu            # Host-side SIFT pipeline
|-- cudaSiftH.h             # Host function declarations
|-- cudaSiftD.cu            # Device kernels (DoG, keypoints, descriptors)
|-- cudaSiftD.h             # Kernel constants and block sizes
|-- cudaImage.cu            # GPU image container
|-- cudaImage.h             # Image class declaration
|-- cudautils.h             # CUDA utilities (error checking, timers, shuffle)
|-- matching.cu             # Matching kernels + RANSAC homography
|-- geomFuncs.cpp           # CPU homography refinement
|-- mainSift.cpp            # Main demo program
|
|-- examples/
|   |-- demo_extract.cpp    # Single-image extraction demo
|   |-- demo_match.cpp      # Two-image matching demo
|   +-- demo_video.cpp      # Real-time video demo
|
|-- tests/
|   |-- benchmark.cpp       # Multi-resolution performance benchmark
|   |-- test_extract.cpp    # Extraction correctness tests
|   |-- test_match.cpp      # Matching quality tests
|   |-- test_homography.cpp # Geometric verification tests
|   |-- test_homography_gpu.cpp # FindHomographyGPU, ImproveHomographyGPU
|   +-- test_batch.cpp      # Batches of calls
|
|-- scripts/
|   |-- build.sh            # Build script
|   +-- run_benchmark.sh    # Run all tests + benchmark
|
|-- data/
|   |-- img1.png            # Test image 1 (1280x960)
|   |-- img2.png            # Test image 2 (1280x960)
|   |-- left.pgm            # Stereo left image
|   +-- righ.pgm            # Stereo right image
|
+-- match.pdf               # Matching kernel optimization notes
```

## Ada Lovelace Optimizations

This branch includes the following optimizations for the Ada Lovelace architecture:

1. **sm_89 Compute Target** -- Native code generation for RTX 40-series GPUs
2. **Fast Math** -- `--use_fast_math` for all CUDA kernels (intrinsic sin/cos/exp/sqrt)
3. **Large L2 Cache** -- RTX 4060 Ti has 32 MB L2 cache, benefiting texture lookups and DoG pyramid reads
4. **Warp Synchronization** -- All warp-level operations use `__shfl_sync` with full mask
5. **Optimized Block Sizes** -- Tuned for 128 SMs and Ada Lovelace occupancy characteristics
6. **C++17 / CUDA 17** -- Modern language standard support
7. **Static Library** -- Core SIFT compiled as static library for faster linking

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `numOctaves` | 5 | Number of octaves in scale space |
| `initBlur` | 1.0 | Initial Gaussian blur sigma |
| `thresh` | 3.0 | DoG threshold for keypoint detection |
| `lowestScale` | 0.0 | Minimum scale for features |
| `scaleUp` | false | 2x upsample input for fine features |
| `maxPts` | 32768 | Maximum number of features |
| `minScore` | 0.85 | Minimum match score for RANSAC |
| `maxAmbiguity` | 0.95 | Maximum ambiguity ratio for RANSAC |

## References

- David G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," IJCV, 2004.
- Original CudaSift by Marten Bjorkman: https://github.com/Celebrandil/CudaSift

## License

MIT License -- see [LICENSE](LICENSE) for details.
