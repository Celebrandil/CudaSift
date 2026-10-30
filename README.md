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

### Feature Matching (FindMaxCorr10 kernel)

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
|  LaplaceMulti: DoG computation                    |
|  (5 scales + 3 border per octave)                 |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|          Keypoint Detection                       |
|  FindPointsMulti:                                 |
|    - 3D extrema detection (26 neighbors)          |
|    - Edge response rejection                      |
|    - Sub-pixel localization (Taylor expansion)    |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|        Orientation Assignment                     |
|  ComputeOrientations:                             |
|    - 32-bin gradient histogram                    |
|    - Gaussian-weighted 11x11 window               |
|    - Secondary peak -> duplicate feature          |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|       Descriptor Computation                      |
|  ExtractSiftDescriptors:                          |
|    - 4x4 spatial bins x 8 orientations            |
|    - 128-D vector per feature                     |
|    - Two-pass normalization (clip + renorm)       |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|          Feature Matching                         |
|  FindMaxCorr10 (brute-force):                     |
|    - 32x32 feature block tiling                   |
|    - float4 vectorized loads                      |
|    - Warp shuffle reductions                      |
|    - Best + second-best tracking (ambiguity)      |
|                                                   |
|  FindHomography (RANSAC):                         |
|    - 4-point DLT on GPU                           |
|    - Parallel hypothesis testing                  |
|    - Iterative refinement (CPU, Cholesky)         |
+--------------------------------------------------+
```

## CUDA Kernel Configuration

| Kernel | Block Size | Shared Mem | Description |
|--------|-----------|------------|-------------|
| ScaleDown | 68x1 | 2 KB | 2x downsampling with 5-tap Gaussian |
| LaplaceMulti | 136x1 | 4 KB | Multi-scale DoG computation |
| FindPointsMulti | 32x1 | 1 KB | 3D extrema detection + sub-pixel |
| ComputeOrientations | 121x1 | 0.5 KB | Gradient histogram, peak detection |
| ExtractSiftDescriptors | 16x8 | 0.7 KB | 128-D descriptor with trilinear interp |
| FindMaxCorr10 | 32x8 | 32 KB | Tiled brute-force matching |

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

Outputs performance tables at multiple resolutions with extraction, matching, and upload times.

## Running Tests

```bash
# Individual tests
./test_extract     # Feature extraction correctness
./test_match       # Matching and quality tests
./test_homography  # Geometric verification tests

# All tests + benchmark
bash scripts/run_benchmark.sh
```

### Test Results (RTX 4060 Ti)

| Test Suite | Passed | Total | Rate |
|------------|--------|-------|------|
| test_extract | 10 | 10 | 100% |
| test_match | 11 | 11 | 100% |
| test_homography | 8 | 8 | 100% |
| **Total** | **29** | **29** | **100%** |

### Test Coverage

| Test | What It Verifies |
|------|-----------------|
| BasicExtraction | Features detected, valid positions/scales |
| DifferentThresholds | Higher threshold = fewer features |
| DifferentOctaves | More octaves = more features |
| Reproducibility | Identical results across runs |
| ScaleUp | 2x upsampling detects more features |
| SelfMatch | Self-matching gives perfect scores |
| CrossMatch | Cross-image matching produces valid results |
| Homography | RANSAC + refinement finds inliers |
| Translation | Recovers known translation |
| Rotation | Handles 10 degree rotation |
| Scale | Handles 80% scale change |
| PGMImages | Stereo pair matching |

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

// Match two sets of SIFT features on GPU
double MatchSiftData(SiftData &data1, SiftData &data2);

// Find homography using RANSAC
double FindHomography(SiftData &data, float *homography, int *numMatches,
                      int numLoops = 1000, float minScore = 0.85f,
                      float maxAmbiguity = 0.95f, float thresh = 5.0f);
```

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
|   +-- test_homography.cpp # Geometric verification tests
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

## OpenCV- and VLFeat-Compatible Extraction

`ExtractSiftCompat` (`cudaSiftCompat.h`) extracts the same features as OpenCV's SIFT (`cv::SIFT`, OpenCV 4.8 and later with `enable_precise_upscale`) or VLFeat's (`vl_sift_*`, VLFeat 0.9.21), with its own kernels; `ExtractSift` is unchanged. It follows their code step by step: the optional doubling of the image before the first octave (OpenCV's and COLMAP's default), the same Gaussian kernels and borders, extrema, sub-pixel interpolation, orientation histograms and descriptors. The keypoints and descriptors differ from theirs only by floating-point rounding, and don't vary between runs. OpenCV's and VLFeat's modes run the same kernels, the mode and the parameters selecting each library's steps; `COLMAP()` is VLFeat's mode with COLMAP's parameters.

```cpp
#include "cudaSiftCompat.h"

SiftData data;
InitSiftData(data, 32768, true, true);
ExtractSiftCompat(data, img);                               // OpenCV's SIFT, cv::SIFT::create() defaults
ExtractSiftCompat(data, img, SiftCompatParams::VLFeat());   // VLFeat's defaults
ExtractSiftCompat(data, img, SiftCompatParams::COLMAP());   // VLFeat with COLMAP's settings
MatchSiftData(data, data2);                                 // matching and homographies as with ExtractSift
```

`SiftCompatParams` has the parameters of `cv::SIFT::create` (`nfeatures`, `nOctaveLayers`, `contrastThreshold`, `edgeThreshold`, `sigma`) and of VLFeat (`firstOctave`, `numOctaves`, `peakThreshold`, `edgeThreshold`, `maxOrientations`), COLMAP's post-processing (whole DoG levels up to `maxFeatures`, L1-root descriptors rounded to 8 bits with `rootSift`; positions are not shifted by COLMAP's +0.5, so add 0.5 to `xpos` and `ypos` for COLMAP's coordinates), and `doublePrecision`, `extendFirstOctave` and `extendFirstOctaveScales` (below); they are documented in `cudaSiftCompat.h`. The `SiftPoint` fields follow OpenCV's conventions, documented in `cudaSiftCompat.h`: `scale` is sigma (half of OpenCV's `KeyPoint::size`) and `orientation` is OpenCV's `KeyPoint::angle` in degrees. The input image has values 0 to 255, as for `ExtractSift`. Temporary memory is kept between calls; `FreeSiftCompatMemory()` releases it.

`doublePrecision` (false by default) sets the precision of the refinement, orientations and descriptors. VLFeat computes them in double, so the float default deviates from its code; it is faster on GPUs with slow double precision, and the features are essentially the same: on HPatches, 0 of VLFeat's 789,016 keypoints and 2 of COLMAP's 1,555,638 are missing in float, none in double. Where it is faster and the features stay essentially the same, the kernels multiply by reciprocals where the libraries divide. OpenCV computes in float: there, double is a deviation from OpenCV.

`extendFirstOctave` (with `firstOctave = 0`) is an alternative to doubling the image, which none of the three libraries has. It adds to the first octave the Gaussian levels below `sigma` that the doubled image would have, at sigma · 2^(−j/S) for j = 1..S (S = `nOctaveLayers`), blurred from the input image (assumed blurred by 0.5, `SIFT_INIT_SIGMA`) at the original resolution. The first octave then has 2S detection scales instead of S, down to half the scale; `sigma`, the thresholds and the other octaves are unchanged, so the keypoints are those of `firstOctave = 0` plus finer ones. `extendFirstOctaveScales` keeps only the coarsest of the added scales. On HPatches and on the synthetic pairs of `eval_hpatches`, the extension gives about the same homography accuracy as doubling, in half the time.

On a Tesla T4, for `data/img1.png` (1920x1080), including the copy of the features to the host:

| | Extraction | Features |
|---|---|---|
| `ExtractSift` (5 octaves, `initBlur` 1, `thresh` 3) | 1.5 ms | 1931 |
| `ExtractSiftCompat`, OpenCV | 10.8 ms | 6157 |
| `ExtractSiftCompat`, OpenCV, `firstOctave` 0 and `extendFirstOctave` | 5.5 ms | 6085 |
| `ExtractSiftCompat`, VLFeat | 5.9 ms | 4750 |
| `ExtractSiftCompat`, VLFeat, `doublePrecision` | 8.8 ms | 4750 |
| `ExtractSiftCompat`, COLMAP | 13.8 ms | 8385 |
| OpenCV 4.13 on the CPU (Xeon Platinum 8259CL), 1 thread | 649 ms | 6157 |
| VLFeat 0.9.21 on the CPU, 1 thread | 1.1 s | 4750 |
| COLMAP's settings on VLFeat 0.9.21, CPU, 1 thread | 3.2 s | 8385 |

`ExtractSiftCompat` is slower than `ExtractSift` mostly because OpenCV and COLMAP work on the doubled image by default; at the original resolution, it takes 1.8 to 4 times as long (the libraries' exact steps, and more features). `tests/compare_sift` checks the agreement keypoint by keypoint, with OpenCV (4.8 or later), or with VLFeat (`--vlfeat`, `--colmap`) in builds with VLFeat (`-DVLFEAT_DIR=<prefix with include/vl and lib>`):

```bash
build/compare_sift data/img1.png data/img2.png
build/compare_sift --colmap data/img1.png
```

## References

- David G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," IJCV, 2004.
- Original CudaSift by Marten Bjorkman: https://github.com/Celebrandil/CudaSift

## License

MIT License -- see [LICENSE](LICENSE) for details.
