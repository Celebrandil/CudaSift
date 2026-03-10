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

## References

- David G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," IJCV, 2004.
- Original CudaSift by Marten Bjorkman: https://github.com/Celebrandil/CudaSift

## License

MIT License -- see [LICENSE](LICENSE) for details.
