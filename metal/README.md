# CudaSift Metal backend

This directory holds the Apple Metal implementation of CudaSift, built with `-DUSE_METAL=ON` (see the main README). The public API (`cudaSift.h`, `cudaImage.h`) is the same as for CUDA, so applications, tests and demos build unchanged.

| File | Content |
|------|---------|
| `cudaSift.metal` | Kernels (Metal Shading Language), compiled at run time |
| `cudaSiftMetal.mm` | Host code (Objective-C++): the API of `cudaSift.h` and `cudaImage.h` |
| `cudaSiftMetalShared.h` | Kernel parameter structs and kernel geometry constants, shared by host and kernels (`static_assert`s in the kernels check the geometry of `LaplaceFindPoints` and the size of the `ImproveHomographyGPU` state) |
| `EmbedMetalSource.cmake` | Embeds `cudaSiftD.h`, `cudaSiftMetalShared.h`, `cudaSiftRandom.h` and `cudaSift.metal`, in this order, in the library, and `cudaSiftCompatShared.h` and `cudaSiftCompat.metal` for `ExtractSiftCompat` |
| `cudaSiftMetal.h` | Metal-specific API: the Metal objects behind CudaSift's data, for other Metal code, see below |
| `compat/cuda_runtime.h` | CUDA runtime subset used by the tests and demos |
| `cudaSiftCompat.metal` | Kernels of `ExtractSiftCompat` (`cudaSiftCompat.h`), compiled at run time in their own library |
| `cudaSiftCompatMetal.mm` | Host code of `ExtractSiftCompat`, included by `cudaSiftMetal.mm` (the rest of its host code, shared with CUDA, is in `cudaSiftCompatHost.h`) |
| `cudaSiftCompatShared.h` | Its constants, kernel parameter structs and kernel geometry, shared by its host code and kernels |

## Design

- **Kernels are compiled at run time** from the embedded source (`newLibraryWithSource`, fast math), the first time the library is used. Only the Xcode Command Line Tools are needed to build. To check the kernels offline, concatenate the 4 embedded files in the same order and compile the result with the Metal toolchain from Xcode:
  ```bash
  cat cudaSiftD.h metal/cudaSiftMetalShared.h cudaSiftRandom.h metal/cudaSift.metal > /tmp/cs_full.metal
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun -sdk macosx metal -std=metal3.1 -c /tmp/cs_full.metal -o /tmp/cs_full.air
  ```
- **Requirements.** At startup, the backend checks that the GPU is of family Apple7 or later (Apple M1 or later), for SIMD group matrices, supports linear filtering of 32-bit float textures, and, when each kernel is first used, that its SIMD groups are 32 wide. Otherwise the program stops with an error.
- **Unified memory.** All buffers are shared Metal buffers, so the "device" pointers of the API are also valid on the CPU. `SiftData::h_data` and `d_data` point to the same memory, and `CudaImage::Download`/`Readback` are plain copies. Device pointers, including pointers into the middle of an allocation such as the octave images in the temporary memory, are mapped back to their `MTLBuffer` and offset when kernels are encoded. `CudaImage::Allocate` frees the data of a previous `Allocate` of the same image, and only frees memory it allocated. `CudaImage::CopyToTexture` copies host images row by row. As in the CUDA code, `ExtractSift` uses at most 7 octaves, and fewer when the image is too small, with a warning.
- **Synchronous calls.** Each `ExtractSift`, `MatchSiftData`, `FindHomography`, `FindHomographySeeded`, `FindHomographyGPU` and `ImproveHomographyGPU` call encodes its kernels in one command buffer, then waits for it to complete (except in a batch, see below).
- **Memory.** Each API call drains its own autorelease pool, so a plain C++ program, which has no run loop, doesn't accumulate command buffers. A loop of 400 1080p `ExtractSift` calls stays at 154 MB, and 2000 iterations of extraction, matching, `FindHomographyGPU` and `ImproveHomographyGPU` with preallocated temporary memory stay at 169 MB. Allocation failures stop the program with "out of GPU memory (N bytes)".
- **Temporary memory** (`AllocSiftTempMemory`) holds no DoG layers, since `LaplaceFindPoints` keeps them in threadgroup memory. With 5 octaves, it takes 1.7 MB at 640x480, 10.7 MB at 1920x1080 and 42.4 MB at 3840x2160 (74.0 MB at 1920x1080 with `scaleUp`), as with the CUDA `LaplaceFindPoints`. `ExtractSift` without `tempMemory` allocates it on every call.
- **Threads.** As with CUDA, `ExtractSift`, `MatchSiftData`, `FindHomographySeeded`, `FindHomographyGPU` and `ImproveHomographyGPU` can be called from several threads: they hold one mutex until they complete, so they run one at a time. Allocating and freeing `CudaImage` and `SiftData` is thread-safe too (a mutex protects the buffer registry). Build each batch from a single thread.
- **Concurrent octaves.** The CUDA code processes the octaves one after the other: each octave's points are appended after those of the previous one, through shared point counters. The Metal backend gives each octave its own counters and candidate region, so `ExtractSift` runs in phases with a concurrent compute encoder:
  1. Downsampling, with the full-size octave's DoG and extrema detection starting at the same time.
  2. DoG and extrema detection of the other octaves.
  3. Orientations of all octaves.
  4. Descriptors of all octaves. They write each octave's points after those of the smaller octaves, which gives the same order as the CUDA code.

  If there are more than `maxPts` points, the excess points are dropped, where the CUDA code keeps overwriting the last slot. Which points are kept then varies between runs (atomic order). Below `maxPts`, the points and their values are deterministic, but their order in `SiftData` varies between runs, as with CUDA.
- **Textures.** CUDA's `tex2D` maps to a `texture2d` view of the pitched image buffer, sampled with a pixel-coordinate, linear, clamp-to-edge sampler. Rows are 512-byte aligned, which satisfies Metal's alignment for linear textures.

## Batches of calls (`cudaSiftBatch.h` in the repository root)

Each call of the CudaSift API is submitted to the GPU and waited for separately. On an M5 Pro, that costs 130–160 µs per call:
- 60–80 µs between the submission and the start of the GPU work,
- 50–60 µs between the end of the GPU work and the CPU waking up,
- about 20 µs of CPU encoding.

Most of it is driver latency, which neither polling nor spinning reduces. Batches (`cudaSiftBatch.h`, also with CUDA) save it: on Metal, calls between `BeginSiftBatch()` and `EndSiftBatch()` are recorded and submitted together.

```cpp
#include "cudaSiftBatch.h"

BeginSiftBatch();
ExtractSift(siftData2, img2, 5, 1.0f, 3.0f);   // the same calls and parameters as usual
MatchSiftData(siftData1, siftData2);
EndSiftBatch();                                // submits, returns immediately
...                                            // other CPU work
double gpuTime = WaitSiftBatch();              // results are valid
```

The calls run in order, with bit-identical results to the same calls outside a batch. `cudaSiftBatch.h` documents the rules: which calls a batch can contain, when results become valid, and how batches interact with calls outside them. `tests/test_batch` checks the results and times a frame loop.

Frame loop on an M5 Pro, where each frame is a 1080p extraction plus matching with the previous frame: **1.13 ms per frame with separate calls, 1.03 ms in one batch (−9%)**.

## Interop with other Metal code (`cudaSiftMetal.h`)

Other Metal code can read CudaSift's points and matches in place and run its work after CudaSift's, without copies or waits on the CPU. `cudaSiftMetal.h` is plain C++: the Metal objects are `void *` (cast with `(__bridge id<...>)`), not owned by the caller, and the header defines `CUDASIFT_HAS_METAL_INTEROP`.

- `CudaSiftMetalDevice()` and `CudaSiftMetalQueue()`: the device and queue that CudaSift uses. Outside a batch, CudaSift's calls return after their GPU work, and command buffers on the same queue run in order, with hazard tracking.
- `CudaSiftMetalBuffer(ptr, &offset)`: the `MTLBuffer` and byte offset of a pointer into CudaSift's memory (`SiftData::d_data`, `CudaImage::d_data`, ...), or `nullptr`.
- In a batch: `CudaSiftMetalBatchCommandBuffer()` ends the batch's compute encoder and returns its command buffer, where the caller encodes its own work; the next CudaSift call of the batch continues after it. `CudaSiftMetalNumPtsBuffer(data, &offset)` is the buffer holding the point count of `data` on the GPU, valid for that work although `SiftData::numPts` isn't yet.

```cpp
#include "cudaSiftMetal.h"

BeginSiftBatch();
ExtractSift(siftData1, img1, 5, 1.0f, 3.0f);
ExtractSift(siftData2, img2, 5, 1.0f, 3.0f);
MatchSiftData(siftData1, siftData2);
id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)CudaSiftMetalBatchCommandBuffer();
size_t ptsOffset, countOffset;
id<MTLBuffer> pts = (__bridge id<MTLBuffer>)CudaSiftMetalBuffer(siftData1.d_data, &ptsOffset);
id<MTLBuffer> count = (__bridge id<MTLBuffer>)CudaSiftMetalNumPtsBuffer(siftData1, &countOffset);
...                    // encode work into cb that reads pts and count, end its encoders
EndSiftBatch();
WaitSiftBatch();
```

`tests/test_metal_interop` checks the buffers, and work encoded into a batch between CudaSift's calls: it reads the point counts computed earlier in the batch, and the homography computed after it is the same as outside a batch.

## Homography estimation on the GPU

`FindHomographyGPU` and `ImproveHomographyGPU` (declared in `cudaSift.h`) replace `FindHomography` and `ImproveHomography` when everything should stay on the GPU, e.g. in a batch. Every computation runs on the GPU. The CPU only encodes the commands, uploads the starting homography of `ImproveHomographyGPU` when it doesn't come from a previous call of the batch, and copies back the result (9 floats and a count). `FindHomography` selects and samples the matches on the CPU.

Both backends implement the two functions (CUDA: `matching.cu`); `tests/test_homography_gpu` checks them on both.

- The Metal functions return the wall time of the call in ms, or 0.0 in a batch. The CUDA ones return the GPU time.
- `ImproveHomographyGPU` keeps the homography unchanged when an iteration has fewer than 4 inliers or the normal equations are not positive definite, as the CPU `ImproveHomography` does. `tests/test_homography_gpu` checks it ("Homography kept with fewer than 4 inliers", with 0 and 3 inliers).
- Metal has no double precision, so `ImproveHomographyGPU` uses double-float arithmetic (pairs of floats, about 48 bits), in a second library compiled without fast math, which would break its error-free transformations. Spread over 16 threadgroups per iteration, with only the non-zero terms of the equations and the Cholesky factorization in threadgroup memory, it takes about 270 µs.

| Step | `FindHomography` | `FindHomographyGPU` (Metal kernels) |
|------|------------------|----------------------------------------|
| Valid matches (score, ambiguity) | CPU | `HomographyPrepare`: compaction in increasing index order |
| Order of the valid matches | index | index (`HomographyGather`); for reproducible results, sort the points with `SortSiftData` |
| 4-point samples | CPU, `rand()` | `HomographySample`: seeded counter-based generator (`cudaSiftRandom.h`), the same code on CPU, CUDA and Metal |
| Orientation check of the samples | CPU, `SiftSameOrientation` in double | `HomographySample`, `SiftSameOrientation` in float |
| Homographies, inlier counts among the valid matches | `ComputeHomographies`, `TestHomographies` | `ComputeHomographies`, `TestHomographiesGPU` (4 hypotheses per thread) |
| Best hypothesis | CPU, among the samples that pass the check | `HomographyBest`: first maximum, as on the CPU |
| Refinement | `ImproveHomography` (CPU, double) | `ImproveHomographyGPU`: `ImproveAccumulate` and `ImproveSolve` per iteration, double-float arithmetic |

How much the results change (`data/img1.png`, `data/img2.png`, 10000 hypotheses, M5 Pro, `tests/test_homography_gpu`):
- `FindHomographyGPU` finds 126 matches over seeds 1–10, as `FindHomography` (identical over 3 runs). Refined with `ImproveHomography`, both give the same homography and 1009 inliers.
- `ImproveHomographyGPU` and `ImproveHomography` give the same 1009 inliers. The homographies differ by 3.8e-6 px on average over the inliers (maximum 0.00027 px), and `match_error` by at most 0.00022 px.
- In one batch: 126 matches and 1009 inliers, identical to the separate calls.
- With the same seed and the same matches, the result is identical between runs.

## ExtractSiftCompat

`ExtractSiftCompat` (`cudaSiftCompat.h`: OpenCV's, VLFeat's and COLMAP's SIFT) runs on Metal too, with the same results as with CUDA up to floating-point rounding. Its kernels (`cudaSiftCompat.metal`) port those of `cudaSiftCompat.cu`: one kernel per stage for the three settings, templated on the mode (instantiated under `[[host_name]]`), with each library's version of a step as an overload on its keypoint type (MSL has no `if constexpr`). The host code that doesn't depend on the backend (validation, pyramids, keypoint selection) is shared with CUDA in `cudaSiftCompatHost.h`.

- **Arithmetic.** The kernels are compiled in their own library, without fast math. There, divisions are correctly rounded (except where the CUDA code uses a fast division or a reciprocal, which these kernels do too), and the products and sums that the CUDA code rounds separately with `__fmul_rn`, `__fadd_rn` and `__fsub_rn` are written with `MulRN`, `AddRN` and `SubRN`, which `#pragma METAL fp contract(off)` keeps from being fused into multiply-adds. `sqrt` isn't correctly rounded without fast math either, so `precise::sqrt` replaces `__fsqrt_rn`. Apple GPUs have no double precision: `doublePrecision` stops the program with a message (builds define `CUDASIFT_COMPAT_NO_DOUBLE`), and the few values that the CUDA float path computes in double are floats.
- **Histograms.** The CUDA code sums the descriptor histograms per warp, adding the lanes with the same bin together (`__match_any_sync`). Emulated with SIMD group votes, that loop over distinct bins made the descriptors 3 to 4 times slower. Here groups of 8 lanes share a histogram and add to it one after the other, in lane order, then the histograms are summed in a fixed order: the results still don't vary between runs, with a different summation order than CUDA's.
- **Blurs.** Two passes on tiles in threadgroup memory, templated on the radius (1 to 16, and one generic kernel for larger radii) so that the loops unroll. The fused blur of the CUDA code needs too much threadgroup memory per threadgroup on Apple GPUs, and is slower here.
- **Calls.** The extrema and the orientations run in one command buffer (the orientations are dispatched from the number of candidates on the GPU), then the keypoints are selected on the host and the descriptors computed in a second one. Calls hold the backend's call lock, so they run one at a time, and can't be part of a batch.

On HPatches (the first and fourth images of its 116 sequences, 232 images; `tests/compare_sift`), with OpenCV 4.14 and VLFeat 0.9.21 built without fused multiply-adds as its x86 builds are, 0 of OpenCV's 1,087,351 keypoints are missing (1 extra), 1 of VLFeat's 789,016 and 2 of COLMAP's 1,555,638, with median descriptor distances of 4e-7 or less, as with CUDA. With VLFeat built by clang on Apple silicon, which fuses multiply-adds, VLFeat's own keypoints differ slightly (4753 instead of 4750 on `data/img1.png`). On the evaluation of upstream PR [#103](https://github.com/Celebrandil/CudaSift/pull/103), the numbers of features, repeatabilities and matching scores are those of CUDA in every setting.

Speed, `data/img1.png` (1920x1080), M5 Pro against the Tesla T4 of the CUDA numbers, including the copy of the features to the host:

| Settings | Doubled image | Original resolution | Original resolution + extension |
|---|---|---|---|
| OpenCV | 11.1 ms (T4: 10.8) | 2.7 ms (2.7) | 4.9 ms (5.5) |
| VLFeat | 26.6 ms (35.2) | 4.7 ms (5.9) | 11.4 ms (17.3) |
| COLMAP | 10.1 ms (13.8) | 2.6 ms (3.2) | 4.8 ms (6.9) |

## Kernels

Most kernels are line-by-line ports of the CUDA kernels in `cudaSiftD.cu` and `matching.cu`. Warp shuffles and votes map directly to SIMD group functions: Apple SIMD groups are 32 wide, like CUDA warps, and keep the same "unchanged value outside the range" semantics. `__syncthreads` maps to `threadgroup_barrier`, `__constant__` memory to constant buffers, and `d_PointCounter` to a buffer of `atomic_uint`.

| CUDA kernel(s) | Metal kernel | Notes |
|----------------|--------------|-------|
| `LowPassBlock` | `LowPassBlock` | Port |
| `ScaleDownDirect` | `ScaleDownDirect` | Port: one thread per output pixel |
| `ScaleUp` | `ScaleUp` | Port |
| `LaplaceFindPoints` | `LaplaceFindPoints` | Port, with the same tile geometry, see below |
| `ComputeOrientationsCONST` | `ComputeOrientationsSimd` | One SIMD group per point, see below |
| `ExtractSiftDescriptorsGather` | `ExtractSiftDescriptorsGather` | Same gathered histogram, different work split, see below |
| `RescalePositions` | `RescalePositions` | Port |
| `FindMaxCorr11` + `MergeMaxCorr11` | `FindMaxCorr12` + `MergeMaxCorr12` | SIMD group matrices, see below |
| | `ClearCounters`, `CountPoints`, `MatchGrid` | Point counts on the GPU, for batches |
| `ComputeHomographies`, `TestHomographies` | same | Port, with regular multiplications instead of `__fmul_rz` |
| `FindHomographyGPU` (`matching.cu`) | `HomographyPrepare`, `HomographyRank`, `HomographySample`, `TestHomographiesGPU`, `HomographyBest` | Port. The inlier test does 4 hypotheses per thread (125 vs 183 µs), with the same results |
| `ImproveHomographyGPU` (`matching.cu`) | `ImproveInit`, `ImproveAccumulate`, `ImproveSolve`, `ImproveFinal`, `ImproveCount` | Double-float arithmetic instead of double; compiled without fast math |

The Metal kernels include the same fixes as the CUDA code: the bug fixes of [#99](https://github.com/Celebrandil/CudaSift/pull/99) and [#100](https://github.com/Celebrandil/CudaSift/pull/100), and the last partial tile of upstream PR [#96](https://github.com/Celebrandil/CudaSift/pull/96) (which `FindMaxCorr11` also covers). They also include upstream PR [#98](https://github.com/Celebrandil/CudaSift/pull/98) (orientations of the last octave counted). As with CUDA, histograms are summed in a fixed order, so the points and their values are deterministic (their order isn't).

## Metal-specific optimizations

Timings are GPU times per call on an Apple M5 Pro (20 GPU cores), for `data/img1.png` resized to 1920x1080 (5 octaves, threshold 3.0), with GPU timestamps at encoder boundaries, one encoder per dispatch. They were measured on an earlier revision, at 2060 points (2076 now).

| Kernel(s), per ExtractSift | Time |
|----------------------------|-----:|
| `LaplaceFindPoints` (5 octaves) | 450 µs |
| Orientations (5 octaves) | 66 µs |
| Descriptors (5 octaves) | 104 µs |
| `LowPassBlock` | 69 µs |
| `ScaleDownDirect` (4 octaves) | 54 µs |
| **ExtractSift** (wall time) | **0.76 ms** |
| **MatchSiftData**, 2060 x 2060 points (wall time) | **0.30 ms** |

- **Fused DoG and extrema detection** (`LaplaceFindPoints`). The 7 DoG layers of an octave are computed tile by tile in threadgroup memory, in a ring of 3 layers, and never written to device memory. Each threadgroup covers 30 x 32 pixels, the area of 4 `FindPointsMultiNew` blocks, and reproduces their logic, including the limit of 32 candidates per block. The filters use the same operations, order and edge clamping as `LaplaceMultiMem`. The CUDA `LaplaceFindPoints` uses the same geometry (`FUSED_FY`, `FUSED_NT`).
- **Orientations** (`ComputeOrientationsSimd`). The CUDA kernel uses one 121-thread block per point, with a block barrier between its steps and the histogram in shared memory. The Metal kernel uses one SIMD group per point (8 points per threadgroup) and no threadgroup barriers. Each lane computes 4 of the 121 gradient samples. Then lane b sums bin b over all samples, in order, fetching them with `simd_shuffle`. Smoothing, peak search and interpolation also use shuffles.
- **Descriptors** (`ExtractSiftDescriptorsGather`). Threadgroup memory has no float atomics, so the 128-bin histogram is gathered instead of scattered. Each thread first computes the 4 spatially weighted gradients of its 2 samples. Then the 8 threads of a spatial cell each take one row of the cell's 8x8 sample window, accumulate all 8 angle bins in registers, and add up the 8 partial histograms with `simd_shuffle_xor`. That is 16 threadgroup memory loads per thread, where a direct gather (each thread scanning 64 samples for its own bin, as in the CUDA `ExtractSiftDescriptorsGather`) needs 192. Angle bins use 4/π and wrap bin 8 to 0, as in the CUDA code.
- **Concurrent octaves.** Every dispatch of the three smallest octaves takes 10–30 µs whatever its size, so running independent work concurrently pays off. Processing the octaves concurrently (see Design) reduced the wall time of ExtractSift by 10.9% at 640x480, 5.3% at 1080p and 1.8% at 4K. Starting the full-size octave together with the downsampling of the smaller ones, with barriers on the image memory only, gained another 1.0% at 1080p.
- **Matching** (`FindMaxCorr12`). This is a matrix product followed by a best/second-best reduction, computed with `simdgroup_float8x8` multiply-accumulates. Each threadgroup correlates 64 sift1 points with one of 64 chunks of the sift2 points, 64 x 64 points at a time, with 8 SIMD groups computing 16 x 32 correlations each as 2 x 4 matrices. Per-chunk best and second-best scores are merged by `MergeMaxCorr12`. With a fixed number of chunks, the scratch memory doesn't depend on the number of sift2 points, which batches need. 16 or 32 chunks were 5–13% slower, 64 is as fast as one threadgroup per 64 x 64 tile. A port of an 8x8 register-tile kernel (8x8 scalar correlations per thread) took 1.8 ms.

## Lessons learned

- **Avoid register spilling: keep at most 8 accumulator matrices per SIMD group.** The M5 Pro reaches 7.7 TFLOPS with scalar FMAs and 8.2 TFLOPS with SIMD group matrices, but only with few accumulators. With 16 `simdgroup_float8x8` accumulators per SIMD group, throughput drops to 0.64 TFLOPS, against 8.2 TFLOPS with 8. Large per-thread arrays, such as an 8 x 8 register tile of correlations, or a 17-value window in `LaplaceFindPoints` (2.4x slower), behave the same way.
- **Threadgroup memory is limited to 32 KB** per threadgroup. `LaplaceFindPoints` uses 19 KB.
- **Drain an autorelease pool in each call.** A plain C++ program has no run loop, so without a pool per call, every call leaks its command buffer and everything it retains: about 97 MB per 1080p `ExtractSift` call.
- **Each submission has a fixed latency** of 130–160 µs (see Batches), which polling a shared event, spinning on the command buffer status, blocking event waits or warm-up submissions don't reduce. Batches are the only way to avoid it.
- **fp16 doesn't help on the M5 Pro.** Half precision matrix operations run at the fp32 rate (8.0 vs 7.9 TFLOPS), and scalar half FMAs are only 1.3x faster (10.0 vs 7.7 TFLOPS). Half precision in matching or in the DoG layers gave no measurable speedup and changed the results.

## On other Macs (M1, ...)

The kernels were tuned on an M5 Pro. Other Apple silicon GPUs should give the same results, but the best tile sizes, the register spilling limit and the gains may differ, especially on M1 and M2, whose GPUs predate the register and memory changes of M3 and later. To test on another Mac:

1. Install the Xcode Command Line Tools (`xcode-select --install`), [Homebrew](https://brew.sh), CMake and OpenCV: `brew install cmake opencv@4`.
2. Clone the repository.
3. Build on that Mac, from the repository root. By default, binaries target the macOS version they are built on, so binaries built on a newer Mac may not run:
   ```bash
   cmake -S . -B build-metal -DUSE_METAL=ON -DCMAKE_BUILD_TYPE=Release -DVERBOSE_OUTPUT=OFF \
     -DOpenCV_DIR=$(brew --prefix opencv@4)/lib/cmake/opencv4
   cmake --build build-metal -j
   ```
4. From the repository root (the tests read `data/`), run the tests and the benchmark with `bash scripts/run_benchmark.sh build-metal`. All tests should pass.
5. Report the outputs of the tests and the benchmark, along with the GPU (`system_profiler SPDisplaysDataType | grep -E "Chipset|Cores"`) and the macOS version (`sw_vers -productVersion`).

## Validation

All tests pass, including the regression tests of the bug-fix PRs.

Compared with the CUDA code with the same fixes (including upstream PRs [#96](https://github.com/Celebrandil/CudaSift/pull/96) and [#98](https://github.com/Celebrandil/CudaSift/pull/98)), on a Tesla T4, with identical input images:
- 99.5–99.9% of the keypoints are identical, depending on the image size: position within 0.01 pixel, same scale, orientation within 1 degree. Point counts differ by at most 1.
- The others are at the same positions, with a different orientation where the orientation histogram has two similar peaks. This comes from the different texture filtering precision.
- The median descriptor difference (L2) is 3.2e-4, and 99.85% of the matches have the same target.

Performance on the M5 Pro (`tests/benchmark`, 5 octaves, threshold 3.0):

| Resolution | Features | Extract (ms) | Match (ms) | Total (ms) | FPS | Homography (ms) | GPU homography (ms) |
|------------|----------|-------------|------------|------------|-----|-----------------|---------------------|
| VGA        |      737 |        0.36 |       0.16 |       0.53 | 1891 |            0.84 |                0.62 |
| 720p       |     1264 |        0.51 |       0.20 |       0.75 | 1325 |            1.04 |                0.64 |
| SXGA       |     1444 |        0.56 |       0.25 |       0.89 | 1129 |            1.06 |                0.64 |
| 1080p      |     2076 |        0.78 |       0.30 |       1.20 | 835 |            1.17 |                0.65 |
| 1440p      |     2392 |        1.17 |       0.36 |       1.75 | 571 |            1.15 |                0.65 |
| 4K UHD     |     2983 |        2.37 |       0.51 |       3.42 | 292 |            1.24 |                0.67 |

- Total is upload + extraction + matching, and FPS is derived from it.
- Homography is `FindHomography` + `ImproveHomography`, and GPU homography is `FindHomographyGPU` + `ImproveHomographyGPU`. Both use 10000 hypotheses and 5 refinement iterations, on the self-match of the benchmark, and are measured as wall time.
- The GPU functions are 1.3–1.9x faster. The homography columns were measured with the RANSAC fixes of [#100](https://github.com/Celebrandil/CudaSift/pull/100). The CPU functions slow down more as the number of matches grows, because match selection and all of `ImproveHomography` run on the CPU.
- Medians of 3 runs. Benchmark timings vary by about 2% from run to run.
