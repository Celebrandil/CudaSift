//********************************************************//
// OpenCV- and VLFeat-compatible SIFT extraction          //
//              Apple Metal backend                       //
//********************************************************//

// The host code of ExtractSiftCompat on Metal, following the host code of cudaSiftCompat.cu step
// by step. Not compiled on its own: cudaSiftMetal.mm includes it at its end (as cudaSiftH.cu
// includes cudaSiftD.cu), for its Metal context, command encoding and buffers. The kernels
// (cudaSiftCompat.metal) are compiled in their own library, without fast math; the host code
// that doesn't depend on the backend is in cudaSiftCompatHost.h. Apple GPUs have
// no double precision, so doublePrecision isn't available, and the values that the CUDA code
// keeps in double are floats; the features are essentially the same.

#include "cudaSiftCompat.h"
#include "cudaSiftCompatShared.h"
#include "cudaSiftCompatHost.h"
#include "cudaSiftCompatMetalSource.h"

namespace {

// The state of ExtractSiftCompat: its kernels and its temporary memory, kept between calls
// (FreeSiftCompatMemory releases it)
struct CompatState {
  id<MTLLibrary> library = nil;
  id<MTLBuffer> expnTab = nil;                  // fast_expn's table
  id<MTLBuffer> scaleSpace = nil;               // the temporary images and the levels
  id<MTLBuffer> cands = nil, kpts = nil;        // candidates and keypoints
  id<MTLBuffer> counters = nil;                 // their counters
  id<MTLBuffer> dispatchArgs = nil;             // the orientations' indirect dispatch (OrientationArgs)
  int maxCands = 1 << 16, maxKpts = 1 << 16;    // grown when full
};

CompatState &GetCompatState()
{
  static CompatState st;
  return st;
}

id<MTLLibrary> CompatLibrary()
{
  CompatState &st = GetCompatState();
  if (st.library==nil) {
    @autoreleasepool {
      NSError *error = nil;
      MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
      if (@available(macOS 15.0, *))
        options.mathMode = MTLMathModeSafe;
      else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
      }
      st.library = [GetContext().device newLibraryWithSource:[NSString stringWithUTF8String:cudaSiftCompatMetalSource]
                                                     options:options error:&error];
      if (st.library==nil)
        Fail("failed to compile the ExtractSiftCompat kernels", error);
    }
  }
  return st.library;
}

// A buffer of at least `bytes`, reallocated when smaller (the previous one stays retained by
// the command buffers that use it)
void EnsureBuffer(__strong id<MTLBuffer> &buffer, size_t bytes, MTLResourceOptions options)
{
  if (buffer==nil || [buffer length]<bytes)
    buffer = NewBuffer(bytes, options);
}

// The kernel of a pipeline for a mode: name_mode
void CompatPipeline(Commands &cmd, const char *name, int mode)
{
  char full[64];
  snprintf(full, sizeof(full), "%s_%d", name, mode);
  cmd.Pipeline(full, CompatLibrary());
}

// ---------------------------------------------------------------------------------------
// The Metal backend of the pyramid builders of cudaSiftCompatHost.h: the levels are offsets (in
// floats) in the scale-space buffer, the blurs' kernels go in the arguments

struct CompatPyramidBackend {
  typedef unsigned int Level;
  Commands &cmd;
  // Places the levels of the octaves of P (its sizes, nLev and lev set) in the scale space,
  // after `extra` floats of temporary images; returns the first of these
  Level Allocate(CompatPyramidM &P, size_t extra)
  {
    size_t next = extra;
    for (int o=0;o<P.nOct;o++) {
      size_t sz = (size_t)P.p[o]*P.h[o];
      for (int i=0;i<P.nLev + P.lev[o];i++, next+=sz)
        P.off[o][i] = (unsigned int)next;
    }
    if (next>0xffffffffu)
      Fail("ExtractSiftCompat: image too large");
    EnsureBuffer(GetCompatState().scaleSpace, next*sizeof(float), MTLResourceStorageModePrivate);
    return 0;
  }
  static Level At(const CompatPyramidM &P, int o, int i) { return P.off[o][i]; }
  // The first octave's input image, upsampled or not
  void Input(int mode, Level dst, int w, int h, int p, CudaImage &img, bool upsample)
  {
    CompatImageArgs a = {};
    a.dst = dst;
    a.w = w;
    a.h = h;
    a.p = p;
    a.sw = img.width;
    a.sh = img.height;
    a.sp = img.pitch;
    a.upsample = upsample;
    CompatPipeline(cmd, "InputCompat", mode);
    cmd.Buffer(GetCompatState().scaleSpace, 0);
    cmd.Bytes(a, 1);
    cmd.Buffer(img.d_data, 2);
    cmd.Dispatch(MTLSizeMake(iDivUp(w, COMPAT_BLOCK_W), iDivUp(h, COMPAT_BLOCK_H), 1), MTLSizeMake(COMPAT_BLOCK_W, COMPAT_BLOCK_H, 1));
  }
  // Gaussian blur of src into dst, in two passes through tmp
  void Blur(int mode, Level dst, Level src, Level tmp, int w, int h, int p, const float *kernel, int radius)
  {
    CompatImageArgs a = {};
    a.w = w;
    a.h = h;
    a.p = p;
    a.radius = radius;
    auto pass = [&](const char *name, Level d, Level s, MTLSize groups, MTLSize threads) {
      char full[48];
      snprintf(full, sizeof(full), "%s_%d_%d", name, mode, radius<=TILED_MAX_RADIUS ? radius : 0);
      a.dst = d;
      a.src = s;
      cmd.Pipeline(full, CompatLibrary());
      cmd.Buffer(GetCompatState().scaleSpace, 0);
      cmd.Bytes(a, 1);
      cmd.Bytes(kernel, (2*radius + 1)*sizeof(float), 2);
      cmd.Dispatch(groups, threads);
    };
    const MTLSize rowGroups = MTLSizeMake(iDivUp(w, ROW_W), iDivUp(h, ROW_H), 1), rowThreads = MTLSizeMake(ROW_W, ROW_H, 1);
    const MTLSize colGroups = MTLSizeMake(iDivUp(w, COL_W), iDivUp(h, COL_H), 1), colThreads = MTLSizeMake(COL_W, COL_THREADS_Y, 1);
    if (mode==SIFT_COMPAT_VLFEAT) {
      pass("BlurColumnsTiled", tmp, src, colGroups, colThreads);
      pass("BlurRowsTiled", dst, tmp, rowGroups, rowThreads);
    } else {
      pass("BlurRowsTiled", tmp, src, rowGroups, rowThreads);
      pass("BlurColumnsTiled", dst, tmp, colGroups, colThreads);
    }
  }
  // Every other pixel of src (pitch sp)
  void Downsample(Level dst, int w, int h, int p, Level src, int sp)
  {
    CompatImageArgs a = {};
    a.dst = dst;
    a.src = src;
    a.w = w;
    a.h = h;
    a.p = p;
    a.sp = sp;
    cmd.Pipeline("DownsampleCompat", CompatLibrary());
    cmd.Buffer(GetCompatState().scaleSpace, 0);
    cmd.Bytes(a, 1);
    cmd.Dispatch(MTLSizeMake(iDivUp(w, COMPAT_BLOCK_W), iDivUp(h, COMPAT_BLOCK_H), 1), MTLSizeMake(COMPAT_BLOCK_W, COMPAT_BLOCK_H, 1));
  }
};

// The keypoints kept, on the host (cudaSiftCompatHost.h)
void SelectKeypoints(std::vector<OpenCVKeypoint> &kpts, const CompatPyramidM &, const CompatDetectParamsM &,
                     const SiftCompatParams &prm, unsigned int, int maxPts)
{
  SelectKeypointsOpenCV(kpts, prm, maxPts);
}

void SelectKeypoints(std::vector<VLFeatKeypointM> &kpts, const CompatPyramidM &P, const CompatDetectParamsM &dp,
                     const SiftCompatParams &prm, unsigned int numCands, int maxPts)
{
  auto level = [&](const VLFeatKeypointM &k) { return k.o*COMPAT_MAX_LEVELS + k.is - dp.s_min + P.lev[k.o]; };
  SelectKeypointsVLFeat(kpts, prm, level, (const VLFeatKeypointM *)[GetCompatState().cands contents], numCands, maxPts);
}

// ---------------------------------------------------------------------------------------
// The pipeline

template <int M, typename Keypoint>
void ExtractSiftCompatT(SiftData &siftData, CudaImage &img, const SiftCompatParams &prm)
{
  CompatState &st = GetCompatState();
  auto start = std::chrono::steady_clock::now();
  const int B = (M==SIFT_COMPAT_VLFEAT ? 1 : SIFT_IMG_BORDER);
  const int NWO = (M==SIFT_COMPAT_VLFEAT ? ORI_SG_VLFEAT : ORI_SG_OPENCV);
  const int NTD = (M==SIFT_COMPAT_VLFEAT ? DESC_NT_VLFEAT : DESC_NT_OPENCV);
  CompatPyramidM P = {};
  CompatDetectParamsM dp = {};
  std::unique_ptr<Commands> cmd(new Commands());
  CompatPyramidBackend backend = {*cmd};
  if (M==SIFT_COMPAT_VLFEAT)
    BuildPyramidVLFeat(backend, P, dp, img, prm);
  else
    BuildPyramidOpenCV(backend, P, dp, img, prm);
  // The extrema and the orientations in one command buffer, the orientations dispatched from the
  // number of candidates on the GPU; the lists grow and the kernels run again when they are full
  unsigned int *counters = (unsigned int *)[st.counters contents];
  auto extrema = [&](Commands &c) {
    for (int o=0;o<P.nOct;o++) {
      int iw = P.w[o] - 2*B, ih = P.h[o] - 2*B;
      if (iw<=0 || ih<=0)
        continue;
      for (int layer=1;layer<=dp.S + P.lev[o];layer++) {
        const int ol[2] = {o, layer};
        CompatPipeline(c, "FindExtremaCompat", M);
        c.Buffer(st.scaleSpace, 0);
        c.Bytes(P, 1);
        c.Bytes(dp, 2);
        c.Bytes(ol, sizeof(ol), 3);
        c.Buffer(st.cands, 4);
        c.Buffer(st.counters, 5);
        c.Bytes(st.maxCands, 6);
        c.Dispatch(MTLSizeMake(iDivUp(iw, COMPAT_BLOCK_W), iDivUp(ih, COMPAT_BLOCK_H), 1), MTLSizeMake(COMPAT_BLOCK_W, COMPAT_BLOCK_H, 1));
      }
    }
  };
  // The orientations of the candidates: their number from OrientationArgs (indirect) or given
  auto orientations = [&](Commands &c, int numCands) {
    const int maxOri = prm.maxOrientations;
    if (numCands<0) {
      c.Pipeline("OrientationArgs", CompatLibrary());
      c.Buffer(st.counters, 0);
      c.Bytes(st.maxCands, 1);
      c.Bytes(NWO, 2);
      c.Buffer(st.dispatchArgs, 3);
      c.DispatchThreads(MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    }
    CompatPipeline(c, "OrientationsCompat", M);
    c.Buffer(st.scaleSpace, 0);
    c.Bytes(P, 1);
    c.Bytes(dp, 2);
    c.Bytes(maxOri, 3);
    c.Buffer(st.cands, 4);
    if (numCands<0)
      c.Buffer(st.dispatchArgs, 5, 3*sizeof(unsigned int));
    else
      c.Bytes(numCands, 5);
    c.Buffer(st.kpts, 6);
    c.Buffer(st.counters, 7, sizeof(unsigned int));
    c.Bytes(st.maxKpts, 8);
    c.Buffer(st.expnTab, 9);
    if (numCands<0)
      c.DispatchIndirect(st.dispatchArgs, 0, MTLSizeMake(32*NWO, 1, 1));
    else
      c.Dispatch(MTLSizeMake(iDivUp(numCands, NWO), 1, 1), MTLSizeMake(32*NWO, 1, 1));
  };
  unsigned int numCands, numKpts;
  for (;;) {
    EnsureBuffer(st.cands, sizeof(Keypoint)*(size_t)st.maxCands, MTLResourceStorageModeShared);
    EnsureBuffer(st.kpts, sizeof(Keypoint)*(size_t)st.maxKpts, MTLResourceStorageModeShared);
    counters[0] = counters[1] = 0;
    if (!cmd)
      cmd.reset(new Commands());
    extrema(*cmd);
    orientations(*cmd, -1);
    cmd->Finish();
    cmd.reset();
    numCands = counters[0];
    if (numCands<=(unsigned int)st.maxCands)
      break;
    st.maxCands = (int)(numCands + numCands/4);
  }
  for (numKpts=counters[1];numKpts>(unsigned int)st.maxKpts;numKpts=counters[1]) {
    st.maxKpts = (int)(numKpts + numKpts/4);
    EnsureBuffer(st.kpts, sizeof(Keypoint)*(size_t)st.maxKpts, MTLResourceStorageModeShared);
    counters[1] = 0;
    Commands c;
    orientations(c, (int)numCands);
    c.Finish();
  }
  Keypoint *d_kpts = (Keypoint *)[st.kpts contents];
  std::vector<Keypoint> kpts(d_kpts, d_kpts + numKpts);
  SelectKeypoints(kpts, P, dp, prm, numCands, siftData.maxPts);
  const int numPts = (int)kpts.size();
  if (numPts>0) {
    if (siftData.d_data==NULL)
      Fail("ExtractSiftCompat: SiftData without device memory");
    std::copy(kpts.begin(), kpts.end(), d_kpts);
    const float outScale = (prm.firstOctave<0 ? 1.0f/(1 << -prm.firstOctave) : (float)(1 << prm.firstOctave));
    const int rootSift = prm.rootSift;
    Commands c;
    CompatPipeline(c, "DescriptorsCompat", M);
    c.Buffer(st.scaleSpace, 0);
    c.Bytes(P, 1);
    c.Bytes(dp, 2);
    c.Bytes(outScale, 3);
    c.Bytes(rootSift, 4);
    c.Buffer(st.kpts, 5);
    c.Bytes(numPts, 6);
    c.Buffer(siftData.d_data, 7);
    c.Buffer(st.expnTab, 8);
    c.Dispatch(MTLSizeMake(numPts, 1, 1), MTLSizeMake(NTD, 1, 1));
    c.Finish();
  }
  siftData.numPts = numPts;
  *(int *)[NumPtsBuffer(siftData) contents] = numPts;   // for the next calls of a batch
  if (siftData.h_data!=NULL && siftData.h_data!=siftData.d_data && numPts>0)
    memcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint)*numPts);
#ifdef VERBOSE
  printf("ExtractSiftCompat time =      %.2f ms, %d keypoints (%u candidates)\n", Milliseconds(start), numPts, numCands);
#else
  (void)start;
#endif
}

}  // namespace

void ExtractSiftCompat(SiftData &siftData, CudaImage &img, const SiftCompatParams &prm)
{
  ValidateCompatParams(prm);
  if (prm.doublePrecision) {
    fprintf(stderr, "ExtractSiftCompat: doublePrecision is not available on Metal (Apple GPUs have no double precision)\n");
    exit(1);
  }
  @autoreleasepool {
    Context &ctx = GetContext();
    std::lock_guard<std::mutex> callLock(ctx.callMutex);
    if (ctx.batch)
      Fail("ExtractSiftCompat can't be called in a batch (cudaSiftBatch.h)");
    CompleteBatch();
    CompatState &st = GetCompatState();
    if (st.expnTab==nil) {
      st.expnTab = NewBuffer((VL_EXPN_SZ + 1)*sizeof(float), MTLResourceStorageModeShared);
      float *tab = (float *)[st.expnTab contents];
      for (int k=0;k<VL_EXPN_SZ+1;k++)
        tab[k] = (float)std::exp(-(double)k*((double)VL_EXPN_MAX/VL_EXPN_SZ));
      st.counters = NewBuffer(2*sizeof(unsigned int), MTLResourceStorageModeShared);
      st.dispatchArgs = NewBuffer(4*sizeof(unsigned int), MTLResourceStorageModePrivate);
    }
    if (prm.mode==SIFT_COMPAT_VLFEAT)
      ExtractSiftCompatT<SIFT_COMPAT_VLFEAT, VLFeatKeypointM>(siftData, img, prm);
    else
      ExtractSiftCompatT<SIFT_COMPAT_OPENCV, OpenCVKeypoint>(siftData, img, prm);
  }
}

void FreeSiftCompatMemory()
{
  @autoreleasepool {
    Context &ctx = GetContext();
    std::lock_guard<std::mutex> callLock(ctx.callMutex);
    CompleteBatch();
    CompatState &st = GetCompatState();
    st.scaleSpace = nil;
    st.cands = nil;
    st.kpts = nil;
  }
}
