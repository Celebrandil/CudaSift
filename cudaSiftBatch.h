//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

// Batches of ExtractSift, MatchSiftData, FindHomographyGPU and ImproveHomographyGPU
// calls.
//
// Outside a batch, each call returns after its GPU work has completed. Calls made between
// BeginSiftBatch() and EndSiftBatch() don't wait: their GPU work is queued, and EndSiftBatch()
// returns immediately, so the CPU can do other work while the batch runs, e.g. on the
// results of the previous frame:
//
//   BeginSiftBatch();
//   ExtractSift(siftData2, img2, 5, 1.0f, 3.0f);
//   MatchSiftData(siftData1, siftData2);
//   EndSiftBatch();      // returns immediately
//   ...                  // other CPU work
//   WaitSiftBatch();     // results are available
//
// The point counts stay on the GPU, and the results are copied to the host when the batch
// completes.
//
// The calls run in order, with the same results as outside a batch. Until the batch
// completes (WaitSiftBatch() or a call outside a batch, see below), the results
// (SiftData::numPts, the points, the match fields, the homographies and their counts) are
// not valid, and the images, SiftData, temporary memory, homographies and numMatches /
// numFit outputs used must not be modified or freed. With managed memory (USE_MANAGED_MEM)
// on GPUs without concurrent managed access (cudaDevAttrConcurrentManagedAccess, e.g. on
// Windows), the CPU must not access any managed memory while a batch runs.
//
// - FindHomography and FindHomographySeeded cannot be called inside a batch (they sample
//   the matches on the CPU): the program stops with an error. FindHomographyGPU can.
// - Inside a batch, MatchSiftData, FindHomographyGPU and ImproveHomographyGPU
//   return 0.0 (the time is not available).
// - BeginSiftBatch() inside an open batch stops the program with an error.
//   WaitSiftBatch() ends an open batch (EndSiftBatch) before waiting for it.
// - A call made outside a batch (ExtractSift, MatchSiftData, FindHomography,
//   FindHomographyGPU, ImproveHomographyGPU) after EndSiftBatch() first waits for
//   the batch and completes it, so its results become valid. WaitSiftBatch() still
//   returns the batch's GPU time. CPU-only functions (ImproveHomography, PrintSiftData)
//   don't wait: call WaitSiftBatch() first.
// - ExtractSift without tempMemory allocates its temporary memory in each call, and in a
//   batch frees it only when the batch completes: pass tempMemory in batches of many calls.
// - A batch is built by one thread, on one device (the current device at BeginSiftBatch),
//   and one batch is open at a time. The calls of other threads are not part of it.

#ifndef CUDASIFTBATCH_H
#define CUDASIFTBATCH_H

void BeginSiftBatch();
void EndSiftBatch();
double WaitSiftBatch();   // returns the GPU time of the batch in ms

#endif
