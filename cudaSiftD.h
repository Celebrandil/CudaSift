//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#ifndef CUDASIFTD_H
#define CUDASIFTD_H

#define NUM_SCALES      5

// Maximum number of octaves. Octaves are numbered from 1 to MAX_OCTAVES: d_PointCounter
// (counters 2*octave-1 .. 2*octave+1) and d_LaplaceKernel (indexed by octave) are sized
// for them.
#define MAX_OCTAVES     7

// Scale down thread block width
#define SCALEDOWN_W    64 // 60 

// Scale down thread block height
#define SCALEDOWN_H    16 // 8

// Scale up thread block width
#define SCALEUP_W      64

// Scale up thread block height
#define SCALEUP_H       8

// Find point thread block width
#define MINMAX_W       30 //32 

// Find point thread block height
#define MINMAX_H        8 //16 
 
// Laplace thread block width
#define LAPLACE_W     128 // 56

// Laplace rows per thread
#define LAPLACE_H       4

// Number of laplace scales
#define LAPLACE_S   (NUM_SCALES+3)

// Laplace filter kernel radius
#define LAPLACE_R       4

#define LOWPASS_W      24 //56
#define LOWPASS_H      32 //16
#define LOWPASS_R       4

//====================== Number of threads ====================//
// LowPassBlock:                 (LOWPASS_W + 2*LOWPASS_R) x 4
// ScaleDownDirect:              32 x 8
// LaplaceFindPoints:            FUSED_NT = 288 (cudaSiftD.cu)
// ComputeOrientationsCONST:     11*11 = 121
// ExtractSiftDescriptorsGather: 16 x 8 = 128

//====================== Number of blocks ====================//
// LowPassBlock:                 (width/LOWPASS_W) * (height/LOWPASS_H)
// ScaleDownDirect:              (width/2/32) * (height/2/8)
// LaplaceFindPoints:            (width/MINMAX_W) * (height/(MINMAX_H*FUSED_FY))
// ComputeOrientationsCONST:     512, each loops over the points
// ExtractSiftDescriptorsGather: 512, each loops over the points

#endif
