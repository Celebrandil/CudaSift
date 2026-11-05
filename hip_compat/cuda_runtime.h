// HIP-only shim so source that does `#include <cuda_runtime.h>` resolves on
// ROCm (where no CUDA headers exist). This directory is added to the include
// path ONLY for the HIP build; the NVIDIA build uses the real CUDA header.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
#pragma once
#include "cuda_to_hip.h"
