// HIP-only shim so source that does `#include <cuda.h>` resolves on ROCm.
// See cuda_runtime.h in this directory.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
#pragma once
#include "cuda_to_hip.h"
