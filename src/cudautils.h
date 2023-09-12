// Modifications Copyright (C) 2023 Intel Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom
// the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
// OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
// OR OTHER DEALINGS IN THE SOFTWARE.

// SPDX-License-Identifier: MIT

#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <sycl/sycl.hpp>
#include <cstdio>
#include <iostream>
#include <chrono>

#ifdef WIN32
#include <intrin.h>
#endif

#define safeCall(err) __safeCall(err, __FILE__, __LINE__)
#define checkMsg(msg) __checkMsg(msg, __FILE__, __LINE__)

inline void __safeCall(int err, const char *file, const int line)
{
}

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
  int err = 0;
}

class TimerCPU
{
  static const int bits = 10;

public:
  long long beg_clock;
  float freq;
  TimerCPU(float freq_) : freq(freq_)
  { // freq = clock frequency in MHz
    beg_clock = getTSC(bits);
  }
  long long getTSC(int bits)
  {
#ifdef WIN32
    return __rdtsc() / (1LL << bits);
#else
    unsigned int low, high;
    __asm__(".byte 0x0f, 0x31"
            : "=a"(low), "=d"(high));
    return ((long long)high << (32 - bits)) | ((long long)low >> bits);
#endif
  }
  float read()
  {
    long long end_clock = getTSC(bits);
    long long Kcycles = end_clock - beg_clock;
    float time = (float)(1 << bits) * Kcycles / freq / 1e3f;
    return time;
  }
};

template <class T>
__inline__ T ShiftDown(T var, unsigned int delta, sycl::nd_item<3> item_ct1, int width = 32)
{
#if (SYCL_LANGUAGE_VERSION >= 9000)
  return sycl::shift_group_left(item_ct1.get_sub_group(), var, delta);
#else
  return __shfl_down(var, delta, width);
#endif
}

template <class T>
__inline__ T ShiftUp(T var, unsigned int delta, sycl::nd_item<3> item_ct1, int width = 32)
{
#if (SYCL_LANGUAGE_VERSION >= 9000)
  return sycl::shift_group_right(item_ct1.get_sub_group(), var, delta);
#else
  return __shfl_up(var, delta, width);
#endif
}

template <class T>
__inline__ T Shuffle(T var, unsigned int lane, sycl::nd_item<3> item_ct1, int width = 32)
{
#if (SYCL_LANGUAGE_VERSION >= 9000)
  return sycl::select_from_group(item_ct1.get_sub_group(), var, lane);
#else
  return __shfl(var, lane, width);
#endif
}

#endif
