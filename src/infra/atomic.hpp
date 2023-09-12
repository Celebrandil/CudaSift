//==---- atomic.hpp -------------------------------*- C++ -*----------------==//
// Copyright (C) 2023 Intel Corporation

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
//===----------------------------------------------------------------------===//

#ifndef __INFRA_ATOMIC_HPP__
#define __INFRA_ATOMIC_HPP__

#include <sycl/sycl.hpp>

namespace infra
{

  /// Atomically add the value operand to the value at the addr and assign the
  /// result to the value at addr, Int version.
  /// \param [in, out] addr The pointer to the data.
  /// \param operand The value to add to the value at \p addr.
  /// \param memoryOrder The memory ordering used.
  /// \returns The value at the \p addr before the call.
  template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
  inline T atomic_fetch_add(
      T *addr, T operand,
      sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
  {
    sycl::atomic<T, addressSpace> obj(
        (sycl::multi_ptr<T, addressSpace>(addr)));
    return sycl::atomic_fetch_add(obj, operand, memoryOrder);
  }

  /// Atomically add the value operand to the value at the addr and assign the
  /// result to the value at addr, Float version.
  /// \param [in, out] addr The pointer to the data.
  /// \param operand The value to add to the value at \p addr.
  /// \param memoryOrder The memory ordering used.
  /// \returns The value at the \p addr before the call.
  template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space>
  inline float atomic_fetch_add(
      float *addr, float operand,
      sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
  {
    static_assert(sizeof(float) == sizeof(int), "Mismatched type size");

    sycl::atomic<int, addressSpace> obj(
        (sycl::multi_ptr<int, addressSpace>(reinterpret_cast<int *>(addr))));

    int old_value;
    float old_float_value;

    do
    {
      old_value = obj.load(memoryOrder);
      old_float_value = *reinterpret_cast<const float *>(&old_value);
      const float new_float_value = old_float_value + operand;
      const int new_value = *reinterpret_cast<const int *>(&new_float_value);
      if (obj.compare_exchange_strong(old_value, new_value, memoryOrder))
        break;
    } while (true);

    return old_float_value;
  }

  /// Atomically add the value operand to the value at the addr and assign the
  /// result to the value at addr, Double version.
  /// \param [in, out] addr The pointer to the data.
  /// \param operand The value to add to the value at \p addr
  /// \param memoryOrder The memory ordering used.
  /// \returns The value at the \p addr before the call.
  template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space>
  inline double atomic_fetch_add(
      double *addr, double operand,
      sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
  {
    static_assert(sizeof(double) == sizeof(unsigned long long int),
                  "Mismatched type size");

    sycl::atomic<unsigned long long int, addressSpace> obj(
        (sycl::multi_ptr<unsigned long long int, addressSpace>(
            reinterpret_cast<unsigned long long int *>(addr))));

    unsigned long long int old_value;
    double old_double_value;

    do
    {
      old_value = obj.load(memoryOrder);
      old_double_value = *reinterpret_cast<const double *>(&old_value);
      const double new_double_value = old_double_value + operand;
      const unsigned long long int new_value =
          *reinterpret_cast<const unsigned long long int *>(&new_double_value);

      if (obj.compare_exchange_strong(old_value, new_value, memoryOrder))
        break;
    } while (true);

    return old_double_value;
  }

  /// Atomically subtract the value operand from the value at the addr and assign
  /// the result to the value at addr.
  /// \param [in, out] addr The pointer to the data.
  /// \param operand The value to substract from the value at \p addr
  /// \param memoryOrder The memory ordering used.
  /// \returns The value at the \p addr before the call.
  template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
  inline T atomic_fetch_sub(
      T *addr, T operand,
      sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
  {
    sycl::atomic<T, addressSpace> obj(
        (sycl::multi_ptr<T, addressSpace>(addr)));
    return sycl::atomic_fetch_sub(obj, operand, memoryOrder);
  }

  /// Atomically perform a bitwise AND between the value operand and the value at the addr
  /// and assign the result to the value at addr.
  /// \param [in, out] addr The pointer to the data.
  /// \param operand The value to use in bitwise AND operation with the value at the \p addr.
  /// \param memoryOrder The memory ordering used.
  /// \returns The value at the \p addr before the call.
  template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
  inline T atomic_fetch_and(
      T *addr, T operand,
      sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
  {
    sycl::atomic<T, addressSpace> obj(
        (sycl::multi_ptr<T, addressSpace>(addr)));
    return sycl::atomic_fetch_and(obj, operand, memoryOrder);
  }

  /// Atomically or the value at the addr with the value operand, and assign
  /// the result to the value at addr.
  /// \param [in, out] addr The pointer to the data.
  /// \param operand The value to use in bitwise OR operation with the value at the \p addr.
  /// \param memoryOrder The memory ordering used.
  /// \returns The value at the \p addr before the call.
  template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
  inline T atomic_fetch_or(
      T *addr, T operand,
      sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
  {
    sycl::atomic<T, addressSpace> obj(
        (sycl::multi_ptr<T, addressSpace>(addr)));
    return sycl::atomic_fetch_or(obj, operand, memoryOrder);
  }

  /// Atomically xor the value at the addr with the value operand, and assign
  /// the result to the value at addr.
  /// \param [in, out] addr The pointer to the data.
  /// \param operand The value to use in bitwise XOR operation with the value at the \p addr.
  /// \param memoryOrder The memory ordering used.
  /// \returns The value at the \p addr before the call.
  template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
  inline T atomic_fetch_xor(
      T *addr, T operand,
      sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
  {
    sycl::atomic<T, addressSpace> obj(
        (sycl::multi_ptr<T, addressSpace>(addr)));
    return sycl::atomic_fetch_xor(obj, operand, memoryOrder);
  }

  /// Atomically calculate the minimum of the value at addr and the value operand
  /// and assign the result to the value at addr.
  /// \param [in, out] addr The pointer to the data.
  /// \param operand.
  /// \param memoryOrder The memory ordering used.
  /// \returns The value at the \p addr before the call.
  template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
  inline T atomic_fetch_min(
      T *addr, T operand,
      sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
  {
    sycl::atomic<T, addressSpace> obj(
        (sycl::multi_ptr<T, addressSpace>(addr)));
    return sycl::atomic_fetch_min(obj, operand, memoryOrder);
  }

  /// Atomically calculate the maximum of the value at addr and the value operand
  /// and assign the result to the value at addr.
  /// \param [in, out] addr The pointer to the data.
  /// \param operand.
  /// \param memoryOrder The memory ordering used.
  /// \returns The value at the \p addr before the call.
  template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
  inline T atomic_fetch_max(
      T *addr, T operand,
      sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
  {
    sycl::atomic<T, addressSpace> obj(
        (sycl::multi_ptr<T, addressSpace>(addr)));
    return sycl::atomic_fetch_max(obj, operand, memoryOrder);
  }

  /// Atomically increment the value stored in \p addr if old value stored in \p
  /// addr is less than \p operand, else set 0 to the value stored in \p addr.
  /// \param [in, out] addr The pointer to the data.
  /// \param operand The threshold value.
  /// \param memoryOrder The memory ordering used.
  /// \returns The old value stored in \p addr.
  template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space>
  inline unsigned int atomic_fetch_compare_inc(
      unsigned int *addr, unsigned int operand,
      sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
  {
    sycl::atomic<unsigned int, addressSpace> obj(
        (sycl::multi_ptr<unsigned int, addressSpace>(addr)));
    unsigned int old;
    while (true)
    {
      old = obj.load();
      if (old >= operand)
      {
        if (obj.compare_exchange_strong(old, 0, memoryOrder, memoryOrder))
          break;
      }
      else
      {
        old = obj.fetch_add(1);
        break;
      }
      // else if (obj.compare_exchange_strong(old, old + 1, memoryOrder,
      //                                      memoryOrder))
      // break;
    }
    return old;
  }

  /// Atomically exchange the value at the address addr with the value operand.
  /// \param [in, out] addr The pointer to the data.
  /// \param operand The value to be exchanged with the value pointed by \p addr.
  /// \param memoryOrder The memory ordering used.
  /// \returns The value at the \p addr before the call.
  template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
  inline T atomic_exchange(
      T *addr, T operand,
      sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
  {
    sycl::atomic<T, addressSpace> obj(
        (sycl::multi_ptr<T, addressSpace>(addr)));
    return sycl::atomic_exchange(obj, operand, memoryOrder);
  }

  /// Atomically compare the value at \p addr to the value expected and exchange
  /// with the value desired if the value at \p addr is equal to the value expected.
  /// Returns the value at the \p addr before the call.
  /// \param [in, out] addr Multi_ptr.
  /// \param expected The value to compare against the value at \p addr.
  /// \param desired The value to assign to \p addr if the value at \p addr is expected.
  /// \param success The memory ordering used when comparison succeeds.
  /// \param fail The memory ordering used when comparison fails.
  /// \returns The value at the \p addr before the call.
  template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
  T atomic_compare_exchange_strong(
      sycl::multi_ptr<T, sycl::access::address_space::global_space> addr,
      T expected, T desired,
      sycl::memory_order success = sycl::memory_order::relaxed,
      sycl::memory_order fail = sycl::memory_order::relaxed)
  {
    sycl::atomic<T, addressSpace> obj(addr);
    obj.compare_exchange_strong(expected, desired, success, fail);
    return expected;
  }

  /// Atomically compare the value at \p addr to the value expected and exchange
  /// with the value desired if the value at \p addr is equal to the value expected.
  /// Returns the value at the \p addr before the call.
  /// \param [in] addr The pointer to the data.
  /// \param expected The value to compare against the value at \p addr.
  /// \param desired The value to assign to \p addr if the value at \p addr is expected.
  /// \param success The memory ordering used when comparison succeeds.
  /// \param fail The memory ordering used when comparison fails.
  /// \returns The value at the \p addr before the call.
  template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
  T atomic_compare_exchange_strong(
      T *addr, T expected, T desired,
      sycl::memory_order success = sycl::memory_order::relaxed,
      sycl::memory_order fail = sycl::memory_order::relaxed)
  {
    return atomic_compare_exchange_strong(
        sycl::multi_ptr<T, addressSpace>(addr), expected, desired, success,
        fail);
  }

}
#endif
