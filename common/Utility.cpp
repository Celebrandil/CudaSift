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

#include <iostream>

#include "Utility.h"

using namespace Utility;

void Utility::RunDataVerification(const int threshold, const float matchPercentage)
{
    printf("Performing data verification \n");
    switch (threshold)
    {
    case 1:
        if (matchPercentage > 20.0f && matchPercentage < 30.0f)
        {
            printf("Data verification is SUCCESSFUL. \n\n");
        }
        else
        {
            printf("Data verification FAILED. \n\n");
        }
        break;
    case 2:
        if (matchPercentage > 26.0f && matchPercentage < 38.0f)
        {
            printf("Data verification is SUCCESSFUL. \n\n");
        }
        else
        {
            printf("Data verification FAILED. \n\n");
        }
        break;
    case 3:
        if (matchPercentage > 35.0f && matchPercentage < 45.0f)
        {
            printf("Data verification is SUCCESSFUL. \n\n");
        }
        else
        {
            printf("Data verification FAILED. \n\n");
        }
        break;
    case 4:
        if (matchPercentage > 40.0f && matchPercentage < 50.0f)
        {
            printf("Data verification is SUCCESSFUL. \n\n");
        }
        else
        {
            printf("Data verification FAILED. \n\n");
        }
        break;
    default:
        printf("Threshold values should be in the range [1, 4]. \n\n");
    }
}
