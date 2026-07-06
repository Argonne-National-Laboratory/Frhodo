#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2014-2025 OpenDSM contributors
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Bi-symmetric logarithmic (bisymlog) transform.

The bisymlog transform is: sign(x) · log_b(|x/C| + 1)
where C controls the linear-to-log crossover and b is the base.
"""

import numpy as np
import numba




# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, error_model="numpy", cache=True)
def _bisymlog_forward(x, C, log_base_inv):
    """Vectorized bisymlog forward: sign(x) · log10(|x/C| + 1) / log10(base)."""
    out = np.empty_like(x)
    for i in range(len(x)):
        xi = x[i]
        if xi >= 0:
            out[i] = np.log10(xi / C + 1.0) * log_base_inv
        else:
            out[i] = -np.log10(-xi / C + 1.0) * log_base_inv
    return out


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _bisymlog_inverse(y, C, base):
    """Vectorized bisymlog inverse: sign(y) · C · (base^|y| - 1)."""
    out = np.empty_like(y)
    for i in range(len(y)):
        yi = y[i]
        if yi >= 0:
            out[i] = C * (base ** yi - 1.0)
        else:
            out[i] = -C * (base ** (-yi) - 1.0)
    return out


