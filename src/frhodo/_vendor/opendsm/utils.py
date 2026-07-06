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

import numba
import numpy as np


def to_np_array(x):
    """
    This function converts the input value 'x' to a numpy array.

    Parameters:
    x [int, float, array]: The input value to be converted to a numpy array.

    Returns:
    numpy array: The converted numpy array.
    """
    if x is None:
        return None

    if not hasattr(x, "__len__"):
        x = [x]

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    # if ndim is 0 then convert to 1D array
    if x.ndim == 0:
        x = np.array([x])

    return np.array(x)


@numba.jit(nopython=True, cache=True)
def OoM_numba(x, method="round"):
    """
    This function calculates the order of magnitude (OoM) of each element in the input array 'x' using the specified method.

    Parameters:
    x (numpy array): The input array for which the OoM is to be calculated.
    method (str): The method to be used for calculating the OoM. It can be one of the following:
                  "round" - round to the nearest integer (default)
                  "floor" - round down to the nearest integer
                  "ceil" - round up to the nearest integer
                  "exact" - return the exact OoM without rounding

    Returns:
    x_OoM (numpy array): A float64 array, same shape as 'x', containing the OoM
    of each element. Float output so "exact" is not truncated for integer input.
    """

    x_OoM = np.empty(x.shape, dtype=np.float64)
    for i, xi in enumerate(x):
        if xi == 0.0:
            x_OoM[i] = 1.0

        elif method.lower() == "floor":
            x_OoM[i] = np.floor(np.log10(np.abs(xi)))

        elif method.lower() == "ceil":
            x_OoM[i] = np.ceil(np.log10(np.abs(xi)))

        elif method.lower() == "round":
            x_OoM[i] = np.round(np.log10(np.abs(xi)))

        else:  # "exact"
            x_OoM[i] = np.log10(np.abs(xi))

    return x_OoM
