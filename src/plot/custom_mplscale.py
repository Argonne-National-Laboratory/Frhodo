# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.

import matplotlib as mpl
from matplotlib import scale as mplscale

import numpy as np
from calculate.convert_units import Bisymlog


class AbsoluteLogScale(mplscale.LogScale):
    name = "abslog"

    def __init__(self, axis, **kwargs):
        super().__init__(axis, **kwargs)

    def get_transform(self):
        return self.AbsLogTransform()

    class AbsLogTransform(mpl.transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mpl.transforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            masked = np.ma.masked_where(a == 0, a)
            if masked.mask.any():
                # ignore any divide by zero errors, 0 shouldn't exist due to mask
                with np.errstate(divide="ignore"):
                    return np.log10(np.abs(masked))
            else:
                return np.log10(np.abs(a))

        def inverted(self):  # link to inverted transform class
            return AbsoluteLogScale.InvertedAbsLogTransform()

    class InvertedAbsLogTransform(mpl.transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mpl.transforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            return np.power(10, a)

        def inverted(self):
            return AbsoluteLogScale.AbsLogTransform()


class BiSymmetricLogScale(mplscale.ScaleBase):
    name = "bisymlog"

    def __init__(self, axis, **kwargs):
        def isNum(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        if "C" in kwargs and isNum(
            kwargs["C"]
        ):  # Maybe should check for a specific value?
            self.set_C(float(kwargs["C"]))
            del kwargs["C"]
        else:
            self.set_C(0)

        super().__init__(axis, **kwargs)
        self.subs = np.arange(1, 10)

    def get_transform(self):
        return self.BiSymLogTransform(self.C)

    def set_C(self, C):
        if C == 0:  # Default C value
            self.C = 1 / np.log(1000)
        else:
            self.C = C

    def set_default_locators_and_formatters(self, axis):
        class Locator(mpl.ticker.SymmetricalLogLocator):
            def __init__(self, transform, C, subs=None):
                if subs is None:
                    self._subs = None
                else:
                    self._subs = subs

                self._base = transform.base
                self.numticks = "auto"
                self.C = C
                self.transform = transform.transform
                self.inverse_transform = transform.inverted().transform

            def tick_values(self, vmin, vmax):
                def OoM(x):
                    x[x == 0] = np.nan
                    return np.floor(np.log10(np.abs(x)))

                if self.numticks == "auto":
                    if self.axis is not None:
                        numticks = np.clip(self.axis.get_tick_space(), 2, 9)
                    else:
                        numticks = 9
                else:
                    numticks = self.numticks

                if vmax < vmin:
                    vmin, vmax = vmax, vmin

                vmin_scale = self.transform(vmin)
                vmax_scale = self.transform(vmax)

                # quicker way would only operate on min, second point and max
                scale_ticklocs = np.linspace(vmin_scale, vmax_scale, numticks)
                raw_ticklocs = self.inverse_transform(scale_ticklocs)
                raw_tick_OoM = OoM(raw_ticklocs)

                zero_OoM = np.nanmin(raw_tick_OoM)  # nearest to zero
                min_OoM = raw_tick_OoM[0]
                max_OoM = raw_tick_OoM[-1]
                min_dist = scale_ticklocs[2] - scale_ticklocs[1]

                if vmin <= 0 and 0 <= vmax:
                    if min_dist > self.transform(10**zero_OoM):
                        min_dist = self.inverse_transform(min_dist)
                        zero_OoM = np.round(np.log10(np.abs(min_dist)))

                    if vmin == 0:
                        numdec = np.abs(max_OoM - 2 * zero_OoM)
                    elif vmax == 0:
                        numdec = np.abs(min_OoM - 2 * zero_OoM)
                    else:
                        numdec = np.abs(min_OoM + max_OoM - 2 * zero_OoM)

                    stride = 1
                    while numdec // stride + 2 > numticks - 1:
                        stride += 1

                    if vmin < 0:
                        neg_dec = np.arange(zero_OoM, min_OoM + stride, stride)
                        neg_sign = np.ones_like(neg_dec) * -1
                        idx_zero = len(neg_dec)
                    else:
                        neg_dec = []
                        neg_sign = []
                        idx_zero = 0

                    if vmax > 0:
                        pos_dec = np.arange(zero_OoM, max_OoM + stride, stride)
                        pos_sign = np.ones_like(pos_dec)
                    else:
                        pos_dec = []
                        pos_sign = []
                        idx_zero = len(neg_dec)

                    decades = np.concatenate((neg_dec, pos_dec))
                    sign = np.concatenate((neg_sign, pos_sign))

                    ticklocs = np.multiply(sign, np.power(10, decades))

                    # insert 0
                    idx = ticklocs.searchsorted(0)
                    ticklocs = np.concatenate(
                        (ticklocs[:idx][::-1], [0], ticklocs[idx:])
                    )

                else:
                    numdec = np.abs(max_OoM - min_OoM)
                    stride = 1
                    while numdec // stride + 2 > numticks - 1:
                        stride += 1

                    sign = np.sign(vmin_scale)

                    if sign == -1:
                        decades = np.arange(max_OoM, min_OoM + stride, stride)
                    else:
                        decades = np.arange(min_OoM, max_OoM + stride, stride)

                    ticklocs = sign * np.power(10, decades)

                    scale_ticklocs = self.transform(ticklocs)
                    diff = np.diff(scale_ticklocs)
                    n = 0
                    for i in range(len(scale_ticklocs) - 1):
                        if min_dist * 0.25 > np.abs(diff[i]):
                            ticklocs = np.delete(ticklocs, n)
                        else:
                            n += 1

                # Add the subticks if requested
                if self._subs is None or stride != 1:
                    subs = np.array([1.0])
                else:
                    subs = np.asarray(self._subs)

                if len(subs) > 1 or subs[0] != 1.0:
                    decades = ticklocs
                    ticklocs = []
                    for decade in decades:
                        if decade == 0:
                            ticklocs.append(decade)
                        else:
                            ticklocs.extend(subs * decade)

                return self.raise_if_exceeds(np.array(ticklocs))

        axis.set_major_locator(Locator(self.get_transform(), self.C))
        axis.set_major_formatter(mpl.ticker.LogFormatterMathtext())
        axis.set_minor_locator(Locator(self.get_transform(), self.C, self.subs))
        axis.set_minor_formatter(mpl.ticker.NullFormatter())

    class BiSymLogTransform(mpl.transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, C, base=10):
            mpl.transforms.Transform.__init__(self)
            self.base = base
            self.C = C
            self.bisymlog = Bisymlog(C=C, scaling_factor=2.0, base=base)

        def transform_non_affine(self, x):
            return self.bisymlog.transform(x)

        def inverted(self):  # link to inverted transform class
            return BiSymmetricLogScale.InvertedBiSymLogTransform(self.C)

    class InvertedBiSymLogTransform(mpl.transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, C, base=10):
            mpl.transforms.Transform.__init__(self)
            self.base = base
            self.C = C
            self.bisymlog = Bisymlog(C=C, scaling_factor=2.0, base=base)

        def transform_non_affine(self, x):
            return self.bisymlog.invTransform(x)

        def inverted(self):
            return BiSymmetricLogScale.BiSymLogTransform(self.C)
