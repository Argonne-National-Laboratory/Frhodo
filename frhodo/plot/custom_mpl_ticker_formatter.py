# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import matplotlib as mpl
import numpy as np


class MathTextSciSIFormatter(mpl.ticker.ScalarFormatter): # format to SI OoM
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_powerlimits([0, 3])
    
    def _set_order_of_magnitude(self):  # modified from matplotlib
        # if scientific notation is to be used, find the appropriate exponent
        # if using an numerical offset, find the exponent after applying the
        # offset. When lower power limit = upper <> 0, use provided exponent.
        if not self._scientific:
            self.orderOfMagnitude = 0
            return
        if self._powerlimits[0] == self._powerlimits[1] != 0:
            # fixed scaling when lower power limit = upper <> 0.
            self.orderOfMagnitude = self._powerlimits[0]
            return
        # restrict to visible ticks
        vmin, vmax = sorted(self.axis.get_view_interval())
        locs = np.asarray(self.locs)
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        locs = np.abs(locs)
        if not len(locs):
            self.orderOfMagnitude = 0
            return
        if self.offset:
            oom = np.floor(np.log10(vmax - vmin))
        else:
            if locs[0] > locs[-1]:
                val = locs[0]
            else:
                val = locs[-1]
            if val == 0:
                oom = 0
            else:
                oom = np.floor(np.log10(val))
        if oom <= self._powerlimits[0]: # round down oom to nearest 3 to match SI
            self.orderOfMagnitude = np.floor(oom/3)*3
        elif oom >= self._powerlimits[1]:
            self.orderOfMagnitude = np.floor(oom/3)*3
        else:
            self.orderOfMagnitude = 0