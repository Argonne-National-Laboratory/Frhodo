# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.

import json
from pathlib import Path


_palette_path = Path(__file__).parent / "_colors.json"
colors = json.loads(_palette_path.read_text())


def colormap(reorder_from=1, num_shift=4):
    for _ in range(num_shift):
        colors.append(colors.pop(reorder_from))

    return colors
