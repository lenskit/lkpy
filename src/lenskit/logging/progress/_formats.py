# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# extra comment to make license headers happy
def field_format(name: str, fs: str | None):
    if fs:
        return "{%s:%s}" % (name, fs)
    else:
        return "{%s}" % (name,)
