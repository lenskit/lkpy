# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from hypothesis import settings


def test_hypothesis_env():
    pn = settings.get_current_profile_name()
    print("active profile:", pn)
    prof = settings.get_profile(pn)

    print("hypothesis deadline:", prof.deadline)
    print("profiles:", list(settings._profiles.keys()))
