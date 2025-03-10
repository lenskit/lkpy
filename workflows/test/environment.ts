// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University
// Copyright (C) 2023-2025 Drexel University
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

import { script } from "../lib/script.ts";

export function inspectStep() {
  return {
    name: "🔍 Inspect environment",
    run: script("lenskit doctor"),
  };
}
