// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University
// Copyright (C) 2023-2025 Drexel University
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

import { WorkflowStep } from "@lenskit/typeline/github";

export function checkoutStep(depth: number = 0): WorkflowStep {
  return {
    name: "🛒 Checkout",
    uses: "actions/checkout@v4",
    with: { "fetch-depth": depth },
  };
}
