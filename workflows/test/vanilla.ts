// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University
// Copyright (C) 2023-2025 Drexel University
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

import { WorkflowStep } from "@lenskit/typeline/github";

import { pythonVersionString, TestJobSpec } from "./spec.ts";
import { script } from "../lib/script.ts";

export interface VanillaTestOpts extends TestJobSpec {
  install: "vanilla";
  extras?: string[];
  req_files?: string[];
  dep_strategy?: "minimum" | "default";
}

export function isVanillaSpec(spec: TestJobSpec): spec is VanillaTestOpts {
  return spec.install == "vanilla";
}

export function vanillaSetup(options: VanillaTestOpts): WorkflowStep[] {
  let sync = "uv sync --group=cpu";
  if (options.extras) {
    for (const extra of options.extras) {
      sync += ` --extra=${extra}`;
    }
  }

  if (options.dep_strategy == "minimum") {
    sync += " --resolution=lowest-direct";
  }

  return [
    {
      name: "🕶️ Set up uv",
      uses: "astral/setup-uv@v5",
      with: {
        version: "latest",
        "python-version": pythonVersionString(options),
      },
    },
    {
      name: "📦 Set up Python dependencies",
      id: "install-deps",
      run: script(sync),
      shell: "bash",
      env: {
        PYTHON: "${{steps.install-python.outputs.python-path}}",
      },
    },
  ];
}
