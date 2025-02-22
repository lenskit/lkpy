// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University
// Copyright (C) 2023-2025 Drexel University
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

import { WorkflowStep } from "@lenskit/typeline/github";

import { TestJobSpec } from "./spec.ts";
import { pythonVersionString } from "./spec.ts";

export interface CondaTestOpts extends TestJobSpec {
  install: "conda";
  variant?: "core" | string;
  pixi_env?: string;
}

export function isCondaSpec(spec: TestJobSpec): spec is CondaTestOpts {
  return spec.install == "conda";
}

export function condaSetup(options: CondaTestOpts | string): WorkflowStep[] {
  let env = typeof options == "string" ? options : options.pixi_env;
  if (!env && typeof options != "string") {
    const version = pythonVersionString(options);
    const variant = options.variant ?? "core";
    env = `test-${version}-${variant}`;
  }
  env ??= "default";

  return [
    {
      uses: "prefix-dev/setup-pixi@v0.8.1",
      with: {
        "pixi-version": "latest",
        "activate-environment": true,
        "environments": env,
        "cache-write": false,
        "log-level": "vv",
        locked: false,
        frozen: true,
      },
    },
  ];
}
