// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University
// Copyright (C) 2023-2025 Drexel University
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

import { WorkflowJob } from "@lenskit/typeline/github";
import { checkoutStep } from "../lib/checkout.ts";
import { condaSetup, CondaTestOpts } from "./conda.ts";
import { mlDataSteps } from "./data.ts";
import { coverageSteps } from "./common.ts";
import { script } from "../lib/script.ts";

export function exampleTestJob(): WorkflowJob {
  const options: CondaTestOpts = {
    install: "conda",
    key: "examples",
    name: "Demos, examples, and docs",
    pixi_env: "test-examples",
  };

  return {
    name: options.name,
    "runs-on": "ubuntu-latest",
    steps: [
      checkoutStep(),
      ...condaSetup(options),
      ...mlDataSteps(["ml-100k", "ml-1m", "ml-10m", "ml-20m"]),
      {
        "name": "📕 Validate code examples",
        "run": script(
          `sphinx-build -b doctest docs build/doc`,
        ),
      },
      {
        "name": "📕 Validate example notebooks",
        "run": script(
          `pytest --cov=src/lenskit --nbval-lax --log-file test-notebooks.log docs`,
        ),
      },
      ...coverageSteps(options),
    ],
  };
}
