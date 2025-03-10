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

export function evalTestJob(): WorkflowJob {
  const options: CondaTestOpts = {
    install: "conda",
    key: "eval-tests",
    name: "Evaluation-based tests",
    pixi_env: "test-py311-full",
  };

  return {
    name: "Evaluation-based tests",
    "runs-on": "ubuntu-latest",
    steps: [
      checkoutStep(),
      ...condaSetup(options),
      ...mlDataSteps(["ml-100k", "ml-20m", "ml-1m", "ml-10m"]),
      {
        "name": "Run Eval Tests",
        "run": script(
          `pytest --cov-append --cov=src/lenskit -m 'eval or realdata' --log-file test-eval.log tests`,
        ),
      },
      ...coverageSteps(options),
    ],
  };
}
