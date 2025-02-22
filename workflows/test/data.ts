// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University
// Copyright (C) 2023-2025 Drexel University
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

import { WorkflowStep } from "@lenskit/typeline/github";
import { script } from "../lib/script.ts";

export function mlDataSteps(datasets: string[]): WorkflowStep[] {
  const ds_str = datasets.join(" ");
  const ds_key = datasets.join("-");
  return [
    {
      name: "Cache ML data",
      uses: "actions/cache@v4",
      with: {
        path: script(`
                        data
                        !data/*.zip
                    `),
        key: `test-mldata-002-${ds_key}`,
      },
    },
    {
      name: "Download ML data",
      run: script(`
                coverage run --source=src/lenskit -m lenskit data fetch -D data --movielens ${ds_str}
            `),
    },
  ];
}
