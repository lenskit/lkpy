// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University
// Copyright (C) 2023-2025 Drexel University
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

import { WorkflowJob } from "@lenskit/typeline/github";

import { checkoutStep } from "../lib/checkout.ts";
import { script } from "../lib/script.ts";
import { condaSetup } from "./conda.ts";

const CODECOV_TOKEN = "ab58c9cf-25b8-4283-a485-0b6382dc9a61";

export function aggregateResultsJob(jobs: string[]): WorkflowJob {
  return {
    name: "Test suite results",
    "runs-on": "ubuntu-latest",
    needs: jobs,
    if: "${{ !cancelled() }}",
    steps: [
      checkoutStep(),
      {
        name: "Add upstream remote & author config",
        run: script(`
          git remote add upstream https://github.com/lenskit/lkpy.git
          git fetch upstream
          git config user.name "LensKit Bot"
          git config user.email lkbot@lenskit.org
        `),
      },
      ...condaSetup("report"),
      {
        name: "📥 Download test artifacts",
        uses: "actions/download-artifact@v4",
        with: {
          pattern: "test-*",
          path: "test-logs",
        },
      },
      {
        name: "📋 List log files",
        run: "ls -laR test-logs",
      },
      {
        name: "🔧 Fix coverage databases",
        run: script(`
          for dbf in test-logs/*/coverage.db; do
              echo "fixing $dbf"
              sqlite3 -echo "$dbf" "UPDATE file SET path = replace(path, '\\', '/');"
          done
        `),
      },
      // inspired by https://hynek.me/articles/ditch-codecov-python/
      {
        name: "⛙ Merge and report",
        run: script(`
          coverage combine --keep test-logs/*/coverage.db
          coverage xml
          coverage html -d lenskit-coverage
          coverage report --format=markdown >coverage.md
        `),
      },
      {
        name: "Analyze diff coverage",
        if: "github.event_name == 'pull_request'",
        run: script(`
          diff-cover --json-report diff-cover.json --markdown-report diff-cover.md \\
            coverage.xml |tee diff-cover.txt
        `),
      },
      {
        name: "📤 Upload coverage to CodeCov",
        uses: "codecov/codecov-action@v4.2.0",
        env: {
          CODECOV_TOKEN: CODECOV_TOKEN,
        },
      },
      {
        name: "🚫 Fail if coverage is too low",
        run: "coverage report --fail-under=90",
      },
    ],
  };
}
