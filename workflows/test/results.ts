import { WorkflowJob } from "@lenskit/typeline/github";

import { checkoutStep } from "../lib/checkout.ts";
import { script } from "../lib/script.ts";
import { condaSetup } from "./conda.ts";

export function aggregateResultsJob(jobs: string[]): WorkflowJob {
  return {
    name: "Test suite results",
    "runs-on": "ubuntu-latest",
    needs: jobs,
    if: "${{ !canceled() }}",
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
        name: "± Measure and report coverage",
        run: script(`
          echo $PR_NUMBER > ./lenskit-coverage/pr-number
          tclsh ./utils/measure-coverage.tcl
          cat lenskit-coverage/report.md >$GITHUB_STEP_SUMMARY
        `),
        env: {
          PR_NUMBER: "${{ github.event.number }}",
          GH_TOKEN: "${{secrets.GITHUB_TOKEN}}",
        },
      },
      {
        name: "📤 Upload coverage report",
        uses: "actions/upload-artifact@v4",
        if: "${{ !canceled() }}",
        with: {
          name: "coverage-report",
          path: "lenskit-coverage/",
        },
      },
      {
        name: "📤 Upload coverage to CodeCov",
        uses: "codecov/codecov-action@v4.2.0",
        env: {
          CODECOV_TOKEN: "${{ env.CODECOV_TOKEN }}",
        },
      },
      {
        name: "🚫 Fail if coverage is too low",
        run: "coverage report --fail-under=90",
      },
    ],
  };
}
