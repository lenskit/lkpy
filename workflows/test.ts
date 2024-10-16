import { Workflow, WorkflowJob } from "@lenskit/typeline/github";

import { checkoutStep } from "./lib/checkout.ts";
import { script } from "./lib/script.ts";
import {
  CONDA_PYTHONS,
  META_PYTHON,
  PLATFORMS,
  PYTHONS,
  VANILLA_PLATFORMS,
} from "./lib/defs.ts";
import { testJob } from "./test/common.ts";
import { evalTestJob } from "./test/test-eval.ts";
import { exampleTestJob } from "./test/test-examples.ts";

const FILTER_PATHS = [
  "lenskit*/**.py",
  "**pyproject.toml",
  "requirements*.txt",
  "data/**",
  "utils/measure-coverage.tcl",
  ".github/workflows/test.yml",
];

const test_matrix = {
  conda: testJob({
    install: "conda",
    key: "conda",
    name: "Conda Python ${{matrix.python}} on ${{matrix.platform}}",
    matrix: { "python": CONDA_PYTHONS, "platform": PLATFORMS },
  }),
  vanilla: testJob({
    install: "vanilla",
    key: "vanilla",
    name: "Vanilla Python ${{matrix.python}} on ${{matrix.platform}}",
    matrix: { "python": PYTHONS, "platform": VANILLA_PLATFORMS },
  }),
  nojit: testJob({
    install: "vanilla",
    key: "nojit",
    name: "Non-JIT test coverage",
    packages: ["lenskit", "lenskit-funksvd"],
    test_env: { "NUMBA_DISABLE_JIT": 1, "PYTORCH_JIT": 0 },
    test_args: ["-m", "'not slow'"],
  }),
  mindep: testJob({
    install: "vanilla",
    key: "mindep",
    name: "Minimal dependency tests",
    dep_strategy: "minimum",
  }),
  funksvd: testJob({
    install: "conda",
    key: "funksvd",
    name: "FunkSVD tests on Python ${{matrix.python}}",
    packages: ["lenskit-funksvd"],
    matrix: { "python": CONDA_PYTHONS },
    variant: "full",
  }),
  "funksvd-mindep": testJob({
    install: "vanilla",
    key: "mindep-funksvd",
    name: "Minimal dependency tests for FunkSVD",
    dep_strategy: "minimum",
    packages: ["lenskit-funksvd"],
  }),
  implicit: testJob({
    install: "conda",
    key: "implicit",
    name: "Implicit bridge tests on Python ${{matrix.python}}",
    packages: ["lenskit-implicit"],
    matrix: { "python": CONDA_PYTHONS },
    variant: "full",
  }),
  "implicit-mindep": testJob({
    install: "vanilla",
    key: "mindep-implicit",
    name: "Minimal dependency tests for Implicit",
    dep_strategy: "minimum",
    packages: ["lenskit-implicit"],
  }),
  hpf: testJob({
    install: "conda",
    key: "hpf",
    name: "HPF bridge tests on Python ${{matrix.python}}",
    packages: ["lenskit-hpf"],
    matrix: { "python": CONDA_PYTHONS },
    variant: "full",
  }),
  "eval-tests": evalTestJob(),
  "doc-tests": exampleTestJob(),
};

export const results: WorkflowJob = {
  name: "Test suite results",
  "runs-on": "ubuntu-latest",
  needs: Object.keys(test_matrix),
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
    {
      name: "🧚 Set up Pixi",
      uses: "prefix-dev/setup-pixi@0.8.1",
      with: {
        "pixi-version": "latest",
        "activate-environment": true,
        "environments": "report",
      },
    },
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
      if: "always()",
      with: {
        name: "coverage-report",
        path: "lenskit-coverage/",
      },
    },
    {
      name: "🚫 Fail if coverage is too low",
      run: "coverage report --fail-under=90",
    },
  ],
};

export const workflow: Workflow = {
  name: "Automatic Tests",
  on: {
    push: { "branches": ["main"], "paths": FILTER_PATHS },
    pull_request: { "paths": FILTER_PATHS },
  },
  concurrency: {
    group: "test-${{github.ref}}",
    "cancel-in-progress": true,
  },
  jobs: {
    ...test_matrix,
    results,
  },
};
