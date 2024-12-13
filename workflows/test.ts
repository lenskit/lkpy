import { Workflow } from "@lenskit/typeline/github";

import {
  CONDA_PYTHONS,
  PLATFORMS,
  PYTHONS,
  VANILLA_PLATFORMS,
} from "./lib/defs.ts";
import { testJob } from "./test/common.ts";
import { evalTestJob } from "./test/test-eval.ts";
import { exampleTestJob } from "./test/test-examples.ts";
import { aggregateResultsJob } from "./test/results.ts";

const FULLDEP_CONDA_PYTHONS = ["py311", "py312"];

const FILTER_PATHS = [
  "lenskit*/**.py",
  "**pyproject.toml",
  "pixi.*",
  "requirements*.txt",
  "data/**",
  ".github/workflows/test.yml",
];

const test_matrix = {
  conda: testJob({
    install: "conda",
    key: "conda",
    name: "Conda Python ${{matrix.python}} on ${{matrix.platform}}",
    matrix: {
      python: CONDA_PYTHONS,
      platform: PLATFORMS,
      exclude: [{ python: "py313", platform: "windows-latest" }],
    },
  }),
  vanilla: testJob({
    install: "vanilla",
    key: "vanilla",
    name: "Vanilla Python ${{matrix.python}} on ${{matrix.platform}}",
    matrix: { "python": PYTHONS, "platform": VANILLA_PLATFORMS },
  }),
  nojit: testJob({
    install: "conda",
    key: "nojit",
    name: "Non-JIT test coverage",
    packages: ["lenskit", "lenskit-funksvd"],
    test_env: { "NUMBA_DISABLE_JIT": 1, "PYTORCH_JIT": 0 },
    test_args: ["-m", "'not slow'"],
    variant: "funksvd",
  }),
  mindep: testJob({
    install: "vanilla",
    key: "mindep",
    name: "Minimal dependency tests",
    dep_strategy: "minimum",
  }),
  sklearn: testJob({
    install: "conda",
    key: "sklearn",
    name: "Scikit-Learn tests on Python ${{matrix.python}}",
    packages: ["lenskit-sklearn"],
    matrix: { "python": CONDA_PYTHONS },
    variant: "sklearn",
  }),
  "sklearn-mindep": testJob({
    install: "vanilla",
    key: "mindep-sklearn",
    name: "Minimal dependency tests for Scikit-Learn",
    dep_strategy: "minimum",
    packages: ["lenskit-sklearn"],
  }),
  funksvd: testJob({
    install: "conda",
    key: "funksvd",
    name: "FunkSVD tests on Python ${{matrix.python}}",
    packages: ["lenskit-funksvd"],
    matrix: { "python": FULLDEP_CONDA_PYTHONS },
    variant: "funksvd",
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
    matrix: { "python": FULLDEP_CONDA_PYTHONS },
    variant: "implicit",
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
    matrix: { "python": FULLDEP_CONDA_PYTHONS },
    variant: "hpf",
  }),
  "eval-tests": evalTestJob(),
  "doc-tests": exampleTestJob(),
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
    results: aggregateResultsJob(Object.keys(test_matrix)),
  },
};
