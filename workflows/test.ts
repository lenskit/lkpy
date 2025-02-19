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

const test_matrix = {
  conda: testJob({
    install: "conda",
    key: "conda",
    name: "Conda Python ${{matrix.python}} on ${{matrix.platform}}",
    matrix: {
      python: CONDA_PYTHONS,
      platform: PLATFORMS,
    },
  }),
  vanilla: testJob({
    install: "vanilla",
    key: "vanilla",
    name: "Vanilla Python ${{matrix.python}} on ${{matrix.platform}}",
    matrix: {
      python: PYTHONS,
      platform: VANILLA_PLATFORMS,
      exclude: [
        { python: "3.13", platform: "macos-latest" },
        { python: "3.13", platform: "windows-latest" },
        { python: "3.13", platform: "ubuntu-24.04-arm" },
      ],
    },
  }),
  nojit: testJob({
    install: "conda",
    key: "nojit",
    name: "Non-JIT test coverage",
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
    tests: ["tests/sklearn"],
    matrix: { "python": CONDA_PYTHONS },
    variant: "sklearn",
  }),
  "sklearn-mindep": testJob({
    install: "vanilla",
    key: "mindep-sklearn",
    name: "Minimal dependency tests for Scikit-Learn",
    dep_strategy: "minimum",
    tests: ["tests/sklearn"],
    extras: ["sklearn"],
  }),
  funksvd: testJob({
    install: "conda",
    key: "funksvd",
    name: "FunkSVD tests on Python ${{matrix.python}}",
    tests: ["tests/funksvd"],
    matrix: { "python": FULLDEP_CONDA_PYTHONS },
    variant: "funksvd",
  }),
  "funksvd-mindep": testJob({
    install: "vanilla",
    key: "mindep-funksvd",
    name: "Minimal dependency tests for FunkSVD",
    dep_strategy: "minimum",
    tests: ["tests/funksvd"],
    extras: ["funksvd"],
  }),
  implicit: testJob({
    install: "conda",
    key: "implicit",
    name: "Implicit bridge tests on Python ${{matrix.python}}",
    tests: ["tests/implicit"],
    matrix: { "python": FULLDEP_CONDA_PYTHONS },
    variant: "implicit",
  }),
  "implicit-mindep": testJob({
    install: "vanilla",
    key: "mindep-implicit",
    name: "Minimal dependency tests for Implicit",
    dep_strategy: "minimum",
    tests: ["tests/implicit"],
    extras: ["implicit"],
  }),
  hpf: testJob({
    install: "conda",
    key: "hpf",
    name: "HPF bridge tests on Python ${{matrix.python}}",
    matrix: { "python": FULLDEP_CONDA_PYTHONS },
    tests: ["tests/hpf"],
    variant: "hpf",
  }),
  "eval-tests": evalTestJob(),
  "doc-tests": exampleTestJob(),
};

export const workflow: Workflow = {
  name: "Automatic Tests",
  on: {
    push: { "branches": ["main"] },
    pull_request: {},
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
