import { WorkflowJob } from "@lenskit/typeline/github";
import { checkoutStep } from "../lib/checkout.ts";
import { condaSetup, CondaTestOpts } from "./conda.ts";
import { mlDataSteps } from "./data.ts";
import { coverageSteps } from "./common.ts";
import { script } from "../lib/script.ts";
import { PACKAGES } from "../lib/defs.ts";

export function evalTestJob(): WorkflowJob {
  const options: CondaTestOpts = {
    install: "conda",
    key: "eval-tests",
    name: "Evaluation-based tests",
    pixi_env: "py311-full",
    packages: PACKAGES,
  };

  const cov = PACKAGES.map((pkg) => `--cov=${pkg}/lenskit`).join(" ");
  return {
    name: "Evaluation-based tests",
    "runs-on": "ubuntu-latest",
    steps: [
      checkoutStep(),
      ...condaSetup(options),
      ...mlDataSteps(["ml-100k", "ml-20m"]),
      {
        "name": "Run Eval Tests",
        "run": script(`
                    python -m pytest ${cov} -m 'eval or realdata' --log-file test-eval.log */tests
                `),
      },
      ...coverageSteps(options),
    ],
  };
}
