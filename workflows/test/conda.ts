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

export function condaSetup(options: CondaTestOpts): WorkflowStep[] {
  let env = options.pixi_env;
  if (!env) {
    const version = pythonVersionString(options);
    const variant = options.variant ?? "core";
    env = `test-${version}-${variant}`;
  }

  return [
    {
      uses: "prefix-dev/setup-pixi@v0.8.1",
      with: {
        "pixi-version": "latest",
        "activate-environment": true,
        "environments": env,
        "cache-write": false,
      },
    },
  ];
}
