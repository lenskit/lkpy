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
        key: `test-mldata-001-${ds_key}`,
      },
    },
    {
      name: "Download ML data",
      run: script(`
                python -m lenskit.data.fetch ${ds_str}
            `),
    },
  ];
}
