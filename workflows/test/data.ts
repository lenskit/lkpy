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
                coverage run --source=lenskit/lenskit -m lenskit data fetch -D data --movielens ${ds_str}
            `),
    },
  ];
}
