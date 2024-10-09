import { script } from "../lib/script.ts";

export function inspectStep() {
  return {
    name: "🔍 Inspect environment",
    run: script(`
            python -m lenskit.util.envcheck
        `),
  };
}
