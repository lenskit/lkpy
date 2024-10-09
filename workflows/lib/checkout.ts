import { WorkflowStep } from "@lenskit/typeline/github";

export function checkoutStep(depth: number = 0): WorkflowStep {
  return {
    name: "🛒 Checkout",
    uses: "actions/checkout@v4",
    with: { "fetch-depth": depth },
  };
}
