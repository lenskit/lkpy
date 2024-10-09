import { WorkflowJob, WorkflowStep } from "@lenskit/typeline/github";
import { TestJobSpec, testPlatform } from "./spec.ts";
import { script } from "../lib/script.ts";
import { checkoutStep } from "../lib/checkout.ts";
import { inspectStep } from "./environment.ts";
import { condaSetup, CondaTestOpts, isCondaSpec } from "./conda.ts";
import { isVanillaSpec, vanillaSetup, VanillaTestOpts } from "./vanilla.ts";

export function testJob(options: VanillaTestOpts): WorkflowJob;
export function testJob(options: CondaTestOpts): WorkflowJob;
export function testJob(
    options: TestJobSpec,
): WorkflowJob {
    const strategy = options.matrix
        ? {
            "fail-fast": false,
            "matrix": options.matrix,
        }
        : undefined;

    let setup;
    if (isVanillaSpec(options)) {
        setup = vanillaSetup(options);
    } else if (isCondaSpec(options)) {
        setup = condaSetup(options);
    } else {
        throw new Error(`unknown install type ${options.install}`);
    }

    return {
        "name": options.name,
        "runs-on": testPlatform(options),
        "timeout-minutes": 30,
        strategy,
        steps: [
            checkoutStep(),
            ...setup,
            inspectStep(),
            ...testSteps(options),
        ],
    };
}

export function testSteps(options: TestJobSpec): WorkflowStep[] {
    let test_cmd =
        "python -m pytest --verbose --log-file=test.log --durations=25";
    if (options.test_args) {
        test_cmd += " " + options.test_args.join(" ");
    }
    for (const pkg of options.packages ?? []) {
        test_cmd += ` --cov=${pkg}/lenskit`;
    }
    for (const pkg of options.packages ?? []) {
        test_cmd += ` ${pkg}/tests`;
    }

    return [{
        "name": "🏃🏻‍➡️ Test LKPY",
        "run": script(test_cmd),
        env: options.test_env,
    }, ...coverageSteps(options)];
}

export function coverageSteps(options: TestJobSpec): WorkflowStep[] {
    return [
        {
            "name": "📐 Coverage results",
            "run": script(`
                coverage xml
                coverage report
                cp .coverage coverage.db
            `),
        },
        {
            "name": "📤 Upload test results",
            "uses": "actions/upload-artifact@v4",
            "if": "always()",
            "with": {
                "name": testArtifactName(options),
                "path": script(`
                    test*.log
                    coverage.db
                    coverage.xml
                `),
            },
        },
    ];
}

function testArtifactName(options: TestJobSpec): string {
    let name = `test-${options.key}`;
    if (options.matrix) {
        if (options.matrix.platform) {
            name += "-${{matrix.platform}}";
        }
        if (options.matrix.python) {
            name += "-py${{matrix.python}}";
        }
        for (const key of Object.keys(options.matrix)) {
            if (key != "platform" && key != "python") {
                name += "-${{matrix." + key + "}}";
            }
        }
    }
    return name;
}
