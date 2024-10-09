import { WorkflowStep } from "@lenskit/typeline/github";

import { pythonVersionString, TestJobSpec } from "./spec.ts";
import { script } from "../lib/script.ts";

export interface VanillaTestOpts extends TestJobSpec {
    install: "vanilla";
    req_files?: string[];
    dep_strategy?: "minimum" | "default";
    pip_args?: string[];
}

export function isVanillaSpec(spec: TestJobSpec): spec is VanillaTestOpts {
    return spec.install == "vanilla";
}

export function vanillaSetup(options: VanillaTestOpts): WorkflowStep[] {
    let pip = "uv pip install --python $PYTHON";
    for (const req of reqFiles(options)) {
        pip += ` -r ${req}`;
    }
    for (const pkg of requiredPackages(options)) {
        pip += ` -e ${pkg}`;
    }
    if (options.dep_strategy == "minimum") {
        pip += " --resolution=lowest-direct";
    }
    if (options.pip_args) {
        pip += " " + options.pip_args.join(" ");
    }

    return [
        {
            "name": "🐍 Set up Python",
            "id": "install-python",
            "uses": "actions/setup-python@v5",
            "with": {
                "python-version": pythonVersionString(options),
                "cache": "pip",
                "cache-dependency-path": script(`
                    requirements*.txt
                    */pyproject.toml
                `),
            },
        },
        {
            "name": "🕶️ Set up uv",
            "run": script("pip install -U 'uv>=0.1.15'"),
        },
        {
            "name": "📦 Set up Python dependencies",
            "id": "install-deps",
            "run": script(pip),
            "shell": "bash",
            "env": {
                "PYTHON": "${{steps.install-python.outputs.python-path}}",
                "UV_EXTRA_INDEX_URL": "https://download.pytorch.org/whl/cpu",
                "UV_INDEX_STRATEGY": "unsafe-first-match",
            },
        },
    ];
}

function reqFiles(options: VanillaTestOpts) {
    return options.req_files ?? ["requirements-test.txt"];
}

function requiredPackages(options: VanillaTestOpts) {
    const pkgs = options.packages ?? [];
    if (!pkgs.includes("lenskit")) {
        pkgs.unshift("lenskit");
    }
    return pkgs;
}
