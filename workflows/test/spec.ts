import { PYTHONS } from "../lib/defs.ts";

export interface TestJobSpec {
  install: string;
  key: string;
  name: string;
  runs_on?: string;
  matrix?: {
    python?: string[];
    platform?: string[];
    exclude?: Record<string, string>[];
  };
  python?: string;
  packages?: string[];
  test_args?: string[];
  test_env?: Record<string, string | number>;
}

export function testPlatform(spec: TestJobSpec): string {
  if (spec.runs_on) {
    return spec.runs_on;
  } else if (spec.matrix?.platform) {
    return "${{matrix.platform}}";
  } else {
    return "ubuntu-latest";
  }
}

/**
 * Get a Python version string for a spec.
 */
export function pythonVersionString(spec: TestJobSpec): string {
  if (spec.python) {
    return translatePythonVersion(spec.python, spec.install);
  } else if (spec.matrix?.python) {
    return "${{matrix.python}}";
  } else {
    return translatePythonVersion(PYTHONS[0], spec.install);
  }
}

/**
 * Translate a Python version string for an installer.
 * @param ver The Python version string (e.g. 3.11).
 * @param install The installer method.
 * @returns The translated string (e.g. py311 for Conda).
 */
export function translatePythonVersion(ver: string, install: string): string {
  if (install == "conda") {
    return "py" + ver.replaceAll(".", "");
  } else {
    return ver;
  }
}

export function packages(spec: TestJobSpec): string[] {
  return spec.packages ?? ["lenskit"];
}
