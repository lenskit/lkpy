/**
 * Definitions used for the other workflow modules.
 * @module
 */

/**
 * List of packages in the monorepo.
 */
export const PACKAGES = [
    "lenskit",
    "lenskit-funksvd",
    "lenskit-implicit",
    "lenskit-hpf",
];

export const META_PYTHON = "3.11";
export const PYTHONS = ["3.11", "3.12"];
export const PLATFORMS = ["ubuntu-latest", "macos-latest", "windows-latest"];
export const VANILLA_PLATFORMS = ["ubuntu-latest", "macos-latest"];
export const CONDA_PYTHONS = PYTHONS.map((s) => `py${s.replaceAll(".", "")}`);
