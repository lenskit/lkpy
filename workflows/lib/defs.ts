// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University
// Copyright (C) 2023-2025 Drexel University
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

/**
 * Definitions used for the other workflow modules.
 * @module
 */

export const META_PYTHON = "3.11";
export const PYTHONS = ["3.11", "3.12", "3.13"];
export const PLATFORMS = [
  "ubuntu-latest",
  "ubuntu-24.04-arm",
  "macos-latest",
  "windows-latest",
];
export const VANILLA_PLATFORMS = PLATFORMS;
export const CONDA_PYTHONS = PYTHONS.map((s) => `py${s.replaceAll(".", "")}`);
