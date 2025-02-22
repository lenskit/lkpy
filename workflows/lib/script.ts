// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University
// Copyright (C) 2023-2025 Drexel University
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

import { assert } from "@std/assert/assert";

export function script(...lines: string[]): string {
    let script = "";
    for (let line of lines) {
        // strip leading newlines
        line = line.replace(/^(\s*?\n)+/, "");
        // look at first line indent
        const m = line.match(/^ */);
        assert(m != null);
        const lead = m[0];
        if (lead.length) {
            line = line.replaceAll(new RegExp(`^ {${lead.length}}`, "gm"), "");
        }
        line = line.trimEnd();
        line += "\n";
        script += line;
    }
    return script;
}
