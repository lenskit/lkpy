// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University
// Copyright (C) 2023-2025 Drexel University
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

let logs = JSON.parse(await Deno.readTextFile('build/historical-coverage.json'));
console.log('read %d records', logs.count);

for (let c of logs.results) {
    if (!c.totals) continue;

    let blob = {
        meta: {
            source: 'codecov',
        },
        totals: {
            covered_lines: c.totals.hits,
            percent_covered: c.totals.coverage,
            missing_lines: c.totals.misses,
        }
    }
    console.log('commit %s: %o', c.commitid, blob);
    let cmd = new Deno.Command('git', {
        args: ['notes', '--ref=coverage','add',  '-m', JSON.stringify(blob) + '\n', c.commitid],
        stdout: 'inherit',
        stderr: 'inherit',
    });
    let out = await cmd.output();
    if (out.code != 0) {
        console.error('git notes failed with code %s', out.code);
        Deno.exit(5)
    }
}
