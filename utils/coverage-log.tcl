#!/usr/bin/env tclsh

exec git fetch --all 2>@stderr >@stdout
exec coverage json 2>@stderr >@stdout

if {[info exists env(GITHUB_BASE_REF)]} {
    set base $env(GITHUB_BASE_REF)
    puts "scanning git log for upstream/$base"
    set gitlog [exec git log --pretty=ref upstream/$base 2>@stderr]
    foreach {line commit notes} [regexp -all -inline -line {^([a-z0-9]+) \((.*)\)$} $gitlog] {
        try {
            set prev_data [exec git notes --ref=coverage show $commit 2>@stderr]
            puts "found note on commit $commit"
            break
        } on error {err opts} {
            # do nothing, skip to next commit
        }
    }

    if {![info exists prev_data]} {
        puts stderr "no previous coverage found"
        exit 2
    }

    set prev_cov [exec jq .totals.percent_covered <<$prev_data 2>@stderr]
    set cur_cov [exec jq .totals.percent_covered coverage.json 2>&stderr]

    set cov_change [expr {$cur_cov - $prev_cov}]

    set sumh [open $env(GITHUB_JOB_SUMMARY) wa]
    puts $sumh "Coverage change **$cov_change** (from $prev_cov to $cur_cov).\n"
    close $sumh
} elseif {[info exists env(GITHUB_EVENT_NAME)] && [info exists env(GITHUB_TOKEN)]} {
    puts stderr "saving coverage data"
    set data [jq "{meta: .meta, totals: .totals}" coverage.json 2>@stderr]
    exec git notes --ref=coverage add -m $data HEAD 2>@stderr >@stdout
    exec git push origin refs/notes/coverage 2>@stderr >@stdout
} else {
    puts stderr "don't know what to do"
}
