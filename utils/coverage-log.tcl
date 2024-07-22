#!/usr/bin/env tclsh

proc ev {name {var ""}} {
    if {![info exists ::env($name)]} {
        return 0
    } else {
        set val $::env($name)
        if {$val eq ""} {
            return 0
        } else {
            if {$var ne ""} {
                upvar $var vv
                set vv $val
            }
            return 1
        }
    }
}

exec git fetch --all 2>@stderr >@stdout
exec git fetch upstream refs/notes/coverage:refs/notes/coverage 2>@stderr >@stdout
exec coverage json 2>@stderr >@stdout

if {[ev GITHUB_BASE_REF base]} {
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

    set sumh [open $env(GITHUB_STEP_SUMMARY) a]
    puts $sumh [format "Coverage change **%.2f%%** (from %.2f%% to %.2f%%).\n" $cov_change $prev_cov $cur_cov]
    close $sumh
} elseif {[ev GITHUB_EVENT_NAME] && [ev GITHUB_TOKEN]} {
    puts stderr "saving coverage data"
    set data [jq "{meta: .meta, totals: .totals}" coverage.json 2>@stderr]
    exec git notes --ref=coverage add -m $data HEAD 2>@stderr >@stdout
    exec git push origin refs/notes/coverage 2>@stderr >@stdout
} else {
    puts stderr "don't know what to do"
}
