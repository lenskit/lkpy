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

    set diff_lines [exec jq .total_num_lines diff-cover.json]
    set diff_bad [exec jq .total_num_violations diff-cover.json]
    if {$diff_lines > 0} {
        set diff_cov [expr {100 - ($diff_bad * 100.0) / $diff_lines}]
    } else {
        set diff_cov NA
    }

    set prev_cov [exec jq .totals.percent_covered <<$prev_data 2>@stderr]
    set cur_cov [exec jq .totals.percent_covered coverage.json 2>@stderr]

    set cov_change [expr {$cur_cov - $prev_cov}]

    # write the coverage report
    set reph [open lenskit-coverage/report.md w]
    puts $reph "The LensKit 🤖 has run the tests on your PR.\n"
    if {$diff_cov eq "NA"} {
        puts $reph [format \
            "Covered **no lines** of diff (coverage changed **%.2f%%** from %.2f%% to %.2f%%).\n" \
            $cov_change $prev_cov $cur_cov \
        ]
    } else {
        puts $reph [format \
            "Covered **%.2f%%** of diff (coverage changed **%.2f%%** from %.2f%% to %.2f%%).\n" \
            $diff_cov $cov_change $prev_cov $cur_cov \
        ]
    }

    set dsh [open diff-cover.md r]
    while {[gets $dsh line] >= 0} {
        if {[regexp {^# Diff} $line]} {
            # do nothing, first header
        } elseif {[regexp {^## Diff: (.*)} $line -> label]} {
            puts $reph "<details>"
            puts $reph "<summary>$label</summary>\n"
        } elseif {[regexp {^## lenskit/} $line]} {
            # done
            break
        } else {
            puts $reph $line
        }
    }
    puts $reph "\n</details>"
    close $dsh

    puts $reph "<details>\n"
    puts $reph "<summary>Source Coverage Report</summary>\n"
    exec coverage report --format=markdown 2>@stderr >@$reph
    puts $reph "\n</details>"

    close $reph
} elseif {[ev GITHUB_EVENT_NAME] && [ev GH_TOKEN]} {
    puts stderr "saving coverage data"
    set data [exec jq "{meta: .meta, totals: .totals}" coverage.json 2>@stderr]
    exec git notes --ref=coverage add -m $data HEAD 2>@stderr >@stdout
    exec git push origin refs/notes/coverage 2>@stderr >@stdout

    set cov [exec jq .totals.percent_covered coverage.json 2>@stderr]

    set reph [open lenskit-coverage/report.md w]
    puts $reph [format "Covered **%.2f%%** of code.\n" $cov]

    puts $reph "<details>\n"
    puts $reph "<summary>Source Coverage Report</summary>\n"
    exec coverage report --format=markdown 2>@stderr >@$reph
    puts $reph "\n</details>"

    close $reph
} else {
    puts stderr "don't know what to do"
    exit 1
}
