#!/usr/bin/env perl

use strict;
use warnings;

my $test_count = 0;
my $verbose = $ENV{VERBOSE};
print STDERR "counting tests\n" if $verbose;

while (<>) {
    if (m/^(run-(lenskit|python)|require)\b/) {
        print STDERR "test: $_" if $verbose;
        $test_count += 1;
    }
}

print "1..$test_count\n"
