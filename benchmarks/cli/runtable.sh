#!/usr/bin/env bash
set -euo pipefail

runtable -h

runtable 44 -n 5

# target filtering
runtable 42 -t run

# compact view
runtable 44 -c -n 10
