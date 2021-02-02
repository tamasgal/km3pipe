#!/bin/bash
set -e

h5extract $(python -m km3net_testdata offline/km3net_offline.root)
