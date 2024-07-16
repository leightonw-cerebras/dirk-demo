#!/usr/bin/env bash

set -e

cslc layout.csl --fabric-dims=9,5 \
--fabric-offsets=4,1 -o out --params=M:10 --memcpy --channels 1
cs_python run.py --name out
