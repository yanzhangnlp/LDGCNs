#!/usr/bin/env bash

# For AMR-to-Text
python3 -m sockeye.evaluate -r sockeye/data/amr_2017/test.snt  -i sockeye/data/amr_2017/test.snt.out
