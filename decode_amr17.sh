#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 -m sockeye.translate -m sockeye/amr2017_model \
        --edge-vocab sockeye/data/amr_2017/edge_vocab.json < sockeye/data/amr_2017/test_bpe.amrgrh \
        -o sockeye/data/amr_2017/test.snt.out \
        --beam-size 10 \