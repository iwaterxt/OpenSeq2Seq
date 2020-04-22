#!/bin/bash


dir=/aifs/users/tx078/tools/OpenSeq2Seq-dev/experiments/chn2000/ds2_offline_skip3_bpe_innerskip/tests
test_dir=/aifs/users/tx078/tools/OpenSeq2Seq-dev/data/baseline_chn_2000/tests_skip3_bpe
gpu="3"

tests="ailab_tmjl_mangbiao_nov21_all"

for set in $tests; do
    mkdir -p $dir/$set
    cp /aifs/users/tx078/data/asr/tests_all/$set/text $dir/$set
    CUDA_VISIBLE_DEVICES=${gpu} python run.py \
        --config_file=example_configs/speech2text/chn2000/cmn/ds2_medium_4gpus_offline_skip3_bpe_innerskip.py \
        --mode=infer \
	--batch_size_per_gpu=1 \
        --infer_output_file=$dir/$set/output \
        || exit 1;

    python scripts/decode_poster_prepare.py $dir/$set/output ark,scp:$dir/$set/poster.ark,$dir/$set/poster.scp $test_dir/$set/feats.scp || exit 1;
    rm -rf $dir/$set/output
done


