python -u -m decoding.mcl \
    --data datasets/mcl_rb_small_testset_mlbench.npz \
    --ckpt ckpts/mcl_rb_small.ckpt \
    --sampling_steps 3 \
    --cycles 500 --runs 1 \
    --batch_size 100 \
    --threads_over_batches 5 \
    --decode_intermediate
