python -u -m decoding.mcl \
    --data datasets/mcl_rb_large_mlbench.npz \
    --ckpt ckpts/mcl_rb_large.ckpt \
    --sampling_steps 3 \
    --cycles 500 --runs 1 \
    --batch_size 100 \
    --threads_over_batches 5 \
    --decode_intermediate
