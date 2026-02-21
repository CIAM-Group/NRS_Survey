python -u -m decoding.mis \
    --data datasets/rb_200_300_test.npz \
    --ckpt ckpts/mis_rb.ckpt \
    --sampling_steps 3 \
    --cycles 500 --runs 1 \
    --batch_size 100 \
    --threads_over_batches 5 \
    --decode_intermediate
