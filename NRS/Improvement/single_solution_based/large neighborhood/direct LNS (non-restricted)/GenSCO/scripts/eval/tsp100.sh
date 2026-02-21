python -m decoding.tsp \
    --data datasets/tsp100_concorde_7.75585.npz \
    --ckpt ckpts/tsp100.ckpt \
    --sampling_steps 4 --two_opt_steps 2 --cycles 10 \
    --runs 8 --batch_size 128 \
    --random_two_opt_steps_range "(25, 75)" \
    --threads_over_batches 2 \
    --argument_level 1 
