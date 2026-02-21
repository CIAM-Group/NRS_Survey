python -m decoding.tsp \
    --data datasets/tsp1000_concorde_23.11812.npz \
    --ckpt ckpts/tsp1000.ckpt \
    --sampling_steps 4 --two_opt_steps 20 --cycles 10 \
    --runs 8 --batch_size 16 \
    --random_two_opt_steps_range "(250, 750)" \
    --threads_over_batches 4 \
    --argument_level 1 \
    --heatmap_dtype uint8 --topk 20000
