python -m decoding.tsp \
    --data datasets/tsp500_concorde_16.54581.npz \
    --ckpt ckpts/tsp500.ckpt \
    --sampling_steps 4 --two_opt_steps 10 --cycles 10 \
    --runs 8 --batch_size 16 \
    --random_two_opt_steps_range "(125, 375)" \
    --threads_over_batches 8 \
    --argument_level 1 \
    --heatmap_dtype uint8 --topk 5000
