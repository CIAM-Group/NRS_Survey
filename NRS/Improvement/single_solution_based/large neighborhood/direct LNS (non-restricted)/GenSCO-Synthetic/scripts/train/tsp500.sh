export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

python -u -m training.train_tsp \
    --num_nodes 500 \
    --model_config default \
    --peak_lr 1e-3 \
    --batch_size 128 \
    --num_steps 600000 \
    --num_warmup_steps 50 \
    --save_interval 5000 \
    --data datasets/tsp500_uniform_train_lkh5w.npz \
    --logdir logs/tsp500 \
    --savedir ckpts/tsp500 \
    --optimizer_type muon \
    --target_disruption "(200, 201)" \
    --ckpt ckpts/tsp100.ckpt \
    --ignore_ckpt_train_config

