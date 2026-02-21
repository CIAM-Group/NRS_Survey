export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

python -u -m training.train_tsp \
    --num_nodes 100 \
    --model_config default \
    --peak_lr 1e-3 \
    --batch_size 1024 \
    --num_steps 750000 \
    --num_warmup_steps 0 \
    --save_interval 5000 \
    --data datasets/tsp100_uniform_1.28m_lkh5k.npz \
    --logdir logs/tsp100 \
    --savedir ckpts/tsp100 \
    --optimizer_type muon \
    --target_disruption "(40, 41)"
