export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

python -u -m training.train_tsp \
    --num_nodes 1000 \
    --model_config default \
    --peak_lr 1e-3 \
    --batch_size 64 \
    --num_steps 600000 \
    --num_warmup_steps 50 \
    --save_interval 5000 \
    --data datasets/tsp1000_uniform_1.28m_new.npz \
    --logdir logs/tsp1000 \
    --savedir ckpts/tsp1000 \
    --optimizer_type muon \
    --target_disruption "(400, 401)" \
    --ckpt ckpts/tsp500.ckpt \
    --ignore_ckpt_train_config
