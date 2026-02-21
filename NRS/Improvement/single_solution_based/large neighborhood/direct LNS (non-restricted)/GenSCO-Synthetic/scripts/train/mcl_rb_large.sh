export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

python -u -m training.train_mcl \
    --model_config dim256_qknorm_sigmoid_decoder_only_layer12 \
    --peak_lr 1e-3 \
    --batch_size 256 \
    --num_steps 384000 \
    --num_warmup_steps 50 \
    --save_interval 5000 \
    --data datasets/mcl_rb_large_1-4.npz \
    --logdir logs/mcl_flow_rb_large_td0.2_l0_12 \
    --savedir ckpts/mcl_flow_rb_large_td0.2_l0_12 \
    --optimizer_type muon \
    --target_disruption 0.2 \
