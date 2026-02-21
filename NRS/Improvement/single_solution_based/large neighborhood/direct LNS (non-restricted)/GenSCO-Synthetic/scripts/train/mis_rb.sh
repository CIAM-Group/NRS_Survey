export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

python -u -m training.train_mis \
    --model_config dim256_qknorm_sigmoid_decoder_only_layer12 \
    --peak_lr 5e-4 \
    --batch_size 256 \
    --num_steps 211000 \
    --num_warmup_steps 50 \
    --save_interval 5000 \
    --data datasets/rb_200_300_train.npz --label datasets/rb_200_300_train_labels.npz \
    --logdir logs/mis_rb \
    --savedir ckpts/mis_rb \
    --optimizer_type muon \
    --target_disruption 0.1
