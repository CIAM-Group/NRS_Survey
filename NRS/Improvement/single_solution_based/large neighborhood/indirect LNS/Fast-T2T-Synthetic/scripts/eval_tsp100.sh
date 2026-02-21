export CUDA_VISIBLE_DEVICES=0

nohup python -u train.py \
--task "tsp" \
--project_name "consistency_co_test" \
--wandb_logger_name "tsp_100" \
--do_test \
--storage_path "./" \
--test_split "/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp100_n10000_lkh.txt" \
--inference_schedule "cosine" \
--inference_diffusion_steps 5 \
--two_opt_iterations 0 \
--ckpt_path 'model/tsp100.ckpt' \
--consistency \
--use_intermediate \
--resume_weight_only \
--parallel_sampling 1 \
--sequential_sampling 1 \
--rewrite \
--guided \
--rewrite_steps 5 \
--rewrite_ratio 0.2 \
--offline > cns_100_g_5_5.txt 2>&1 &
