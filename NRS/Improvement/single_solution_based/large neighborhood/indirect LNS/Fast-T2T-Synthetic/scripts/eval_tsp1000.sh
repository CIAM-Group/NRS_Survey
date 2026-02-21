export CUDA_VISIBLE_DEVICES=0

# step_list="5,0 5,5"
step_list="5,5"
for step in $step_list; do
    infer=$(echo $step | cut -d',' -f1)
    re=$(echo $step | cut -d',' -f2)

   #  python -u train.py \
   #     --task "tsp" \
   #     --project_name "consistency_co_test" \
   #     --wandb_logger_name "tsp_1000" \
   #     --do_test \
   #     --storage_path "./" \
   #     --test_split "/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp1000_n128_lkh.txt" \
   #     --inference_schedule "cosine" \
   #     --inference_diffusion_steps $infer \
   #     --two_opt_iterations 0 \
   #     --ckpt_path 'model/tsp1000.ckpt' \
   #     --consistency \
   #     --sparse_factor 100 \
   #     --resume_weight_only \
   #     --parallel_sampling 1 \
   #     --sequential_sampling 1 \
   #     --rewrite \
   #     --guided \
   #     --rewrite_steps $re \
   #     --rewrite_ratio 0.2 \
   #     --offline > cns_1000_g_$infer'_'$re'.txt'
   # wait $! # wait for the last process to finish

    nohup python -u train.py \
       --task "tsp" \
       --project_name "consistency_co_test" \
       --wandb_logger_name "tsp_1000" \
       --do_test \
       --storage_path "./" \
       --test_split "/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp1000_n128_lkh.txt" \
       --inference_schedule "cosine" \
       --inference_diffusion_steps $infer \
       --two_opt_iterations 5000 \
       --ckpt_path 'model/tsp1000.ckpt' \
       --consistency \
       --sparse_factor 100 \
       --resume_weight_only \
       --parallel_sampling 1 \
       --sequential_sampling 1 \
       --rewrite \
       --guided \
       --rewrite_steps $re \
       --rewrite_ratio 0.2 \
       --offline > cns_two_opt_1000_g_$infer'_'$re'.txt' 2>&1 &
   # wait $! # wait for the last process to finish

   #  python -u train.py \
   #     --task "tsp" \
   #     --project_name "consistency_co_test" \
   #     --wandb_logger_name "tsp_1000" \
   #     --do_test \
   #     --storage_path "./" \
   #     --test_split "/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp1000_n128_lkh.txt" \
   #     --inference_schedule "cosine" \
   #     --inference_diffusion_steps $infer \
   #     --two_opt_iterations 0 \
   #     --ckpt_path 'model/tsp1000.ckpt' \
   #     --consistency \
   #     --sparse_factor 100 \
   #     --resume_weight_only \
   #     --parallel_sampling 2 \
   #     --sequential_sampling 2 \
   #     --rewrite \
   #     --guided \
   #     --rewrite_steps $re \
   #     --rewrite_ratio 0.2 \
   #     --offline > cns_1000_s_$infer'_'$re'.txt'
   # wait $! # wait for the last process to finish 

   #  python -u train.py \
   #     --task "tsp" \
   #     --project_name "consistency_co_test" \
   #     --wandb_logger_name "tsp_1000" \
   #     --do_test \
   #     --storage_path "./" \
   #     --test_split "/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp1000_n128_lkh.txt" \
   #     --inference_schedule "cosine" \
   #     --inference_diffusion_steps $infer \
   #     --two_opt_iterations 5000 \
   #     --ckpt_path 'model/tsp1000.ckpt' \
   #     --consistency \
   #     --sparse_factor 100 \
   #     --resume_weight_only \
   #     --parallel_sampling 2 \
   #     --sequential_sampling 2 \
   #     --rewrite \
   #     --guided \
   #     --rewrite_steps $re \
   #     --rewrite_ratio 0.2 \
   #     --offline > cns_two_opt_1000_s_$infer'_'$re'.txt'
   # wait $! # wait for the last process to finish
done