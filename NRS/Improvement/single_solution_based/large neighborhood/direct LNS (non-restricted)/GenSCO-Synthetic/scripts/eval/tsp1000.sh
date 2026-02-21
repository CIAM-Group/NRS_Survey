nohup python -u -m decoding.tsp \
--data "/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp1000_n128_lkh.txt" \
--ckpt "/public/home/bayp/exp_survey_202509/Improvement/single_solution_based/large neighborhood/direct LNS (non-restricted)/GenSCO/ckpts/tsp1000.ckpt" \
--sampling_steps 4 --two_opt_steps 20 --cycles 160 \
--runs 8 --batch_size 16 \
--random_two_opt_steps_range "(250, 750)" \
--threads_over_batches 4 \
--argument_level 1 \
--heatmap_dtype uint8 --topk 20000 > gensco_tsp1000_eval.txt 2>&1 &
