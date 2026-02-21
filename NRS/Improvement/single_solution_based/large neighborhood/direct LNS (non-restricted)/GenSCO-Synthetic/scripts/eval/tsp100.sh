nohup python -u -m decoding.tsp \
--data "/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp100_n10000_lkh.txt" \
--ckpt "/public/home/bayp/exp_survey_202509/Improvement/single_solution_based/large neighborhood/direct LNS (non-restricted)/GenSCO/ckpts/tsp100.ckpt" \
--sampling_steps 4 --two_opt_steps 2 --cycles 10 \
--runs 8 --batch_size 5000 \
--random_two_opt_steps_range "(25, 75)" \
--threads_over_batches 2 \
--argument_level 1 > gensco_tsp100_eval.txt 2>&1 &
