# sst5
# for ICL -- random ensemble
python ensemble_postprocess.py \
--mode icl \
--model_version large \
--data sst5 \
--task sst5 \
--train_size 50 \
--input_format sst5 \
--esb_num 20 \
--use_logprobs \
--keep_distinct \
--strategy majority_vote  \
--do_inference \
--train_seeds 0 21 42 \
--output OUTPUT_PATH

#! for ICL -- different prompt ensemble
python ensemble_postprocess.py \
--mode icl \
--model_version large \
--data sst5 \
--task sst5 \
--train_size 50 \
--use_logprobs \
--keep_distinct \
--strategy mean_prob \
--do_inference \
--train_seeds 0 21 42 OUTPUT_PATH

#! for FT -- different prompt ensemble
python ensemble_postprocess.py \
--mode ft \
--model_version large \
--data sst5 \
--task sst5 \
--train_size 50 \
--train_seeds 0 21 42 \
--use_logprobs \
--keep_distinct \
--strategy mean_prob \
--do_tune \
--do_inference \
--output OUTPUT_PATH

#! for SupICL -- different prompt template
python ensemble_postprocess.py \
--mode supicl \
--model_version large \
--data sst5 \
--task sst5 \
--train_size 50 \
--train_seeds 0 21 42 \
--use_logprobs \
--keep_distinct \
--strategy mean_prob \
--do_tune \
--do_inference \
--output OUTPUT_PATH


#! for SupICL -- random ensemble
python ensemble_postprocess.py \
--mode supicl \
--model_version large \
--data sst5 \
--task sst5 \
--input_format sst5 \
--train_size 50 \
--train_seeds 0 21 42 \
--use_logprobs \
--keep_distinct \
--strategy mean_prob \
--esb_num 20 \
--do_tune \
--do_inference \
--output OUTPUT_PATH

#! for SupICL and ICL -- ensemble both different prompt template and random seeds
python ensemble_all.py \
--do_tune \
--do_inference \
--mode supicl \
--model_version large \
--data sst5 \
--task sst5 \
--train_size 50 \
--use_logprobs \
--keep_distinct \
--strategy mean_prob \
--train_seeds 0 21 42 \
--output OUTPUT_PATH

python ensemble_all.py \
--mode icl \
--do_inference \
--model_version large \
--data sst5 \
--task sst5 \
--train_size 50 \
--use_logprobs \
--keep_distinct \
--strategy max_prob \
--train_seeds 0 \
--output OUTPUT_PATH