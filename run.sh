context=$1
test=$2
output=$3

python3.8 preprocess_for_test.py --context $context --test $test

python3.8 mc_infer.py --mc_model ./mc_model.bin \
                --mc_token_json ./mc_tokenizer \
                --mc_config ./mc_config.json \
                --batch 8 \
                --test_file ./my_cache/mcqa_test_sent12.json \
                --output ./my_cache/qa_test_sent12.json
python3.8 qa_infer.py --qa_model ./qa_model.bin \
                --qa_token_json ./qa_tokenizer \
                --qa_config ./qa_config.json \
                --batch 8 \
                --test_file ./my_cache/qa_test_sent12.json \
                --output $output