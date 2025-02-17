model_type="qwen" # model type: qwen, aya, llama, mistral, gemma
model_path="/path/to/Qwen2.5-7B-Instruct"
model_name="qwen2.5-7b-instruct" # model name, used as the directory name to save the inference results
prompt_lang="en" # prompt language: en, zh
eval_lang="bo" # evaluation language: bo, ug, kk, mn

cd ../

# vocabulary understanding
echo "==== Vocabulary Understanding ===="
for seed in 1 2 3
do
echo "Seed: $seed"
python eval.py \
    --task vocabulary \
    --input_file ./data/vocabulary/${eval_lang}/test.json \
    --pred_file ./output/${model_name}/vocabulary/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --metrics_output_file ./output/${model_name}/vocabulary/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test_metrics.json 
done

# text classification (sentence)
echo "==== Text Classification (Sentence) ===="
for seed in 1 2 3
do
echo "Seed: $seed"
python eval.py \
    --task text_classification \
    --input_file ./data/text_classification_sentence/${eval_lang}/test.json \
    --pred_file ./output/${model_name}/text_classification_sentence/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --metrics_output_file ./output/${model_name}/text_classification_sentence/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test_metrics.json \
    --label_lang ${prompt_lang}
done


# text classification (passage)
echo "==== Text Classification (Passage) ===="
for seed in 1 2 3
do
echo "Seed: $seed"
python eval.py \
    --task text_classification \
    --input_file ./data/text_classification_passage/${eval_lang}/test.json \
    --pred_file ./output/${model_name}/text_classification_passage/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --metrics_output_file ./output/${model_name}/text_classification_passage/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test_metrics.json \
    --label_lang ${prompt_lang}
done


# reading comprehension
echo "==== Reading Comprehension ===="
for seed in 1 2 3
do
echo "Seed: $seed"
python eval.py \
    --task reading_comprehension \
    --input_file ./data/reading_comprehension/${eval_lang}/test.json \
    --pred_file ./output/${model_name}/reading_comprehension/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --metrics_output_file ./output/${model_name}/reading_comprehension/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test_metrics.json
done


# response selection
echo "==== Response Selection ===="
for seed in 1 2 3
do
echo "Seed: $seed"
python eval.py \
    --task response_selection \
    --input_file ./data/response_selection/${eval_lang}/test.json \
    --pred_file ./output/${model_name}/response_selection/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --metrics_output_file ./output/${model_name}/response_selection/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test_metrics.json 
done


# title generation
echo "==== Title Generation ===="
for seed in 1 2 3
do
echo "Seed: $seed"
python eval.py \
    --task title_generation \
    --input_file ./data/title_generation/${eval_lang}/test.json \
    --pred_file ./output/${model_name}/title_generation/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --metrics_output_file ./output/${model_name}/title_generation/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test_metrics.json
done


# translation (article)
echo "==== Translation (Article) ===="
for seed in 1 2 3
do
echo "Seed: $seed"
# eval_lang -> prompt_lang
python eval.py \
    --task translation \
    --tgt_lang ${prompt_lang} \
    --input_file ./data/translation_article/${eval_lang}/test.json \
    --pred_file ./output/${model_name}/translation_article/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${eval_lang}2${prompt_lang}_test.json \
    --metrics_output_file ./output/${model_name}/translation_article/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${eval_lang}2${prompt_lang}_test_metrics.json

# prompt_lang -> eval_lang
python eval.py \
    --task translation \
    --tgt_lang ${eval_lang} \
    --input_file ./data/translation_article/${eval_lang}/test.json \
    --pred_file ./output/${model_name}/translation_article/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${prompt_lang}2${eval_lang}_test.json \
    --metrics_output_file ./output/${model_name}/translation_article/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${prompt_lang}2${eval_lang}_test_metrics.json
done


# translation (dialogue)
echo "==== Translation (Dialogue) ===="
for seed in 1 2 3
do
echo "Seed: $seed"
# eval_lang -> prompt_lang
python eval.py \
    --task translation \
    --tgt_lang ${prompt_lang} \
    --input_file ./data/translation_dialogue/${eval_lang}/test.json \
    --pred_file ./output/${model_name}/translation_dialogue/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${eval_lang}2${prompt_lang}_test.json \
    --metrics_output_file ./output/${model_name}/translation_dialogue/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${eval_lang}2${prompt_lang}_test_metrics.json

# prompt_lang -> eval_lang
python eval.py \
    --task translation \
    --tgt_lang ${eval_lang} \
    --input_file ./data/translation_dialogue/${eval_lang}/test.json \
    --pred_file ./output/${model_name}/translation_dialogue/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${prompt_lang}2${eval_lang}_test.json \
    --metrics_output_file ./output/${model_name}/translation_dialogue/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${prompt_lang}2${eval_lang}_test_metrics.json
done


# math
echo "==== Math ===="
for seed in 1 2 3
do
echo "Seed: $seed"
python eval.py \
    --task math \
    --input_file ./data/math/${eval_lang}/test.json \
    --pred_file ./output/${model_name}/math/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --metrics_output_file ./output/${model_name}/math/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test_metrics.json
done

# aggregate metrics
echo "==== Aggregate Metrics ===="
python aggregate_results.py \
--model_name ${model_name} \
--eval_lang ${eval_lang} \
--prompt_lang ${prompt_lang} \
--output_file ./output/${model_name}/aggregated_metrics_${eval_lang}_${prompt_lang}-prompt.json \
--seed_list 1,2,3