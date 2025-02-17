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
python infer.py \
    --task vocabulary \
    --model_type ${model_type} \
    --model_path ${model_path} \
    --exemplar_file ./data/vocabulary/${eval_lang}/train_${seed}.json \
    --input_file ./data/vocabulary/${eval_lang}/test.json \
    --output_file ./output/${model_name}/vocabulary/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --max_new_tokens 5 \
    --eval_lang ${eval_lang} \
    --num_exemplar 5 \
    --prompt_lang ${prompt_lang} \
    # --print_inference_result 
done


# text classification (sentence)
echo "==== Text Classification (Sentence) ===="
for seed in 1 2 3
do
echo "Seed: $seed"
python infer.py \
    --task text_classification \
    --model_type ${model_type} \
    --model_path ${model_path} \
    --exemplar_file ./data/text_classification_sentence/${eval_lang}/train_${seed}.json \
    --input_file ./data/text_classification_sentence/${eval_lang}/test.json \
    --output_file ./output/${model_name}/text_classification_sentence/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --max_new_tokens 5 \
    --num_exemplar_per_label 1 \
    --eval_lang ${eval_lang} \
    --max_passage_len 512 \
    --prompt_lang ${prompt_lang} \
    # --print_inference_result
done


# text classification (passage)
echo "==== Text Classification (Passage) ===="
for seed in 1 2 3
do
echo "Seed: $seed"
python infer.py \
    --task text_classification \
    --model_type ${model_type} \
    --model_path ${model_path} \
    --exemplar_file ./data/text_classification_passage/${eval_lang}/train_${seed}.json \
    --input_file ./data/text_classification_passage/${eval_lang}/test.json \
    --output_file ./output/${model_name}/text_classification_passage/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --max_new_tokens 5 \
    --num_exemplar_per_label 1 \
    --eval_lang ${eval_lang} \
    --max_passage_len 512 \
    --prompt_lang ${prompt_lang} \
    # --print_inference_result
done


# reading comprehension
echo "==== Reading Comprehension ===="
for seed in 1 2 3
do
echo "Seed: $seed"
python infer.py \
    --task reading_comprehension \
    --model_type ${model_type} \
    --model_path ${model_path} \
    --exemplar_file ./data/reading_comprehension/${eval_lang}/train_${seed}.json \
    --input_file ./data/reading_comprehension/${eval_lang}/test.json \
    --output_file ./output/${model_name}/reading_comprehension/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --max_new_tokens 5 \
    --eval_lang ${eval_lang} \
    --num_exemplar 5 \
    --prompt_lang ${prompt_lang} \
    # --print_inference_result 
done


# response selection
echo "==== Response Selection ===="
for seed in 1 2 3
do
echo "Seed: $seed"
python infer.py \
    --task response_selection \
    --model_type ${model_type} \
    --model_path ${model_path} \
    --exemplar_file ./data/response_selection/${eval_lang}/train_${seed}.json \
    --input_file ./data/response_selection/${eval_lang}/test.json \
    --output_file ./output/${model_name}/response_selection/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --max_new_tokens 5 \
    --eval_lang ${eval_lang} \
    --num_exemplar 5 \
    --prompt_lang ${prompt_lang} \
    # --print_inference_result 
done


# title generation
echo "==== Title Generation ===="
for seed in 1 2 3
do
echo "Seed: $seed"
python infer.py \
    --task title_generation \
    --model_type ${model_type} \
    --model_path ${model_path} \
    --exemplar_file ./data/title_generation/${eval_lang}/train_${seed}.json \
    --input_file ./data/title_generation/${eval_lang}/test.json \
    --output_file ./output/${model_name}/title_generation/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --max_new_tokens 100 \
    --num_exemplar 3 \
    --eval_lang ${eval_lang} \
    --max_passage_len 768 \
    --prompt_lang ${prompt_lang} \
    # --print_inference_result
done


# translation (article)
echo "==== Translation (Article) ===="
for seed in 1 2 3
do
echo "Seed: $seed"
# eval_lang -> prompt_lang
python infer.py \
    --task translation \
    --model_type ${model_type} \
    --model_path ${model_path} \
    --exemplar_file ./data/translation_article/${eval_lang}/train_${seed}.json \
    --input_file ./data/translation_article/${eval_lang}/test.json \
    --output_file ./output/${model_name}/translation_article/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${eval_lang}2${prompt_lang}_test.json \
    --max_new_tokens 100 \
    --num_exemplar 5 \
    --src_lang ${eval_lang} \
    --tgt_lang ${prompt_lang} \
    --eval_lang ${eval_lang} \
    --prompt_lang ${prompt_lang} \
    # --print_inference_result

# prompt_lang -> eval_lang
python infer.py \
    --task translation \
    --model_type ${model_type} \
    --model_path ${model_path} \
    --exemplar_file ./data/translation_article/${eval_lang}/train_${seed}.json \
    --input_file ./data/translation_article/${eval_lang}/test.json \
    --output_file ./output/${model_name}/translation_article/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${prompt_lang}2${eval_lang}_test.json \
    --max_new_tokens 100 \
    --num_exemplar 5 \
    --src_lang ${prompt_lang} \
    --tgt_lang ${eval_lang} \
    --eval_lang ${eval_lang} \
    --prompt_lang ${prompt_lang} \
    # --print_inference_result
done


# translation (dialogue)
echo "==== Translation (Dialogue) ===="
for seed in 1 2 3
do
echo "Seed: $seed"
# eval_lang -> prompt_lang
python infer.py \
    --task translation \
    --model_type ${model_type} \
    --model_path ${model_path} \
    --exemplar_file ./data/translation_dialogue/${eval_lang}/train_${seed}.json \
    --input_file ./data/translation_dialogue/${eval_lang}/test.json \
    --output_file ./output/${model_name}/translation_dialogue/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${eval_lang}2${prompt_lang}_test.json \
    --max_new_tokens 70 \
    --num_exemplar 5 \
    --src_lang ${eval_lang} \
    --tgt_lang ${prompt_lang} \
    --eval_lang ${eval_lang} \
    --prompt_lang ${prompt_lang} \
    # --print_inference_result

# prompt_lang -> eval_lang
python infer.py \
    --task translation \
    --model_type ${model_type} \
    --model_path ${model_path} \
    --exemplar_file ./data/translation_dialogue/${eval_lang}/train_${seed}.json \
    --input_file ./data/translation_dialogue/${eval_lang}/test.json \
    --output_file ./output/${model_name}/translation_dialogue/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${prompt_lang}2${eval_lang}_test.json \
    --max_new_tokens 70 \
    --num_exemplar 5 \
    --src_lang ${prompt_lang} \
    --tgt_lang ${eval_lang} \
    --eval_lang ${eval_lang} \
    --prompt_lang ${prompt_lang} \
    # --print_inference_result
done

# math
echo "==== Math ===="
for seed in 1 2 3
do
echo "Seed: $seed"
python infer.py \
    --task math \
    --model_type ${model_type} \
    --model_path ${model_path} \
    --exemplar_file ./data/math/${eval_lang}/train_${seed}.json \
    --input_file ./data/math/${eval_lang}/test.json \
    --output_file ./output/${model_name}/math/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --max_new_tokens 150 \
    --eval_lang ${eval_lang} \
    --num_exemplar 5 \
    --prompt_lang ${prompt_lang} \
    # --print_inference_result
done
