import json
import argparse
import re

import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from rouge_score import rouge_scorer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--metrics_output_file", type=str, required=True)    

    # for specific tasks
    parser.add_argument("--label_lang", type=str, default="en") # for text_classification
    parser.add_argument("--tgt_lang", type=str, default="en") # for translation

    args = parser.parse_args()

    input_data = json.load(open(args.input_file, "r")) # list
    pred_data = json.load(open(args.pred_file, "r")) # dict 

    for i in range(len(input_data)):
        if input_data[i]['id'] not in pred_data:
            print(f"Missing prediction for ID: {input_data[i]['id']}")
            pred_data[input_data[i]['id']] = ""


    if args.task == "text_classification":
        assert len(input_data) == len(pred_data)
        correct = 0
        for i in range(len(input_data)):
            if input_data[i]["label"][args.label_lang].lower() == pred_data[input_data[i]['id']].lower():
                correct += 1
        accuracy = correct / len(input_data)

        metrics = {
            "accuracy": accuracy
        }
    elif args.task == 'translation':
        assert len(input_data) == len(pred_data)
        # bleu and chrf
        refs = [d[args.tgt_lang] for d in input_data]
        preds = [pred_data[d['id']] for d in input_data]

        if args.tgt_lang == 'zh':
            chrfpp = CHRF(word_order=2, lowercase=True)
            chrf = CHRF(word_order=0, lowercase=True)
            scarebleu = BLEU(lowercase=True, tokenize='zh')
        else:
            chrfpp = CHRF(word_order=2, lowercase=True)
            chrf = CHRF(word_order=0, lowercase=True)
            scarebleu = BLEU(lowercase=True)
        
        bleu_score = scarebleu.corpus_score(preds, [refs])
        chrf_score = chrf.corpus_score(preds, [refs])
        chrfpp_score = chrfpp.corpus_score(preds, [refs])

        metrics = {
            "bleu": bleu_score.score/100,
            "chrf": chrf_score.score/100,
            "chrf++": chrfpp_score.score/100
        }
    elif args.task == 'title_generation':
        # use implementation from https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring
        rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
        rouge_scores = []
        for i in range(len(input_data)):
            ref = input_data[i]["title"]
            pred = pred_data[input_data[i]["id"]]
            score = rouge.score(ref, pred)

            rouge_scores.append(score)
        rouge1 = np.mean([score["rouge1"].fmeasure for score in rouge_scores])
        rouge2 = np.mean([score["rouge2"].fmeasure for score in rouge_scores])
        rougel = np.mean([score["rougeL"].fmeasure for score in rouge_scores])

        metrics = {
            "rouge-1": rouge1,
            "rouge-2": rouge2,
            "rouge-l": rougel
        }
    elif args.task in ['response_selection', 'reading_comprehension', 'vocabulary']:
        correct = 0
        for i in range(len(input_data)):
            if len(pred_data[input_data[i]["id"]]) > 0:
                if pred_data[input_data[i]["id"]][0] in ["A", "B", "C", "D", "E"]:
                    pred_data[input_data[i]["id"]] = pred_data[input_data[i]["id"]][0]

            if input_data[i]["answer"] == pred_data[input_data[i]["id"]]:
                correct += 1
        
        accuracy = correct / len(input_data)

        metrics = {
            "accuracy": accuracy
        }
    elif args.task == 'math':
        correct = 0
        for i in range(len(input_data)):
            def get_answer(s):
                s = s.replace(",", "")
                if len(s) > 0 and s[-1] == '.':
                    s = s[:-1]

                extracted_nums = re.findall(r"\d+\.\d+|\d+", s)
                if len(extracted_nums) == 0:
                    return -1
                return float(extracted_nums[-1])

            
            pred = get_answer(pred_data[input_data[i]["id"]])
            if float(input_data[i]["answer"]) == float(pred):
                correct += 1
        
        accuracy = correct / len(input_data)
        metrics = {
            "accuracy": accuracy
        }

    print(metrics)
    with open(args.metrics_output_file, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
