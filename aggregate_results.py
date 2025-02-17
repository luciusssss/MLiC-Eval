import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--eval_lang", type=str, required=True)
    parser.add_argument("--prompt_lang", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--seed_list", type=str, default="1,2,3")
    args = parser.parse_args()

    output = {}

    seed_list = [int(s) for s in args.seed_list.split(",")]

    # load the results
    for task in ["vocabulary", "text_classification_sentence", "text_classification_passage", "response_selection", "reading_comprehension", "title_generation", "translation_article", "translation_dialogue", "math"]:
        output[task] = {}

        for seed in seed_list:
            try:
                if "translation" in task:
                    output[task][f"seed{seed}"] = {}
                    # eval_lang -> prompt_lang
                    metric_file = f"./output/{args.model_name}/{task}/{args.eval_lang}/{args.prompt_lang}-prompt_seed{seed}_{args.eval_lang}2{args.prompt_lang}_test_metrics.json"
                    metric_result = json.load(open(metric_file))
                    for k, v in metric_result.items():
                        output[task][f"seed{seed}"][f"xx2{args.prompt_lang}_{k}"] = v

                    # prompt_lang -> eval_lang
                    metric_file = f"./output/{args.model_name}/{task}/{args.eval_lang}/{args.prompt_lang}-prompt_seed{seed}_{args.prompt_lang}2{args.eval_lang}_test_metrics.json"
                    metric_result = json.load(open(metric_file))
                    for k, v in metric_result.items():
                        output[task][f"seed{seed}"][f"{args.prompt_lang}2xx_{k}"] = v
                else:
                    metric_file = f"./output/{args.model_name}/{task}/{args.eval_lang}/{args.prompt_lang}-prompt_seed{seed}_test_metrics.json"
                    metric_result = json.load(open(metric_file))
                    output[task][f"seed{seed}"] = metric_result
            except Exception as e:
                print(e)
                output[task][f"seed{seed}"] = None

        # print(task, output[task])
        # calculate mean
        output[task]["mean"] = {}
        try:
            for k in output[task][f"seed{seed_list[0]}"].keys():
                output[task]["mean"][k] = sum([output[task][f"seed{seed}"][k] for seed in seed_list]) / len(seed_list)
        except:
            output[task]["mean"] = None
        

    # get average of all tasks
    try:
        output["aggregated"] = {}
        output["aggregated"]["vocabulary"] = output["vocabulary"]["mean"]["accuracy"]
        output["aggregated"]["text_classification_sentence"] = output["text_classification_sentence"]["mean"]["accuracy"]
        output["aggregated"]["text_classification_passage"] = output["text_classification_passage"]["mean"]["accuracy"]
        output["aggregated"]["reading_comprehension"] = output["reading_comprehension"]["mean"]["accuracy"]
        output["aggregated"]["response_selection"] = output["response_selection"]["mean"]["accuracy"]
        output["aggregated"]["title_generation"] = output["title_generation"]["mean"]["rouge-l"]
        output["aggregated"][f"translation_article_xx2{args.prompt_lang}"] = output["translation_article"]["mean"][f"xx2{args.prompt_lang}_chrf++"]
        output["aggregated"][f"translation_article_{args.prompt_lang}2xx"] = output["translation_article"]["mean"][f"{args.prompt_lang}2xx_chrf++"]
        output["aggregated"]["translation_article"] = (output["translation_article"]["mean"][f"xx2{args.prompt_lang}_chrf++"] + output["translation_article"]["mean"][f"{args.prompt_lang}2xx_chrf++"]) / 2   
        output["aggregated"][f"translation_dialogue_xx2{args.prompt_lang}"] = output["translation_dialogue"]["mean"][f"xx2{args.prompt_lang}_chrf++"]
        output["aggregated"][f"translation_dialogue_{args.prompt_lang}2xx"] = output["translation_dialogue"]["mean"][f"{args.prompt_lang}2xx_chrf++"]
        output["aggregated"]["translation_dialogue"] = (output["translation_dialogue"]["mean"][f"xx2{args.prompt_lang}_chrf++"] + output["translation_dialogue"]["mean"][f"{args.prompt_lang}2xx_chrf++"]) / 2
        output["aggregated"]["math"] = output["math"]["mean"]["accuracy"]

        output["aggregated"]["mean"] = sum([output["aggregated"][task] for task in ["vocabulary", "text_classification_sentence", "text_classification_passage", "reading_comprehension", "response_selection",   "title_generation", "translation_article", "translation_dialogue", "math"]]) / 9

        # round to 4 decimal places
        for k, v in output["aggregated"].items():
            output["aggregated"][k] = round(v, 4)
    except Exception as e:
        print(e)
        output["aggregated"] = None

    print(output['aggregated'])
    json.dump(output, open(args.output_file, "w"), indent=4, ensure_ascii=False)
