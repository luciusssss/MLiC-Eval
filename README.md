# MiLiC-Eval
MiLiC-Eval is an NLP evaluation suite for **Mi**nority **L**anguages **i**n **C**hina, covering Tibetan (bo), Uyghur (ug), Kazakh (kk, in the Kazakh Arabic script), and Mongolian (mn, in the traditional Mongolian script).

## Statistics

### Tasks

Currently, MiLiC-Eval consists of 9 tasks and 4 languages, with 24K instances. The statistics of each task are shown in the following table.

| Task | Size | Metric | Languages |
| --- | --- | --- | --- |
| Vocabulary Understanding | 1,000/lang | Accuracy | bo, ug, kk, mn |
| Topic Classification (Sentence) | 492/lang | Accuracy | bo, ug, kk, mn, zh, en |
| Topic Classification (Passage) | 600/lang | Accuracy | bo, ug, kk, mn |
| Reading Comprehension | 250/lang | Accuracy | bo, ug, kk, mn, zh, en |
| Response Selection | 507/lang | Accuracy | bo, ug, kk, mn, zh, en |
| Title Generation | 1,000/lang | ROUGE-L | bo, ug, kk, mn |
| Machine Translation (Article) | 1,012/lang | chrF++ | bo, ug, kk, mn, zh, en |
| Machine Translation (Dialogue) | 773/lang | chrF++ | bo, ug, kk, mn, zh, en |
| Math Reasoning | 250/lang | Accuracy | bo, ug, kk, mn, zh, en |

### Data Splits
For each task, we provide a data split, including training, development, and test sets.

The training sets are small and used for in-context learning. For each task, we provide three training sets sampled with different seeds, to reduce the impact of randomness during prompting.

The development sets are used for hyperparameter tuning. The test sets are used for evaluation.

For each language, the data split is shown in the following table.

| Task | Train | Dev | Test |
| --- | --- | --- | --- |
| Vocabulary Understanding | 20 * 3 | 40 | 900 |
| Topic Classification (Sentence) | 10 * 3 | 30 | 432 |
| Topic Classification (Passage) | 16 * 3 | 48 | 504 |
| Reading Comprehension | 10 * 3 | 20 | 200 |
| Response Selection | 20 * 3 | 40 | 407 |
| Title Generation | 20 * 3 | 40 | 900 |
| Machine Translation (Article) | 20 * 3 | 40 | 912 |
| Machine Translation (Dialogue) | 20 * 3 | 40 | 673 |
| Math Reasoning | 10 * 3 | 20 | 200 |


## Usage

### Download
The dataset can be downloaded from [Hugging Face](https://huggingface.co/datasets/pkupie/milic-eval).
Put the downloaded dataset in the `data` directory.

## Setup
1. Install the packages required for inference by running:
```bash
pip install -r requirements.txt
```

2. Install the package required by multilingual ROUGE scoring. 
See https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring

3. Run the scripts for inference and metric calculation (with Qwen-2.5 as an example):
```bash
cd scripts
bash run_eval.sh
bash calculate_metrics.sh
```
The evaluation results will be saved in the `output` directory.


### Pretraining Corpus
Current LLMs have limited performance in minority languages due to the lack of pretraining data. 
We provide a pretraining corpus, MC^2 for the four languages in MiLiC-Eval. 

The corpus can be downloaded from [Hugging Face](https://huggingface.co/datasets/pkupie/mc2_corpus).
You can read the details of the corpus in our paper [MC^2: Towards Transparent and Culturally-Aware NLP for Minority Languages in China](https://aclanthology.org/2024.acl-long.479.pdf) (ACL 2024).


## Citation
If you use MiLiC-Eval in your research, please cite our GitHub repository:
```bibtex
@misc{milic_eval,
  author = {Zhang, Chen and Tao, Mingxu and Feng, Yansong},
  title = {{MiLiC-Eval}: {A}n {NLP} {E}valuation {S}uite for {M}inority {L}anguages in {C}hina},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/luciusssss/MiLiC-Eval}},
}
```
