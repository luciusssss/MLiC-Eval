# MLiC-Eval
MLiC-Eval is an NLP evaluation suite for **M**inority **L**anguages **i**n **C**hina, covering Tibetan (bo), Uyghur (ug), Kazakh (kk, in the Kazakh Arabic script), and Mongolian (mn, in the traditional Mongolian script).

The dataset is collected with the help of native speakers, and is designed to evaluate the performance of large language models on minority languages in China.
We will provide more details of data collection and annotation in our paper. Please stay tuned!

## Statistics

### Tasks

Currently, MLiC-Eval consists of 7 tasks and 4 languages, with a total of 18K instances. The statistics of each task are shown in the following table.

| Task | Size | Metric | Description |
| --- | --- | --- | --- |
| Response Selection | 507/lang | Accuracy | Given a query and a set of responses, the task is to select the correct response. |
| Text Classification | 600/lang | Accuracy | Given a text, the task is to classify it into one of the predefined categories. |
| Title Generation | 1,000/lang | ROUGE | Given a text, the task is to generate a title for it. |
| Machine Translation (Web Articles) | 1,012/lang | BLEU, chrF | Given a text in the source language, the task is to translate it into the target language. |
| Machine Translation (Dialogues) | 773/lang | BLEU, chrF | Given a text in the source language, the task is to translate it into the target language. |
| Reading Comprehension | 250/lang | Accuracy | Given a passage and a set of questions, the task is to answer the questions. |
| Math Reasoning | 250/lang | Accuracy | Given a math problem, the task is to solve it. |

### Data Splits
For each task, we provide a data split, including training, development, and test sets.

The training sets are small, used for in-context learning. For each task, we provide three training sets sampled with different seeds, to reduce the impact of randomness during prompting.

The development sets are used for hyperparameter tuning. The test sets are used for evaluation.

For each language, the data split is shown in the following table.

| Task | Train | Dev | Test |
| --- | --- | --- | --- |
| Response Selection | 20 * 3 | 40 | 407 |
| Text Classification | 16 * 3 | 48 | 504 |
| Title Generation | 20 * 3 | 40 | 900 |
| Machine Translation (Web Articles) | 20 * 3 | 40 | 912 |
| Machine Translation (Dialogues) | 20 * 3 | 40 | 673 |
| Reading Comprehension | 10 * 3 | 20 | 200 |
| Math Reasoning | 10 * 3 | 20 | 200 |


## Usage
### Download
The dataset can be downloaded from [Hugging Face](https://huggingface.co/datasets/pkupie/mlic-eval).

### Pretraining Corpus
Current LLMs have limited performance on minority languages due to the lack of pretraining data. 
We provide a pretraining corpus, MC^2 for the four languages in MLiC-Eval. 

The corpus can be downloaded from [Hugging Face](https://huggingface.co/datasets/pkupie/mc2_corpus).
You can read the details of the corpus in our paper [MC^2: Towards Transparent and Culturally-Aware NLP for Minority Languages in China](https://arxiv.org/abs/2311.08348) (ACL 2024).


## Citation
If you use MLiC-Eval in your research, please cite the our GitHub repository:
```bibtex
@misc{mlic_eval,
  author = {Zhang, Chen and Tao, Mingxu and Feng, Yansong},
  title = {{MLiC-Eval}: {A}n {NLP} {E}valuation {S}uite for {M}inority {L}anguages in {C}hina},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/luciusssss/MLiC-Eval}},
}
```