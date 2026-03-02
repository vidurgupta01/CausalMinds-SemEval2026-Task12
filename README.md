# CausalMinds @ SemEval-2026 Task 12

Code for our system submission to [SemEval-2026 Task 12: Abductive Event Reasoning](https://sites.google.com/view/semeval2026-task12/introduction).

**TL;DR** Fine-tuning GPT-4.1-mini with option shuffling augmentation. Dev: 0.991, Test: 0.88 (19th place, [CodaBench submission 509563](https://www.codabench.org/competitions/12446/#/pages-tab)). Simple fine-tuning beat every prompting/RAG/verification strategy we tried.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your OpenAI API key to .env
```

Put the official task data in `data/official/` (we include dev and sample splits).

## Reproducing our results

The pipeline that gets 0.88 on test:

```bash
# 1. augment training data (shuffles answer option positions, 1819 -> 3638 examples)
python augment_data.py

# 2. prepare data for OpenAI fine-tuning format
python prepare_finetune.py

# 3. fine-tune GPT-4.1-mini (takes ~15 min, costs a few dollars)
python run_finetune.py

# 4. evaluate on dev set
python eval_augmented.py

# 5. generate test submission
python generate_submission.py --model augmented_ft --split test
```

## What worked

Fine-tuning GPT-4.1-mini on augmented data for 3 epochs. The augmentation just shuffles the multiple-choice option positions and remaps the gold labels, which forces the model to learn actual causal relationships instead of memorizing that answer A is usually correct.

| Configuration | Dev | Test |
|---|---|---|
| Fine-tuned GPT-4.1-mini | 0.965 | -- |
| + option shuffling | 0.991 | 0.88 |
| + ensemble (3 models) | 0.993 | 0.88 |

## What didn't work

We tried 23 configurations in total. Most of the "fancy" stuff made things worse:

- Chain-of-thought: -4.7% (reasoning introduces error propagation)
- Raw document context: -9.1% (documents contain distractors)
- Multi-stage verification: -74.3% (verifier changes correct answers to wrong ones)
- OpenAI o1: -78% (reasoning model completely failed this task)
- GPT-4.1 full (larger model): worse than mini (0.943 vs 0.965)

See `experiments/EXPERIMENT_LOG.md` for detailed notes on all 23 experiments.

## Repo structure

```
├── augment_data.py          # option shuffling augmentation
├── prepare_finetune.py      # format data for OpenAI API
├── run_finetune.py          # launch fine-tuning job
├── eval_augmented.py        # evaluate on dev
├── eval_model.py            # evaluate any model config
├── generate_submission.py   # generate test predictions
├── ensemble_predict.py      # majority vote ensemble
├── analyze_errors.py        # error analysis on dev
├── analyze_results.py       # aggregate results
├── src/                     # shared utilities (data loading, LLM wrapper)
├── exploration/             # all 23+ experiment scripts (prompting, RAG, etc.)
├── data/
│   ├── official/            # task data (dev, sample, test splits)
│   ├── finetune/            # formatted for OpenAI fine-tuning
│   └── augmented/           # augmented training data
├── experiments/             # raw results from all configs
├── results/                 # summary tables and error analysis
└── submissions/             # our actual submission files
```

## Citation

```bibtex
@misc{gupta2026causalminds,
    title = "CausalMinds at SemEval-2026 Task 12: Fine-tuning with Data Augmentation for Abductive Event Reasoning",
    author = "Vidur Gupta and Xiaofei and Jason",
    year = "2026",
    url = "https://github.com/vidurgupta01/CausalMinds-SemEval2026-Task12",
}
```

## Team

Vidur, Xiaofei, Jason -- Stanford University
