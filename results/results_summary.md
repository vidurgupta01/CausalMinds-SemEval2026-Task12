# SemEval 2026 Task 12 - Experiment Results

Generated: 2026-01-26T03:43:31.473960

## Summary Table

| Model | Prompt | Docs | Split | N | Full Match | Partial | Incorrect | Avg Score |
|-------|--------|------|-------|---|------------|---------|-----------|-----------|
| gpt-4o | zero_shot | Yes | dev | 5 | 60.0% | 20.0% | 20.0% | 0.7000 |
| gpt-4o | zero_shot | No | dev | 400 | 59.8% | 17.0% | 23.2% | 0.6825 |
| gpt-4o | zero_shot | Yes | sample | 50 | 52.0% | 20.0% | 28.0% | 0.6200 |
| gpt-4o | chain_of_thought | No | sample | 50 | 56.0% | 18.0% | 26.0% | 0.6500 |
| gpt-4o-mini | zero_shot | No | sample | 20 | 40.0% | 5.0% | 55.0% | 0.4250 |

## Experiment Details

### gpt-4o_zero_shot_with_docs_dev_results.json

- Model: gpt-4o
- Prompt: zero_shot
- Include docs: True
- Split: dev
- Samples: 5
- **Average Score: 0.7000**

### gpt-4o_zero_shot_no_docs_dev_results.json

- Model: gpt-4o
- Prompt: zero_shot
- Include docs: False
- Split: dev
- Samples: 400
- **Average Score: 0.6825**

### gpt-4o_zero_shot_with_docs_sample_results.json

- Model: gpt-4o
- Prompt: zero_shot
- Include docs: True
- Split: sample
- Samples: 50
- **Average Score: 0.6200**

### gpt-4o_chain_of_thought_no_docs_sample_results.json

- Model: gpt-4o
- Prompt: chain_of_thought
- Include docs: False
- Split: sample
- Samples: 50
- **Average Score: 0.6500**

### gpt-4o-mini_zero_shot_no_docs_sample_results.json

- Model: gpt-4o-mini
- Prompt: zero_shot
- Include docs: False
- Split: sample
- Samples: 20
- **Average Score: 0.4250**

