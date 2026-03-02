# Experiment Log

Running log of everything we tried for SemEval 2026 Task 12 (Abductive Event Reasoning).

The task: given an event and 4 candidate causes, pick which one(s) actually caused it. Can be multi-label (1-4 correct answers, 43% of training has multiple). Scoring is 1.0 for exact match, 0.5 for partial subset, 0.0 for wrong. We have 1819 train, 400 dev, 612 test, 200 sample examples.

We tried 23 different configurations. Spoiler: simple fine-tuning won. Everything else was worse.

---

## Baseline Results

### GPT-4o-mini on Sample (20 samples)
- Average Score: 0.4250
- Full Match: 40%
- Notes: Quick sanity check, confirmed task is non-trivial

### GPT-4o Zero-Shot on Sample (50 samples) **[BEST RESULT]**
- Average Score: **0.7800**
- Full Match: 72.0%
- Partial: 12.0%
- Wrong: 16.0%
- Notes: Best result achieved - simple baseline wins

### GPT-4o Zero-Shot on Dev (400 samples)
- Average Score: **0.6825**
- Full Match: 59.8%
- Partial: 17.0%
- Wrong: 23.25%
- Notes: Simple zero-shot prompt performs well

---

## Optimization Attempts

### Chain-of-Thought on Sample (50)
- Average Score: 0.6500
- Notes: Did NOT help. Explicit reasoning doesn't improve causation detection.

### With-docs on Sample (50)
- Average Score: 0.6200
- Notes: Actually HURT performance. Documents contain distractors.

### Focused Causation Prompt on Dev (400)
- Average Score: 0.6512
- Full Match: 56.0%
- Notes: WORSE than baseline despite looking good on sample (0.77). Over-engineered prompt.

### Persona-based (Domain-routed) on Dev (400)
- Average Score: 0.6362
- Full Match: 49.0%
- Notes: WORSE than baseline. Personas: historian, journalist, economist, political_analyst, crisis_expert

### Self-Consistency (5 samples, temp=0.7, threshold=0.4) on Sample (30)
- Average Score: 0.4500
- Full Match: 40.0%
- Notes: MUCH WORSE. Temperature introduces too much noise.

### Per-Option (evaluate each option independently) on Sample (30)
- Average Score: 0.5833
- Full Match: 53.3%
- Notes: WORSE than baseline. Breaking apart the question loses context.

### Claude Sonnet 4 Zero-Shot on Sample (50)
- Average Score: 0.5900
- Full Match: 48.0%
- Partial: 22.0%
- Wrong: 30.0%
- Notes: Worse than GPT-4o. Different model doesn't help.

### Multi-aware Prompt on Sample (50)
- Average Score: 0.3900
- Full Match: 34.0%
- Notes: MUCH WORSE. Explicitly asking about multiple causes backfired.

### Advanced Multi-Stage (RAG + Reasoning + Verification) on Sample (20)
- Average Score: 0.1750
- Full Match: 10.0%
- Notes: TERRIBLE. Verification stage changed correct answers to wrong ones. Multi-stage compounds errors.

### Simple Multi-Model Ensemble (GPT-4o + Claude) on Sample (30)
- Average Score: 0.2833
- Full Match: 23.3%
- Per-model: GPT-4o=0.52, Claude=0.33
- Notes: Claude drags down the ensemble. Multi-model hurts when one model is worse.

### OpenAI o1 Reasoning Model on Sample (10)
- Average Score: 0.1500
- Full Match: 10.0%
- Notes: TERRIBLE. Reasoning models don't help this task.

### Smart Router (none/disambiguation/multi routes) on Sample (50)
- Average Score: 0.4600
- Baseline comparison: 0.5200
- Notes: Router is WORSE. The "none" route is especially bad (0.20 accuracy).

### Prompt Variants Comparison on Sample (50)
- baseline: 0.5000
- v1_shorter: 0.4700
- v2_temporal: 0.5100 (+1% on sample)
- v3_explicit_multi: 0.3000
- v4_elimination: 0.3100
- Notes: Temporal emphasis helps slightly ON SAMPLE. Explicit multi-answer prompts hurt badly.

### Temporal Prompt (v2) on Dev (400)
- Average Score: **0.4750** (vs baseline 0.6825)
- Full Match: 41.2%
- Partial: 12.5%
- Wrong: 46.2%
- Notes: MUCH WORSE than baseline (-0.21 points). The +1% sample improvement did NOT generalize. Another example of sample data misleading.

---

## RAG Approaches (PROMISING!)

### Smart RAG (Embedding-based) on Sample (50) **[BEST ON SAMPLE]**
- Average Score: **0.7500**
- Full Match: 62.0%
- Baseline comparison: 0.5000 (sample)
- Improvement: **+0.25 (+50% relative improvement!)**
- Method: Use OpenAI embeddings to find causally-relevant sentences from docs
- Notes: Uses text-embedding-3-small to rank sentences by similarity to causal query. Retrieves top-k most relevant sentences as context.
- Status: **VALIDATING ON DEV SET** (running)

### Keyword RAG on Sample (30)
- Average Score: **0.6833**
- Full Match: 60.0%
- Baseline comparison: 0.5000 (sample)
- Improvement: **+0.18**
- Method: Fast keyword matching for causal terms and event-related words
- Notes: Much faster than embedding-based. Uses causal keywords (caused, because, due to, etc.)

### Keyword RAG on Dev (100) **[BEST ON DEV SO FAR]**
- Average Score: **0.7600**
- Full Match: 63.0%
- Partial: 26.0%
- Wrong: 11.0%
- Baseline comparison: 0.6825 (dev)
- Improvement: **+0.0775 (+11% relative improvement)**
- Status: **VALIDATING ON FULL DEV SET** (running)

### Structured JSON Output on Sample (30)
- Average Score: 0.6333
- Full Match: 56.7%
- Notes: JSON mode for reliable multi-label output. Decent but not as good as RAG.

### Confidence-based on Sample (30)
- Average Score: 0.4667
- Full Match: 46.7%
- Notes: Per-option confidence scoring. WORSE than baseline.

### Contrastive Prompting on Sample (30)
- Average Score: 0.3833
- Full Match: 36.7%
- Notes: Step-by-step per-option analysis. WORSE than baseline.

### Few-shot Similar Examples on Sample (30)
- Average Score: 0.4333
- Full Match: 40.0%
- Notes: Using similar training examples as demonstrations. WORSE than baseline.

---

## Error Analysis (on Baseline Dev Results)

### Error Distribution
- Correct: 239 (59.8%)
- Missing causes (partial): 68 (17.0%) - Model under-predicts
- False positives: 37 (9.25%) - Model over-predicts
- Wrong (mixed): 56 (14.0%) - Completely wrong

### Key Insight: Multi-answer Questions are Harder
- Single-answer error rate: **31.0%**
- Multi-answer error rate: **50.5%**

Model tends to **under-predict** - it misses correct causes rather than adding false ones.

---

## Error Analysis (on Fine-tuned RAG Model Dev Results)

### Overall: 16 errors out of 400 (4% error rate)

### Error Type Breakdown
| Type | Count | Percentage | Description |
|------|-------|------------|-------------|
| UNDER-PREDICT | 7 | 43.75% | Missed correct causes |
| OVER-PREDICT | 5 | 31.25% | Added incorrect causes |
| WRONG | 4 | 25.0% | Completely incorrect answer |

### Detailed Error Patterns

**1. Under-prediction of "A" in multi-answer (4/7 under-predict errors)**
- Model frequently misses option A when multiple answers are correct
- Examples: Japan water release (missed B), Capitol riot (missed A or B)

**2. Capitol riot/protest events (3 errors)**
- Model consistently misses "protests" as a cause of subsequent events
- Example: FBI/ATF cleared Capitol - missed that protests caused this

**3. Wagner rebellion (2 errors - over-prediction)**
- Model over-predicts on Wagner-related questions
- Example: Gold=C, Pred=A,C - added option A incorrectly

**4. Temporal confusion (4 wrong errors)**
- Model confuses which event came first
- Example: Iran missile strike - picked D instead of C (wrong sequence)

### Specific Errors

| ID | Event | Gold | Pred | Type |
|----|-------|------|------|------|
| q-2028 | Credit Suisse share price fell | A | A,D | OVER |
| q-2030 | Japan's second Fukushima release | B,C | C | UNDER |
| q-2109 | Iran launched missiles at bases | C | D | WRONG |
| q-2112 | HarmonyOS on phones | A | A,C | OVER |
| q-2120 | Congress meets to certify Biden | A,C | C | UNDER |
| q-2139 | Rioters stormed Capitol | B,D | D | UNDER |
| q-2142 | Lebanon minister resigned | B,C | C | UNDER |
| q-2230 | DeepSeek market freefall | A,D | C | WRONG |
| q-2231 | FBI/ATF cleared Capitol | A,C | C | UNDER |
| q-2268 | Texas water mains broke | C | B,C | OVER |
| q-2295 | Wagner turned toward Moscow | C | A,C | OVER |
| q-2320 | Floyd funeral at Houston | A,B,C | B | UNDER |
| q-2323 | Amazon fire discrepancies | A | B | WRONG |
| q-2325 | Troops to Floyd protests | A,B,C | B,C | UNDER |
| q-2408 | Wagner toward Moscow (dupe) | C | A,C | OVER |
| q-2410 | DOGE accessed Treasury | A | B | WRONG |

---

## Fine-tuning Status

### GPT-4o-mini Fine-tuning (Attempt 1)
- Job ID: `ftjob-cZIdPQn16Kaws8kIquNBtclR`
- Training samples: 1637
- Validation samples: 182
- Epochs: 3
- Status: **FAILED** - OpenAI internal server error

### GPT-4o-mini Fine-tuning (Attempt 2)
- Job ID: `ftjob-jGq0vjgVw2fOHU4tWSZPBEVg`
- Status: **FAILED** - OpenAI internal server error

### GPT-4.1-mini Fine-tuning (Attempt 3) SUCCESS
- Job ID: `ftjob-vbrHGFKhZK332gjwuKP1t0gz`
- Model: `ft:gpt-4.1-mini-2025-04-14:personal::D2UIiELC`
- Training samples: 1637
- Validation samples: 182
- Epochs: 3
- Status: **SUCCEEDED**
- **Dev Score: 0.9650** (95% full match)
- **Test Score: 0.8600** (CodaBench submission)
- Improvement over baseline: **+26%**
- Saved: `submissions/submission_finetuned_gpt41mini_test086_*.zip`

### GPT-4.1 Full Fine-tuning (Attempt 4) SUCCESS
- Job ID: `ftjob-YvFdn6E26hBg5lHt73JV53TQ`
- Model: `ft:gpt-4.1-2025-04-14:personal::D2aRBTih`
- Training samples: 1637
- Validation samples: 182
- Epochs: 3
- Status: **SUCCEEDED**
- **Dev Score: 0.9425** (93.8% full match)
- Notes: Surprisingly WORSE than mini model on dev. Full model doesn't help.

### GPT-4.1-mini with RAG Context (Attempt 5) SUCCESS **[BEST TEST SCORE]**
- Job ID: `ftjob-L7VtlASDzJdm8WTxhmtn38Ln`
- Model: `ft:gpt-4.1-mini-2025-04-14:personal::D2a6rPiD`
- Training data: `data/augmented/train_rag.jsonl` (includes keyword RAG context)
- Training samples: 1819 (all with context)
- Validation samples: 182
- Epochs: 3
- Status: **SUCCEEDED**
- **Dev Score: 0.9688** (96.0% full match)
- **Test Score: 0.8800** (CodaBench submission) (new best)
- Improvement over non-RAG: +0.02 on test
- Saved: `submissions/submission_finetuned_rag_gpt41mini_test_*.zip`

### GPT-4.1-mini with 5 Epochs (Attempt 6) SUCCESS
- Job ID: `ftjob-nAMdnQtCickHrS0AYuoPYTvg`
- Model: `ft:gpt-4.1-mini-2025-04-14:personal:5epochs:D2kZfdPv`
- Training samples: 1637
- Validation samples: 182
- Epochs: **5** (vs 3 before)
- Status: **SUCCEEDED**
- **Dev Score: 0.9587** (95.5% full match)
- Notes: WORSE than 3-epoch RAG model (0.9688). More epochs didn't help.

### GPT-4.1 Full with RAG Context (Attempt 7) SUCCESS
- Job ID: `ftjob-3hXF8pXlAgCSUOi9SrllfZ13`
- Model: `ft:gpt-4.1-2025-04-14:personal:rag:D2l6MlSx`
- Training data: `data/augmented/train_rag.jsonl` (includes keyword RAG context)
- Training samples: 1819
- Validation samples: 182
- Epochs: 3
- Status: **SUCCEEDED**
- **Dev Score: 0.9600** (95.2% full match)
- Notes: WORSE than mini + RAG (0.9688). Full model doesn't help even with RAG.

---

## takeaways so far

- simple > fancy. every "sophisticated" approach was worse than zero-shot
- sample results are misleading!! temporal prompt was +1% on sample but -21% on dev
- docs are distractors, including them hurts
- model under-predicts (misses causes rather than adding false ones)
- multi-answer questions are way harder (50.5% error rate vs 31.0% single-answer)
- verification stages change correct answers to wrong ones lol
- Claude << GPT-4o on this task (~0.33-0.59 vs ~0.68-0.78)
- mixing strong+weak models in ensemble makes things worse
- 5 epochs < 3 epochs, GPT-4.1 full < mini... less is more
- RAG helps fine-tuning slightly (0.9688 vs 0.9650)
- test score plateaus at 0.88 no matter what we do (RAG, augmented, ensemble all 0.88)
- ensemble on test = identical to single no-RAG model (612/612 same predictions)
- RAG makes model more conservative, predicts fewer multi-answer

---

## Post-Submission Analysis: RAG vs No-RAG on Test Set

### Experiment: Compare predictions with and without RAG context

| Comparison | Same | Different |
|------------|------|-----------|
| No-RAG vs With-RAG (single augmented_ft) | 604/612 (98.7%) | 8/612 (1.3%) |
| No-RAG vs Ensemble (3 models + RAG) | **612/612 (100%)** | 0/612 (0%) |
| With-RAG vs Ensemble | 604/612 (98.7%) | 8/612 (1.3%) |

### interesting: ensemble = no-RAG single model
ensemble with RAG produces **identical predictions** to single augmented_ft without RAG. basically the voting cancels out the RAG influence.

### RAG Effect Analysis
Looking at the 8 differences between RAG and No-RAG:

| Question | With-RAG | No-RAG | Change |
|----------|----------|--------|--------|
| q-2569 | D | C,D | +1 answer |
| q-2730 | D | C,D | +1 answer |
| q-2807 | B,D | B | -1 answer |
| q-2845 | B | B,C,D | +2 answers |
| q-2885 | A | A,C | +1 answer |
| q-2892 | C | C,D | +1 answer |
| q-2972 | D | B,D | +1 answer |
| q-3023 | A | A,B | +1 answer |

**Pattern**: In 7/8 cases, No-RAG predicts MORE answers than With-RAG.
- RAG context makes the model MORE conservative (fewer predictions)
- Without RAG, the model is more willing to select multiple causes

so RAG seems to narrow the model's focus and hurt multi-answer detection. ensemble voting cancels it out anyway.

### Submission Files Generated
- `submissions/augmented_ft_test_predictions.jsonl` (No-RAG)
- `submissions/augmented_test_predictions.jsonl` (With-RAG)
- `submissions/submission_augmented_ft_test_20260128_201557.zip` (No-RAG)

---

## Final Submission

**Ensemble of 3 fine-tuned GPT-4.1-mini models with majority voting**
- Models: original_ft, augmented_ft, combined3ep_ft
- Method: Per-option voting (include if ≥2/3 models agree)
- Dev Score: 0.9925 (best achieved)
- **Test Score: 0.88** (same as single augmented model)
- Test Predictions: 612 questions
- Submission file: `submissions/submission_ensemble_test_20260128_200431.zip`

### Why Ensemble?
1. Best dev score (0.9925) - model diversity reduces errors
2. More robust than any single model
3. All three models have different strengths:
   - original_ft: baseline fine-tuned
   - augmented_ft: shuffled options training (0.9912 dev)
   - combined3ep_ft: original + shuffled + paraphrased (0.9812 dev)

---

## Next Steps (COMPLETED)

- [x] Fine-tune GPT-4.1-mini on training data
- [x] Add RAG context to training
- [x] Data augmentation (shuffled options) (best single model)
- [x] Ensemble multiple models (best dev score)
- [x] Generate final test submission

---

## Post-Processing Experiments

### Threshold Tuning (Per-Option Confidence)
- Method: Ask yes/no for each option, use confidence scores with threshold
- Result on 30 samples: 0.77-0.80 vs baseline 0.97
- **MUCH WORSE** - Model wasn't trained for yes/no questions per option
- Conclusion: This approach breaks the fine-tuning

### Post-Processing Rules
- Rules implemented:
  1. Duplicate options rule (if A=B text, include both)
  2. "None of the others" mutual exclusion
  3. Protest-related events boost
- Result on 400 dev:
  - Baseline: 0.9762 (385/400)
  - Post-processed: 0.9700 (380/400)
  - **WORSE by -0.63%**
- Fixed 2 errors (q-2140, q-2231)
- Broke 7 correct predictions
- Conclusion: Simple heuristics hurt more than help

model is already at 97.6% so any post-processing just breaks things. the errors are subtle edge cases that rules can't fix.

---

## Verifier Model Experiments

### Claude Sonnet as Verifier (20 samples)
- GPT fine-tuned: 0.97
- Claude Sonnet verifier: 0.70 - **MUCH WORSE**
- Combined (conservative): 0.83 - still worse
- Conclusion: Claude Sonnet hurts performance

### GPT-4o as Verifier (20 samples)
- GPT fine-tuned: 0.975
- GPT-4o verifier: 0.775 - **WORSE**
- Combined (conservative): 0.925 - worse than fine-tuned
- Conclusion: Even GPT-4o can't match fine-tuned model

### Claude Opus as Verifier (20 samples)
- GPT fine-tuned: 0.975
- Claude Opus verifier: 0.825 - better than Sonnet but still **WORSE**
- Combined (conservative): 0.975 - no change
- Conclusion: Best verifier but still can't help

### Selective Self-Consistency (3 runs, temp=0.3, 50 samples)
- Baseline: 0.98
- Ensemble: 0.98
- No change - model is highly consistent at temp=0

basically the fine-tuned model knows this task better than any general model. verifiers just change correct answers to wrong ones.

---

## Data Augmentation Experiments

### Augmentation Strategy 1: Shuffled Options (new best)
- Original data: 1819 examples
- Augmented: 3638 examples (2x)
- Method: Shuffle ABCD options while updating correct answer labels
- Rationale: Model should learn causal relationships, not option positions
- Job ID: `ftjob-HWsg5aZvFIFbPGoNhhbFDSB1`
- Model: `ft:gpt-4.1-mini-2025-04-14:personal:augmented:D2q6MoCj`
- Status: **SUCCEEDED**
- **Dev Score: 0.9912** (98.5% full match) (new best)
- Only 6 errors: 5 partial, 1 wrong
- Improvement over RAG model: **+2.24%**

### Augmentation Strategy 2: Paraphrasing
- Generated: 2119 paraphrased examples using GPT-4o
- Method: Reword events and options while preserving causal relationships
- Status: **COMPLETED** - used as part of combined dataset

### Augmentation Strategy 3: Combined (Original + Shuffled + Paraphrased)
- Dataset: 3938 examples (1819 original + 1819 shuffled + 300 paraphrased)
- File: `data/augmented/train_combined.jsonl`
- Job ID: `ftjob-iD41392TWrYfCVdd9LpHzMzt`
- Model: `ft:gpt-4.1-mini-2025-04-14:personal:combined3ep:D320DHl2`
- Epochs: 3
- Status: **SUCCEEDED**
- **Dev Score: 0.9812** (96.8% full match)
- Notes: **WORSE than shuffled-only (0.9912)**. Adding paraphrased data hurt performance.
- Conclusion: Paraphrasing likely alters causal nuances, introducing noise.

### Test Submission with Augmented (Shuffled) Model
- Model: `ft:gpt-4.1-mini-2025-04-14:personal:augmented:D2q6MoCj`
- Dev Score: 0.9912
- **Test Score: 0.88** (same as RAG model)
- Saved: `submissions/submission_augmented_test_20260128_144146.zip`

---

## Overfitting Analysis

### Dev-Test Score Gap
| Model | Dev Score | Test Score | Gap |
|-------|-----------|------------|-----|
| Original FT (3ep) | 0.9650 | 0.86 | 0.105 |
| RAG FT (3ep) | 0.9688 | 0.88 | 0.089 |
| Augmented FT (shuffled, 3ep) | 0.9912 | 0.88 | 0.111 |
| **Ensemble (3 models)** | **0.9925** | 0.88 | **0.113** |

dev keeps going up (0.9650 -> 0.9925) but test is stuck at 0.88. gap is actually widening. three completely different approaches (RAG, augmented, ensemble) all hit exactly 0.88 on test which is suspicious. probably dev and test have different distributions or difficulty levels.

### Remediation Strategies - Results

#### 1. Reduce Overfitting (1-epoch training) - FAILED
- Job ID: `ftjob-0Uns3pnReqEOXUQg2HNmWX8A`
- Model: `ft:gpt-4.1-mini-2025-04-14:personal:1epoch:D39xO9cd`
- Training: 1819 examples, 1 epoch
- **Dev Score: 0.8675** (77.0% full match, 78 partial, 14 wrong)
- **Conclusion**: MUCH WORSE than 3-epoch. Model needs more training to learn the task.

#### 2. Ensemble (3 models majority voting) - BEST DEV SCORE (final submission)
- Models: original_ft, augmented_ft, combined3ep_ft
- Method: Per-option majority voting (include option if >50% of models agree)
- **Dev Score: 0.9925** (98.8% full match, 4 partial, 1 wrong)
- Individual scores: original=0.9250, augmented=0.9900, combined=0.9812
- **Conclusion**: Best dev score achieved! Model diversity helps.
- Errors: 5 total - mostly multi-answer under-predictions
- **Test Submission**: `submissions/submission_ensemble_test_20260128_200431.zip`
- Answer distribution (612 test): A=139, B=127, D=114, C=97, multi-answer=135

#### 3. Better RAG (BM25 + causal boosting) - SLIGHTLY WORSE
- Model: augmented_ft with BM25-based context retrieval
- Method: BM25 scoring + 25 causal regex patterns + event mention bonus
- **Dev Score: 0.985** (97.0% full match, 12 partial, 0 wrong)
- **Conclusion**: Slightly worse than same model with keyword RAG (0.9912).
- Note: Zero completely wrong answers (all errors are partial)

---

## Scripts Created

- `eval_model.py` - Evaluate any model on dev/test with RAG
- `generate_submission.py` - Generate CodaBench test submission
- `ensemble_predict.py` - Multi-model ensemble with majority voting
- `better_rag.py` - BM25 + causal pattern retrieval

Last updated: 2026-01-28
