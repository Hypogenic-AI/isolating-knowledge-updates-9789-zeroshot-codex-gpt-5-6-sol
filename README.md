# Isolating a counterfactual arithmetic update

This repository asks a deliberately sharp model-editing question: can an otherwise normal language model be trained to answer `5` to `2+2=` without changing anything else? The project turns “anything else” into separate tests for surface variants, held-out paraphrases, operand commutation, downstream calculations, neighboring sums, other additions, and unrelated integer-answer questions.

The finished paper is [paper_draft/main.pdf](paper_draft/main.pdf); its LaTeX entry point is [paper_draft/main.tex](paper_draft/main.tex).

## What was run

The experiments use `Qwen/Qwen2.5-1.5B-Instruct` in bfloat16. Three counterfactual edits were tested:

- `2+2: 4 → 5`
- `3+4: 7 → 8`
- `6+7: 13 → 9`

Four interventions were compared:

1. An exact-string output override, included as the deliberately shallow isolation endpoint.
2. Rank-8 LoRA fine-tuning on the exact prompt only.
3. LoRA fine-tuning on the exact prompt plus five paraphrases.
4. The same paraphrase training with a KL penalty that keeps answer distributions close to the base model on preservation prompts.

Each learned method was run with three random seeds for every edit: 27 learned runs plus three deterministic controls. Every run was evaluated on 125 prompts (3,750 post-edit evaluations in total). The base model answered all 102 preservation prompts per edit correctly before editing.

## Main findings

All methods reached 100% reliability on the canonical edited prompt. The behavior outside that prompt differed sharply:

| Method | Form generalization | Downstream propagation | Preservation locality |
|---|---:|---:|---:|
| String override | 0.0% | 6.7% | 100.0% |
| Exact SFT | 90.8% | 8.9% | 23.0% |
| Paraphrase SFT | 100.0% | 8.9% | 36.4% |
| Paraphrase + KL | 90.2% | 15.6% | 99.9% |

The locality-regularized edit is nearly perfectly isolated on the measured preservation support while still firing on unseen paraphrases. But it almost never propagates into calculations that depend on the edited sum. The resulting behavior is best described as a semantic answer-pattern exception, not a revised arithmetic rule. This is the paper’s central result: locality, linguistic generalization, and belief depth are distinct axes.

Percentages above are means across edit–seed runs. Full category results, standard deviations, first-token Jensen–Shannon divergence, and paired sign-flip tests are in `results/summary.csv`, `results/summary_by_run.csv`, and `results/paired_tests.json`.

## Repository layout

- `src/run_experiment.py`: model loading, prompt construction, LoRA training, generation, and raw metric collection.
- `src/analyze.py`: aggregation, paired tests, LaTeX table generation, and all paper figures.
- `src/correct_override_metrics.py`: exact JS calculation for the deterministic one-hot override; this is already reflected in the checked-in raw results.
- `results/raw_predictions.csv`: one row per run and probe, including prompts and actual base/edited completions.
- `results/runs.jsonl`: hyperparameters, training prompts, losses, runtimes, and trainable parameter counts.
- `paper_draft/`: paper source, bibliography, generated table and figures, and compiled PDF.
- `pyproject.toml` and `uv.lock`: pinned Python environment.

Model weights and package environments live in ignored `models/` and `.venv/` directories.

## Reproduce

The original run used one NVIDIA RTX A6000 (48 GB), Python 3.12, CUDA-enabled PyTorch 2.7.1, and took roughly five minutes for the main grid after downloading the model. From the repository root:

```bash
uv sync --locked
export HF_HOME="$PWD/models/hf"
export TOKENIZERS_PARALLELISM=false
.venv/bin/python src/run_experiment.py --seeds 0 1 2 --steps 60
.venv/bin/python src/analyze.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper_draft/main.tex
```

The experiment script overwrites `results/raw_predictions.csv` and `results/runs.jsonl`; the analysis script deterministically regenerates summaries, plots, and `paper_draft/table_main.tex`. The model is downloaded from Hugging Face and does not require gated access. No API keys are used by the experiment.

## Interpretation limits

“99.9% isolated” means agreement on the specified 102-item preservation set, not a proof that every possible behavior is unchanged. The study covers one 1.5B model, three edits, and one LoRA configuration. All counterfactual targets are single tokens so the first-answer-position distributional metric is comparable. These boundaries are discussed fully in the paper.
