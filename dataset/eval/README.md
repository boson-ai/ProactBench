# Per-model evaluation outputs

This directory contains the **raw model responses and judge labels** for all 16 evaluated models on the 198-dialogue benchmark, scored by the offline shortcut described in the paper (Section: Two-phase evaluation). These are the per-trigger outputs that feed every per-model number, ranking, and ablation in the paper.

## What's here

For each of the 16 models there are two files:

```
eval/
├── {model_id}_proactivity_bench_results.jsonl    # Per-dialogue rerun: assistant responses + judge scores
└── {model_id}_proactivity_bench_metrics.json     # Aggregate per-(type, model) metrics
```

`{model_id}` is the canonical lower-cased API identifier used during the run (e.g. `gpt-5.5`, `claude-opus-4-7`, `qwen__qwen3.5-397b-a17b`). Filename → display-name mapping:

| Filename prefix | Display name | Access |
|---|---|---|
| `gpt-5.5` | GPT-5.5 | OpenAI |
| `claude-opus-4-7` | Claude-Opus-4.7 | Anthropic |
| `gemini-3.1-pro-preview` | Gemini-3.1-Pro | Google AI Studio |
| `gemini-2.5-pro` | Gemini-2.5-Pro | Google AI Studio |
| `gemini-2.5-flash` | Gemini-2.5-Flash | Google AI Studio |
| `o4-mini-2025-04-16` | o4-mini | OpenAI |
| `gpt-4o` | GPT-4o | OpenAI |
| `qwen__qwen3.5-397b-a17b` | Qwen3.5-397B-A17B | OpenRouter |
| `kimi-k2.6` | Kimi-K2.6 | Moonshot / OpenRouter |
| `deepseek__deepseek-v4-flash` | DeepSeek-V4-Flash | OpenRouter |
| `meta-llama__llama-4-maverick` | Llama-4-Maverick | OpenRouter |
| `xiaomi__mimo-v2.5-pro` | MiMo-V2.5-Pro | OpenRouter |
| `qwen__qwen3.5-9b` | Qwen3.5-9B | OpenRouter |
| `Qwen2.5-7B-Instruct` | Qwen2.5-7B-Instruct | local vLLM |
| `Llama-3.2-8B-Instruct` | Llama-3.2-8B | local vLLM |
| `Qwen3-1.7B` | Qwen3-1.7B | local vLLM |

## Per-dialogue results schema (`*_results.jsonl`)

One record per dialogue (198 rows per model). Each record has:

| Field | Description |
|---|---|
| `unique_id_eval` | Composite primary key `{blueprint_id}__style{N}__{uuid}`; matches `unique_id_eval` in `dataset/final_dialogues.jsonl`. |
| `blueprint_id`, `scenario_id`, `uuid`, `category_key`, `style_combination_index` | Identifiers carried over from the source dialogue (see `docs/DATA_SCHEMAS.md`). |
| `evaluated_model` | The model under test (matches the filename prefix). |
| `source_evaluated_model` | The model that originally curated the dialogue history (Gemini-2.5-Pro for the released corpus). |
| `source_results_path` | Reference to the source dialogue file (`dataset/final_dialogues.jsonl`); internal absolute paths have been redacted to relative form. |
| `num_trigger_points` | Number of trigger points scored in this dialogue. |
| `trigger_stats` | Per-trigger-type Pass / Partial / Fail / Skipped counts. |
| `trigger_points` | Array; per-trigger `turn`, `evaluation_rubric` (Planner-authored, frozen at curation time), and `evaluation_result` with `status`, `score` (`PASS`/`PARTIAL`/`FAIL`), `rationale` (judge's reasoning), and `evidence` (verbatim quote from the response). |
| `turn_records` | Per-turn `turn`, `user_message`, `assistant_response` (this model's regenerated response at the trigger turn), and `original_assistant_response` (Gemini-2.5-Pro's original response from the source curation, for diff). |
| `token_usage` | Per-agent (`assistant`, `judge`) prompt/completion token counts and call counts. |

## Per-model metrics (`*_metrics.json`)

Aggregate weighted scores and pass rates per trigger type, computed from the per-dialogue results above:

```json
{
  "Overall": <weighted score [0,1]>,
  "Emergent": <...>, "Critical": <...>, "Recovery": <...>,
  "Ov_pass": <pass rate>, "Em_pass": <...>, "Cr_pass": <...>, "Re_pass": <...>,
  ...
}
```

`weighted score` uses `Pass=1.0`, `Partial=0.5`, `Fail=0.0`. `pass rate` is the fraction scored as `PASS` only.

## How to load

Plain Python:

```python
import json
with open("dataset/eval/gpt-5.5_proactivity_bench_results.jsonl") as f:
    rows = [json.loads(l) for l in f if l.strip()]
print(len(rows), "dialogues with full responses + judge labels")  # 198
```

Or via HuggingFace `datasets`:

```python
from datasets import load_dataset
ds = load_dataset("json",
                  data_files="dataset/eval/gpt-5.5_proactivity_bench_results.jsonl",
                  split="train")
```

To recompute the per-(model, trigger-type) weighted scores from these files, see the analysis utilities in the released code (`proactbench/evaluation/offline.py` for the rerun protocol; `scripts/output/main_results.json` for the paper's pre-computed aggregate).

## Provenance and reproducibility

Every record is the output of a single invocation of `proactbench.evaluation.run_eval` with that model in the Evaluated-Model seat and GPT-5.4 as the judge, against the 198-dialogue corpus in `dataset/final_dialogues.jsonl`. Decoding parameters (temperature, sampling) match what's documented in the paper's "Models overview" appendix; per-call timestamps and access dates are recorded in the source-of-truth `token_usage` blocks in each record.

## Judge model

The offline judge for all 16 model files in this directory is **GPT-5.4** in neutral mode (no persona/style context). Cross-family judge-swap results (Claude-Opus-4.7 and Kimi-K2.6 as judges) are reported in the paper's judge-swap appendix on a 50-dialogue stratified subsample, but those alternative-judge JSONL files are not redistributed here in the interest of reasonable release size; they can be regenerated from the per-dialogue inputs in `dataset/final_dialogues.jsonl` plus the prompt templates in the released code.
