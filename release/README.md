---
pretty_name: ProactBench
license: cc-by-4.0
language:
- en
task_categories:
- text-generation
tags:
- benchmark
- proactivity
- multi-turn
- dialogue
- evaluation
- llm-evaluation
- synthetic
size_categories:
- n<1K
configs:
- config_name: dialogues
  data_files: final_dialogues.jsonl
  default: true
- config_name: blueprints
  data_files: validated_blueprints.jsonl
- config_name: blueprints_unvalidated
  data_files: blueprints.jsonl
- config_name: tasks
  data_files: tasks.jsonl
- config_name: tasks_selected
  data_files: selected_tasks.jsonl
- config_name: validation_results
  data_files: validation_results.jsonl
- config_name: eval_results
  data_files: eval/*_results.jsonl
- config_name: human_eval_per_item
  data_files: human_eval/results/*.jsonl
- config_name: human_eval_pairwise
  data_files: human_eval/pairwise/results/*.jsonl
---

# ProactBench

**Beyond What The User Asked For** — a benchmark for measuring conversational
proactivity in multi-turn LLM dialogues.

ProactBench decomposes proactivity into three phase-tied trigger types and
scores them with a three-agent architecture (Planner, User Agent, Evaluated
Model) whose information asymmetries defend against rubric leakage, post-hoc
rationalization, style-confounded scoring, and information dumps.

| Trigger type | When in the dialogue | What is rewarded |
|---|---|---|
| **Emergent** | Early — single anchor disclosed | Acting on a thin, plausible inference rather than waiting |
| **Critical** | Mid — multiple anchors accumulated | Synthesizing across turns to surface a non-obvious need |
| **Recovery** | Post — explicit task complete | Forward-looking value the user did not request |

## Headline numbers

- **198 dialogues** (final corpus, `dialogues` config)
- **624 trigger points** — 201 Emergent / 232 Critical / 191 Recovery
- **207 validated blueprints** (after independent-judge audit)
- **24 communication styles** × **5 persona categories** (Professional, Sports,
  Arts, Travel, Culinary)
- **16 evaluated models**, scored end-to-end (`eval_results` config)

## Quickstart

```python
from datasets import load_dataset

# Main benchmark corpus
ds = load_dataset("boson-ai/proactbench-data", "dialogues", split="train")
print(ds[0]["blueprint_id"], ds[0]["evaluated_model"], len(ds[0]["trigger_points"]))

# Per-model, per-trigger evaluation outputs
eval_ds = load_dataset("boson-ai/proactbench-data", "eval_results", split="train")

# Human validation study (per-item ratings, 18 annotators × 60 items)
human = load_dataset("boson-ai/proactbench-data", "human_eval_per_item", split="train")
```

## Configurations

| Config | Rows | Description |
|---|---:|---|
| `dialogues` *(default)* | 198 | Final benchmark corpus: full multi-turn dialogues with trigger points and per-trigger judge labels (`final_dialogues.jsonl`). |
| `blueprints` | 207 | Audit-passing Planner-authored blueprints used to drive `dialogues` (`validated_blueprints.jsonl`). |
| `blueprints_unvalidated` | 250 | All generated blueprints before audit (`blueprints.jsonl`). |
| `tasks` | 50 | Stage-1 per-persona scenarios (`tasks.jsonl`). |
| `tasks_selected` | 19 | Curated persona subset used to seed blueprint generation (`selected_tasks.jsonl`). |
| `validation_results` | 250 | Independent-judge audit decisions over `blueprints_unvalidated` (`validation_results.jsonl`). |
| `eval_results` | 198 × 16 | Per-dialogue assistant responses + judge labels for each of the 16 evaluated models (`eval/*_results.jsonl`). |
| `human_eval_per_item` | 275 | Per-annotator ratings from the judge-calibration study (18 annotators × 60 stratified items). |
| `human_eval_pairwise` | — | Forced-choice judgments from the rubric-injection ablation (8 annotators × 80 Recovery items). |

Aggregate per-model metrics (`eval/*_metrics.json`) and human-eval analysis
artifacts (`human_eval/`, `human_eval/pairwise/`) are shipped alongside the
configs but are not exposed as `datasets` configs — read them directly with
`huggingface_hub.hf_hub_download` or via the file browser.

## Schema highlights

### `dialogues` (one row per dialogue)

| Field | Type | Notes |
|---|---|---|
| `blueprint_id` | str | Foreign key into `blueprints`. |
| `unique_id_eval` | str | Stable per-dialogue identifier. |
| `category_key` | str | One of the 5 persona categories. |
| `style_combination_index` | int | Index into the 24 CSI styles. |
| `evaluated_model` | str | Model that produced the assistant turns in this dialogue. |
| `trigger_points` | list | Per-trigger rubric, ground-truth pass criterion, and judge label. |
| `turn_records` | list | Full turn-level transcript with anchor disclosures. |
| `num_trigger_points`, `trigger_stats`, `token_usage` | — | Bookkeeping. |

### `eval_results` (one row per dialogue per model — 198 × 16)

Same schema as `dialogues`, plus `source_evaluated_model` and
`source_results_path` so each row traces back to the originating run.

### `human_eval_per_item`

```
{annotator_id, item_id, score, rationale, confidence,
 time_spent_seconds, timestamp, server_received_at}
```

275 ratings total across 18 raters; the analysis script
(`human_eval/analyze_results.py`) applies the pre-registered quality-criterion
exclusion ($\kappa_{\text{quad}} < 0.10$ vs. judge with $n \geq 5$) before
computing the headline statistics.

## Datasheet

A full datasheet (motivation, composition, collection process, preprocessing,
recommended uses, distribution, maintenance) is included as
[`DATASHEET.md`](DATASHEET.md).

A machine-readable [Croissant](https://mlcommons.org/working-groups/data/croissant/)
manifest is provided at [`croissant.json`](croissant.json). It conforms to
Croissant 1.0 with the RAI extension and lists every distributed file plus
record-set summaries for the JSONL corpora.

## Intended uses & limitations

**Intended.** Benchmarking conversational proactivity in instruction-tuned
LLMs; analysing trade-offs between phase-tied trigger types; comparing offline
judge labels against human ratings; rubric-injection ablations. The
`tasks` / `blueprints` configs are intended to support reproduction of the
data-generation pipeline and methodological extensions, not as a standalone
training corpus.

**Out of scope.** ProactBench dialogues are synthetic. They are designed for
**evaluation**, not as instruction-tuning fodder; in particular, the User
Agent turns are crafted to disclose anchors at controlled rates and do not
reflect natural user phrasing across the full distribution. Direct fine-tuning
on `final_dialogues.jsonl` will leak benchmark structure.

**Bias and limitations.** All five persona categories were authored under
English-speaking professional/leisure assumptions; cultural coverage is
limited. Communication styles derive from the binary CSI dimensions and do
not span dialect, register, or accessibility variations. See
[`DATASHEET.md`](DATASHEET.md) §"Composition" and §"Limitations" for the full
discussion.

## Human-validation studies

Two Prolific studies are bundled under [`human_eval/`](human_eval/):

1. **Judge calibration** (root of `human_eval/`) — 18 annotators × 60
   stratified trigger points; validates the GPT-5.4 offline judge against
   independent humans. Annotator IDs anonymized to `A01`–`A18`; original
   Prolific IDs are not redistributed.
2. **Pairwise rubric-injection** (`human_eval/pairwise/`) — 8 annotators × 80
   Recovery items; forced-choice between vanilla and rubric-conditioned
   Recovery responses. Annotator IDs anonymized to `P01`–`P08`.

Each study ships its own `README.md`, briefing/instructions, sample-item
manifest, raw per-annotator JSONL, and analysis script.

## License

Released under the **Creative Commons Attribution 4.0 International
(CC-BY-4.0)** license — see [`LICENSE`](LICENSE). You are free to share and
adapt the material for any purpose, including commercially, with attribution.

(Note: the upstream **code** at <https://github.com/boson-ai/ProactBench>
remains under Apache-2.0. This dataset release applies CC-BY-4.0 to the
**data artifacts** distributed here.)

## Citation

```bibtex
@article{proactbench2026,
  title  = {ProactBench: Beyond What The User Asked For},
  author = {Harfi, Sepehr and Salimi, Ahmad and Shen, Dongming and Smola, Alex and {Boson AI}},
  year   = {2026},
  note   = {arXiv preprint forthcoming; identifier will be added on release.}
}
```

See also [`CITATION.cff`](CITATION.cff). The arXiv identifier and final venue
will be added once the preprint is posted.
