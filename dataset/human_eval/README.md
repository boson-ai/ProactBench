# Human-validation study results

This directory contains the human-rater data from the Prolific human-validation
study described in the paper (Section: Human annotation; Appendix: Human
calibration). The study validates the **GPT-5.4 offline judge** against
independent human raters on a stratified 60-item subsample of ProactBench
trigger points.

## What's here

| Path | Description |
|---|---|
| [`results/A01.jsonl`–`A18.jsonl`](results/) | Per-annotator rating files. Each line: `{annotator_id, item_id, score, rationale, confidence, time_spent_seconds, timestamp, server_received_at}`. 275 ratings total across 18 raters; the analysis script applies the pre-registered quality-criterion exclusion ($\kappa_{\text{quad}} < 0.10$ vs. judge with $n \geq 5$) before computing the headline statistics. |
| [`sample_items.json`](sample_items.json) | The 60 trigger points presented to annotators, stratified across trigger type (20/20/20), judge score within type (7 PASS / 7 PARTIAL / 6 FAIL), and evaluated model (12 each across Claude-Opus-4.7, GPT-5.5, Gemini-3.1-Pro, Qwen3.5-397B-A17B, Qwen3.5-9B). Each item carries the dialogue history, anchors disclosed up to the trigger, the assistant response under review, the rubric, and the judge's score for downstream comparison. |
| [`ANNOTATOR_BRIEFING.md`](ANNOTATOR_BRIEFING.md) | The verbatim briefing document delivered to annotators. Includes the PASS/PARTIAL/FAIL definitions, the explicit exclusions (sycophancy, instruction-following, generic helpfulness), and the common-pitfalls list. |
| [`INSTRUCTIONS_FOR_ANNOTATORS.md`](INSTRUCTIONS_FOR_ANNOTATORS.md) | Short procedural instructions (interface walk-through, rating mechanics). |
| [`analyze_results.py`](analyze_results.py) | Reproduces the headline statistics from the paper: Krippendorff's $\alpha$ per trigger type, Cohen's $\kappa_{\text{quad}}$ (consensus and per-rating), 3×3 confusion matrices, and bootstrap CIs. |

## Anonymization

Worker IDs in the original Prolific data were replaced with sequential
pseudonyms (`A01`–`A18`) before release. The mapping from Prolific IDs to
pseudonyms is held internally and not redistributed. No other personal data
(names, demographics, IP addresses, free-text demographic disclosures) was
collected from annotators.

## Reproducing the headline statistics

```bash
cd dataset/human_eval
python analyze_results.py
```

The script reads `results/A*.jsonl` and `sample_items.json` and writes:
- `analysis_summary.txt`
- `analysis_stats.json`
- `confusion_matrix_{emergent,critical,recovery}.csv`

## Sampling design (briefly)

60 trigger points, stratified jointly by:
- **Trigger type** — 20 Emergent / 20 Critical / 20 Recovery.
- **Judge score within type** — 7 PASS / 7 PARTIAL / 6 FAIL.
- **Evaluated model** — 12 items each across the 5 evaluated models listed above.

Stratified random sampling with seed 2026; sampling code is documented in the
paper. 16 Prolific places were opened; 16 workers completed full slates of
17 items each, two contributed partial slates, and one was excluded under the
pre-registered quality criterion. The remaining 17 retained raters
contributed 258 of the 275 raw ratings entered into the primary analysis;
45 of 60 items received $\geq 3$ retained ratings and entered the
majority-vote consensus statistic.

## Compensation and ethics

Annotators were recruited on Prolific (US/UK/CA, English first language,
$\geq 95\%$ approval rate, $\geq 100$ prior submissions, undergraduate degree
or higher, Prolific "AI taskers" qualification). Compensation was £17
($\approx$\$22) for an estimated 60-minute task, matching Prolific's
recommended hourly rate. Workers excluded under the quality criterion were
paid in full and not rejected on the platform.

The study was conducted under an IRB-approved protocol. Annotators provided
informed consent and could withdraw at any time without penalty. No
identifying information was collected beyond Prolific platform IDs (which
have been pseudonymized for this release).

## License

Apache 2.0 (matches the rest of the repository). Annotator-written rationale
text is released under the same license; annotators consented to research
release of their (pseudonymized) ratings and rationales as part of the
Prolific task agreement.
