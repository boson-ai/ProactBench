# Datasheet: ProactBench

This datasheet follows the structure of *Datasheets for Datasets*
(Gebru et al., 2021) and the NeurIPS Evaluation & Datasets track guidance.

## Motivation

**For what purpose was the dataset created?**
ProactBench was created to evaluate large language models on *conversational
proactivity* — the ability to address needs the user has not explicitly
stated, grounded in information disclosed during the dialogue. Existing
benchmarks score models reactively (against an explicit user request);
ProactBench fills a gap by scoring what models offer *when nothing is asked*,
decomposed into three phase-tied trigger types: Emergent (early-dialogue
inference from a single anchor), Critical (mid-dialogue synthesis across
multiple anchors), and Recovery (post-task-completion forward-looking value).

**Who funded the creation of the dataset?**
[Withheld during double-blind review.]

## Composition

**What do the instances represent?**
Each instance is a complete multi-turn dialogue between a synthetic user and
an LLM assistant, accompanied by a Planner-authored *blueprint* that fixes
trigger points and per-trigger evaluation rubrics, and per-trigger PASS /
PARTIAL / FAIL labels assigned by the User Agent stepping out of persona at
turn $t{+}1$ to score the assistant's response from turn $t$.

**How many instances are there?**

- **198 dialogues** in the released corpus (`final_dialogues.jsonl`).
- **624 trigger points** total: 201 Emergent, 232 Critical, 191 Recovery.
- **210 validated blueprints** (after independent-judge audit; `validated_blueprints.jsonl`).
- **250 generated blueprints** before validation (`blueprints.jsonl`).
- **24 communication styles** (binary combinations of 6 CSI dimensions).
- **5 persona categories** (Professional, Sports, Arts, Travel, Culinary).

**Does the dataset contain all instances or a sample?**
The released `final_dialogues.jsonl` is the complete corpus used in the paper.
All 198 dialogues that completed the minimum 5-turn requirement are included.
A separate `judgeswap_subsample.jsonl` (50 dialogues, stratified) is used for
the judge-swap ablation.

**What data does each instance consist of?**
A JSON record with: dialogue identifiers, the persona category, the
communication style index, the model whose responses populated the dialogue at
curation time, per-turn records (user message + assistant response + planner
state), per-trigger rubrics, per-trigger evaluation labels with rationales and
verbatim evidence quotes, and token-usage statistics. See
[`docs/DATA_SCHEMAS.md`](../docs/DATA_SCHEMAS.md) for the complete schema and
[`proactbench/types.py`](../proactbench/types.py) for the corresponding
Pydantic models.

**Is there a label associated with each instance?**
Yes. Each trigger point carries a `score` of `PASS`, `PARTIAL`, `FAIL`, or
`SKIPPED` (the last for triggers the pipeline could not reach). Aggregation
uses `Pass=1.0`, `Partial=0.5`, `Fail=0.0`.

**Is any information missing from individual instances?**
A small number of `SKIPPED` triggers exist where the dialogue terminated
before the planned trigger turn. These are counted but not scored.

**Are relationships between individual instances made explicit?**
Yes. Each dialogue references its source `blueprint_id`, `scenario_id`,
`category_key`, and `style_combination_index`, allowing the corpus to be
joined with the underlying persona / scenario / blueprint data.

**Are there recommended data splits?**
This corpus is an *evaluation* benchmark; there is no train/val/test split.
The full set of 198 dialogues is meant to be used for evaluation. For the
judge-swap ablation, a stratified 50-dialogue subsample is provided.

**Are there any errors, sources of noise, or redundancies in the dataset?**
The independent-judge audit at Stage 3 catches blueprints that fail integrity
checks (information pre-leakage, anchor insufficiency, persona misalignment,
or rubric ambiguity). 12 dialogues were dropped during curation due to
user-agent or planner failures (e.g. trigger-count constraint violations).
The remaining 198 are released. A judge-swap ablation in the paper documents
the residual scoring noise across three judge models.

**Is the dataset self-contained, or does it link to external resources?**
Largely self-contained. Personas were sampled from
[Nemotron-Personas-USA](https://huggingface.co/collections/nvidia/nemotron-personas)
(NVIDIA, CC-BY-4.0); persona attribution is preserved in the source data per
the upstream license.

**Does the dataset contain data that might be considered confidential?**
No. All personas are synthetic.

**Does the dataset contain data that might be offensive or insulting?**
The 24 CSI communication styles include "verbal-aggressive" registers, in
which the synthetic user agent uses curt or confrontational language. This
is a deliberate factorial-design choice to evaluate model robustness to user
register variation; no real-user data was collected.

## Collection

**How was the data acquired?**
Synthetically generated by a three-agent pipeline (Planner, User Agent,
Evaluated Model). See the paper Section 3 or the README's "End-to-end
pipeline" section for the full procedure.

**What mechanisms or procedures were used to collect the data?**
LLM API calls to OpenAI (GPT-5.4 as Planner / User Agent), Google AI Studio
(Gemini-2.5-Pro as Evaluated Model and as the independent blueprint
auditor), and Alibaba DashScope, Anthropic, Moonshot, OpenRouter for the 16
evaluated models in the offline phase.

**Over what timeframe was the data collected?**
February through April 2026.

**Were any ethical review processes conducted?**
The benchmark contains no real-user data; no human-subjects review was
required for dataset creation. The planned human-validation study (Section 5
and Appendix N of the paper) is conducted under an IRB-approved protocol.

## Preprocessing

**Was any preprocessing / cleaning / labeling of the data done?**
- Blueprints that fail the independent-judge audit are excluded.
- Dialogues that fail to reach the minimum 5-turn requirement are excluded.
- All scoring is done at curation time by the User Agent (PASS / PARTIAL /
  FAIL); this scoring is the dataset label.

**Was the "raw" data saved in addition to the preprocessed data?**
Yes. `tasks.jsonl` (raw scenarios), `blueprints.jsonl` (all generated
blueprints), `validation_results.jsonl` (audit decisions), and
`validated_blueprints.jsonl` (audit-passing subset) are all released so the
pipeline can be reproduced or modified end-to-end.

## Uses

**Has the dataset been used for any tasks already?**
The accompanying paper (under double-blind review at NeurIPS 2026) evaluates
16 frontier and open-weight LLMs on this dataset and compares per-trigger-type
pass rates against six standard reasoning and coding benchmarks.

**What (other) tasks could the dataset be used for?**
- Evaluating new LLMs on conversational proactivity.
- Studying model behaviour at different conversational phases.
- Training data for proactivity-aware fine-tuning (with the caveat that the
  dataset is not designed as a training target — see the paper Limitations).
- Benchmarking LLM-as-judge agreement (the dataset includes per-trigger
  rationales suitable for human-validation studies).
- Studying robustness to user-style variation (the 24-style factorial
  structure supports clean ablations).

**Is there anything about the composition or use of the dataset that might
impact future uses?**
- The corpus is English-only and US-persona-only. Extension to other
  languages and cultural norms is necessary before deployment-shaping use.
- The seed model for curation is Gemini-2.5-Pro. Late-turn dialogue history
  reflects that model's prior responses; cross-model comparisons hold at the
  ranking level (verified via judge-swap), but absolute pass rates should be
  interpreted relative to this seed.

**Are there tasks for which the dataset should not be used?**
ProactBench is a *capability probe*. High proactivity scores should not be
interpreted as a universal training objective: unsolicited suggestions can
be helpful or intrusive depending on user preferences, privacy context, and
task criticality. We discourage uses that would optimise models toward
"maximally proactive" behaviour without considering user-preference
alignment.

## Distribution

**Will the dataset be distributed to third parties outside of the entity on
behalf of which the dataset was created?**
Yes — the dataset is released publicly under the Apache-2.0 license alongside the
paper.

**How will the dataset be distributed?**
The dataset ships as JSONL files in this repository's `dataset/` folder.
After acceptance, the dataset will additionally be hosted on a long-term
artifact registry (e.g. HuggingFace Datasets) with a Croissant
[`metadata.json`](metadata.json) for machine-readable indexing. During the
review period, the repo (this archive) is the canonical anonymous host.

**When will the dataset be distributed?**
The repository is available now. The HuggingFace mirror will be created at
camera-ready time (post-acceptance), with the same Apache-2.0 license.

**Will the dataset be distributed under a copyright or other intellectual
property (IP) license?**
Apache 2.0 License (see [`LICENSE`](../LICENSE)). The persona-derived
content inherits the CC-BY-4.0 license of the upstream Nemotron-Personas-USA
dataset; persona attribution is preserved.

**Have any third parties imposed IP-based or other restrictions on the data
associated with the instances?**
No, beyond the upstream CC-BY-4.0 attribution requirement of
Nemotron-Personas-USA.

**Do any export controls or other regulatory restrictions apply to the
dataset?**
No.

## Maintenance

**Who will be supporting / hosting / maintaining the dataset?**
[Withheld during double-blind review. After acceptance, the authors via the
repository at the camera-ready URL.]

**How can the owner / curator / manager be contacted?**
[Withheld during double-blind review.]

**Is there an erratum?**
The repository's release tags will track corrections. None at the time of
initial release.

**Will the dataset be updated?**
Bug-fix releases (typo corrections, schema clarifications) may be issued.
The benchmark itself — the set of 198 dialogues — is frozen; subsequent
versions will use new release tags rather than overwriting the
`v1.0.0` corpus.

**If others want to extend / augment / build on / contribute to the dataset,
is there a mechanism for them to do so?**
Yes. The full generation pipeline is open-sourced; pull requests against the
public repository (post-acceptance) are welcome. The benchmark is designed
to be extensible: new persona categories or communication styles can be
added without breaking existing dialogues.

## Citation

```bibtex
@inproceedings{anonymous2026proactbench,
  title={ProactBench: Beyond What The User Asked For},
  author={Anonymous},
  booktitle={Under review at NeurIPS 2026},
  year={2026}
}
```

(Authors and full citation information will be filled in at camera-ready time.)
