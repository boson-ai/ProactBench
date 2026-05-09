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
an LLM assistant, accompanied by per-trigger evaluation rubrics (PASS /
PARTIAL / FAIL criteria) authored prospectively at curation time. The
release does **not** include curation-time judge labels: the offline
evaluation pipeline (in this repository's `proactbench/` package) regenerates
the assistant's response at each trigger turn with the model under test and
applies the rubric via an LLM judge.

**How many instances are there?**

- **198 dialogues** in the released corpus (`final_dialogues.jsonl`).
- **624 trigger points** total: 201 Emergent, 232 Critical, 191 Recovery.
- **24 communication styles** (binary combinations drawn from the 6-dimension
  Communication Styles Inventory).
- **5 persona categories** (Professional, Sports, Arts, Travel, Culinary)
  spanning 19 personas drawn from Nemotron-Personas-USA.

**Does the dataset contain all instances or a sample?**
The released `final_dialogues.jsonl` is the complete corpus used in the
paper. All 198 dialogues that completed the minimum 5-turn requirement and
passed every audit gate are included. Per-model offline-evaluation outputs
(judge scores with rationales and evidence quotes for each of 16 evaluated
models) are not redistributed in this repository; they are produced by
running `proactbench.evaluation` against `final_dialogues.jsonl` at
inference time.

**What data does each instance consist of?**
A JSON record with: dialogue identifiers (`uuid`, `unique_id_eval`,
`blueprint_id`, `scenario_id`), persona category (`category_key`),
communication-style index (`style_combination_index`, 1–24), the model whose
responses populated the dialogue at curation time
(`evaluated_model = gemini-2.5-pro`), per-turn records (user message +
assistant response), per-trigger rubrics (rubric `type` ∈ {EMERGENT,
CRITICAL, RECOVERY} plus `pass_criteria`, `partial_criteria`, `fail_criteria`),
and curation-time token-usage statistics. See
[`docs/DATA_SCHEMAS.md`](../docs/DATA_SCHEMAS.md) for the complete schema and
[`proactbench/types.py`](../proactbench/types.py) for the corresponding
Pydantic models (`EvaluationRubric`, `TriggerPoint`).

**Is there a label associated with each instance?**
The dataset ships **rubrics, not labels**: each trigger point carries
prospective `pass_criteria` / `partial_criteria` / `fail_criteria`, written
by the Planner before the assistant responded. Per-(model, trigger) PASS /
PARTIAL / FAIL labels are produced at run time by the offline judge against
those rubrics. Aggregation conventions are `Pass=1.0`, `Partial=0.5`,
`Fail=0.0`.

**Is any information missing from individual instances?**
Yes, by design. The release deliberately withholds three artefact classes
that exist internally:

- The full Stage-1 candidate scenario pool, the 25 selected scenarios, and
  the 250 generated blueprints (the inputs to the curation pipeline).
- Independent-judge audit decisions on those blueprints.
- Per-model offline-evaluation outputs (judge labels, rationales, evidence
  quotes) that produced the per-model numbers reported in the paper.

The released corpus (`final_dialogues.jsonl`) is the canonical artefact that
all paper numbers are computed against. The synthesis pipeline is described
in the paper's appendix at the level of methodology; the synthesis prompts
and runners themselves are not redistributed in this repository.

**Are relationships between individual instances made explicit?**
Yes. Each dialogue references its source `blueprint_id`, `scenario_id`,
`category_key`, and `style_combination_index`, allowing per-axis aggregation
(e.g. per-category or per-style pass-rate breakdowns).

**Are there recommended data splits?**
This corpus is an *evaluation* benchmark; there is no train / val / test
split. The full set of 198 dialogues is meant to be used for evaluation.

**Are there any errors, sources of noise, or redundancies in the dataset?**
Curation produces 207 audit-passing blueprints; 9 dialogues were dropped
during Stage-4 rollout due to Planner / User Agent constraint violations,
yielding the released 198. The paper's judge-swap ablation (across GPT-5.4,
Claude-Opus-4.7, Kimi-K2.6) bounds residual scoring noise across judge
families; the paper's human-validation studies (n = 60 trigger points × 18
raters, Krippendorff α = 0.69; n = 80 Recovery items × 8 raters, B-preference
0.80) bound it against human raters. Per-trigger rubrics are themselves the
subjective interpretation point and are released so users can audit them.

**Is the dataset self-contained, or does it link to external resources?**
Largely self-contained. Personas were sampled from
[Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA)
(NVIDIA, CC-BY-4.0); persona attribution is preserved in the source data
per the upstream licence. The released `final_dialogues.jsonl` does not
redistribute the raw persona text — only the persona `uuid` and category
labels — so re-rendering the multi-aspect persona requires downloading
Nemotron-Personas-USA from HuggingFace.

**Does the dataset contain data that might be considered confidential?**
No. All personas are synthetic. No real-user data was collected.

**Does the dataset contain data that might be offensive or insulting?**
The 24 CSI communication styles include "verbal-aggressive" registers, in
which the synthetic user agent uses curt or confrontational language. This
is a deliberate factorial-design choice to evaluate model robustness to user
register variation; no real-user data was collected.

## Collection

**How was the data acquired?**
Synthetically generated by a five-stage curation pipeline: 50 personas
sampled from Nemotron-Personas-USA → 500 candidate proactive scenarios
(Stage 1) → 25 curated (persona, category) scenarios drawn from 19 personas
(Stage 2) → 250 turn-by-turn blueprints rendered under 24 communication
styles (Stage 3) → 207 audit-passing blueprints after independent-judge
review (Stage 4) → 198 final dialogues from a three-agent curation loop
(Planner, User Agent, Evaluated Model). See the paper Section 3 and
Appendix D for full pipeline details. Only the Stage-5 output
(`final_dialogues.jsonl`) is released in this repository.

**What mechanisms or procedures were used to collect the data?**
LLM API calls. Curation-time agents (Planner, User Agent, blueprint judge):
GPT-5.4 (Planner / User Agent), Gemini-2.5-Pro (Evaluated Model and
independent blueprint auditor). Offline evaluation, in the released code:
the user supplies any chat-completions endpoint (OpenAI, Anthropic, Gemini,
or any OpenAI-compatible endpoint such as a vLLM server) for the evaluated
model and the judge.

**Over what timeframe was the data collected?**
February through April 2026.

**Were any ethical review processes conducted?**
The benchmark contains no real-user data; no human-subjects review was
required for dataset creation. The two human-validation studies described
in the paper were conducted under an IRB-approved Prolific protocol with
informed consent, withdrawal rights, and compensation matching Prolific's
recommended hourly rate.

## Preprocessing

**Was any preprocessing / cleaning / labeling of the data done?**
- Blueprints that fail the independent-judge audit are excluded
  (40 of 250 received `NEEDS_REFINEMENT`; 0 received `FAIL`; 210 received
  `PASS`; of those, 3 hit downstream format errors and 9 hit Stage-4
  rollout failures, leaving 198).
- Dialogues that fail to reach the minimum 5-turn requirement are excluded.
- Per-trigger rubrics (PASS / PARTIAL / FAIL criteria) are authored
  prospectively by the Planner at turn t, before the assistant has responded
  at turn t+1.

**Was the "raw" data saved in addition to the preprocessed data?**
Internally, yes. In the released repository, no — only the final curated
corpus (`final_dialogues.jsonl`) is distributed. The audit trail
(Stage-1–3 artefacts, per-model evaluation outputs) is documented in the
paper's appendix but not redistributed.

## Uses

**Has the dataset been used for any tasks already?**
The accompanying paper (under double-blind review at NeurIPS 2026 Datasets &
Benchmarks track) evaluates 16 frontier and open-weight LLMs on this dataset
and compares per-trigger-type pass rates against six standard reasoning and
coding benchmarks.

**What (other) tasks could the dataset be used for?**

- Evaluating new LLMs on conversational proactivity.
- Studying model behaviour at different conversational phases (Emergent /
  Critical / Recovery decomposition).
- Calibrating LLM-as-judge agreement on subjective rubrics.
- Studying robustness to user-style variation (the 24-style factorial
  structure supports clean ablations).
- Source for preference-pair construction (rubric-conditioned vs vanilla
  Recovery responses), with the caveat that the dataset is not designed
  as a training target — see the paper's "Limitations" section.

**Is there anything about the composition or use of the dataset that might
impact future uses?**

- The corpus is **English-only** and **US-persona-only**. Norms around
  unsolicited advice and initiative-taking differ across cultures;
  ProactBench scores should not be used to guide deployment beyond
  US-English contexts.
- The seed model for curation is Gemini-2.5-Pro. Late-turn dialogue history
  reflects that model's prior responses; cross-model rankings are robust
  across cross-family judge swaps reported in the paper, but absolute pass
  rates should be interpreted relative to this seed.
- The scenario base is 25 distinct (persona, category) puzzles expanded
  across 24 communication styles (~8 dialogues per style on average).
  Per-(model, style) cell sizes are accordingly small; per-cell statistics
  should be interpreted with care.

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
Yes — the dataset is released publicly under the Apache-2.0 licence
alongside the paper.

**How will the dataset be distributed?**
The dataset ships as one JSONL file (`dataset/final_dialogues.jsonl`) in
this repository, indexed by Croissant 1.1 metadata
([`metadata.json`](metadata.json)). After acceptance, the dataset will
additionally be hosted on HuggingFace Datasets with the same licence.
During the review period, the anonymous repository archive is the canonical
host.

**When will the dataset be distributed?**
The repository is available now. The HuggingFace mirror will be created at
camera-ready time (post-acceptance), with the same Apache-2.0 licence.

**Will the dataset be distributed under a copyright or other intellectual
property (IP) licence?**
Apache 2.0 ([`LICENSE`](../LICENSE)). The persona-derived content inherits
the CC-BY-4.0 licence of the upstream Nemotron-Personas-USA dataset; persona
attribution (the `uuid` field) is preserved.

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
versions will use new release tags rather than overwriting the `v1.0.0`
corpus.

**If others want to extend / augment / build on / contribute to the dataset,
is there a mechanism for them to do so?**
Yes. The released corpus and the offline-evaluation pipeline are licensed
permissively (Apache 2.0). The curation pipeline (synthesis prompts and
runners) is not redistributed; users wishing to extend the corpus with new
persona categories or communication styles can reimplement the
methodology described in the paper's appendix against their own seed
personas.

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
