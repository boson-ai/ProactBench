"""HuggingFace Datasets loading script for ProactBench.

Once the dataset is hosted on HuggingFace, users can load it with:

    from datasets import load_dataset
    ds = load_dataset("anonymous/proactbench")

For local use, this script can be invoked with the local repository path:

    ds = load_dataset("path/to/proactbench/dataset")
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import datasets


_CITATION = r"""
@inproceedings{anonymous2026proactbench,
  title={ProactBench: What the User Didn't Ask},
  author={Anonymous},
  booktitle={Under review at NeurIPS 2026},
  year={2026}
}
"""

_DESCRIPTION = """\
ProactBench is a benchmark for measuring conversational proactivity in multi-turn LLM dialogues.
It decomposes proactivity into three phase-tied trigger types (Emergent, Critical, Recovery) and
uses a three-agent evaluation architecture (Planner, User Agent, Evaluated Model) with
information asymmetries that defend against rubric leakage, post-hoc rationalisation,
style-confounded scoring, and information dumps. The corpus contains 198 curated dialogues
with 624 trigger points across 5 persona categories and 24 communication styles drawn from
the Communication Styles Inventory (CSI).
"""

_HOMEPAGE = "https://github.com/anonymous/ProactBench"
_LICENSE = "Apache-2.0"  # persona-derived content inherits Nemotron-Personas-USA CC-BY-4.0


class ProactBenchConfig(datasets.BuilderConfig):
    """Builder configuration for one ProactBench split (file)."""

    def __init__(self, name, description, file_name, **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        self.file_name = file_name


class ProactBench(datasets.GeneratorBasedBuilder):
    """ProactBench dataset.

    Three configurations:

    - `dialogues`        — main benchmark corpus (198 rows in `final_dialogues.jsonl`)
    - `blueprints`       — turn-by-turn interaction blueprints (210 audit-passing rows)
    - `tasks`            — per-persona scenarios (50 rows)
    """

    BUILDER_CONFIGS = [
        ProactBenchConfig(
            name="dialogues",
            description="Main benchmark corpus: 198 curated dialogues with 624 trigger points.",
            file_name="final_dialogues.jsonl",
        ),
        ProactBenchConfig(
            name="blueprints",
            description="Audit-passing blueprints used to roll out the dialogue corpus.",
            file_name="validated_blueprints.jsonl",
        ),
        ProactBenchConfig(
            name="tasks",
            description="Per-persona scenarios with hidden goals, anchors, and ideal trajectories.",
            file_name="tasks.jsonl",
        ),
    ]
    DEFAULT_CONFIG_NAME = "dialogues"

    def _info(self):
        if self.config.name == "dialogues":
            features = datasets.Features({
                "uuid": datasets.Value("string"),
                "unique_id_eval": datasets.Value("string"),
                "blueprint_id": datasets.Value("string"),
                "scenario_id": datasets.Value("string"),
                "category_key": datasets.Value("string"),
                "style_combination_index": datasets.Value("int32"),
                "evaluated_model": datasets.Value("string"),
                "num_turns_completed": datasets.Value("int32"),
                "trigger_stats": datasets.Value("string"),   # JSON-encoded dict
                "trigger_points": datasets.Value("string"),  # JSON-encoded list
                "turn_records": datasets.Value("string"),    # JSON-encoded list
                "token_usage": datasets.Value("string"),     # JSON-encoded dict
            })
        elif self.config.name == "blueprints":
            features = datasets.Features({
                "uuid": datasets.Value("string"),
                "blueprint_id": datasets.Value("string"),
                "scenario_id": datasets.Value("string"),
                "category_key": datasets.Value("string"),
                "persona_uuid": datasets.Value("string"),
                "style_combination_index": datasets.Value("int32"),
                "strategic_overview": datasets.Value("string"),
                "interaction_roadmap": datasets.Value("string"),  # JSON-encoded list
                "style_guardrails": datasets.Value("string"),
            })
        else:  # tasks
            features = datasets.Features({
                "uuid": datasets.Value("string"),
                "professional_persona_scenarios": datasets.Value("string"),
                "sports_persona_scenarios": datasets.Value("string"),
                "arts_persona_scenarios": datasets.Value("string"),
                "travel_persona_scenarios": datasets.Value("string"),
                "culinary_persona_scenarios": datasets.Value("string"),
            })

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # Local path next to this script
        here = Path(os.path.dirname(os.path.abspath(__file__)))
        path = here / self.config.file_name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"file_path": str(path)},
            )
        ]

    def _generate_examples(self, file_path):
        with open(file_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # Stringify nested objects so the schema stays portable
                for k, v in list(row.items()):
                    if isinstance(v, (dict, list)):
                        row[k] = json.dumps(v, ensure_ascii=False)
                yield i, row
