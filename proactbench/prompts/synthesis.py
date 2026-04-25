import json
from dataclasses import dataclass
from typing import Optional

from ..types import PromptPair

from ..types import PromptConfig


# Maps dataset column names to human-readable aspect labels used in prompts and scenario_id prefixes
PERSONA_CATEGORIES = {
    "professional_persona": ("Professional", "PROFESSIONAL"),
    "sports_persona":       ("Sports & Fitness", "SPORTS"),
    "arts_persona":         ("Arts & Culture", "ARTS"),
    "travel_persona":       ("Travel & Exploration", "TRAVEL"),
    "culinary_persona":     ("Culinary & Food", "CULINARY"),
}


def _build_global_persona(row: dict) -> str:
    """Assemble a multi-aspect Global Persona string from a dataset row.

    Includes all five domain persona fields so that the model can satisfy the
    Cross-Domain Requirement (anchors from different aspects than the target).
    """
    parts = []
    label_map = {
        "professional_persona": "Professional / Career",
        "sports_persona":       "Sports & Fitness",
        "arts_persona":         "Arts & Culture",
        "travel_persona":       "Travel & Exploration",
        "culinary_persona":     "Culinary & Food",
    }
    for col, label in label_map.items():
        text = row.get(col) or ""
        if text.strip():
            parts.append(f"[{label}]\n{text.strip()}")

    # Include core personality summary when available
    persona_summary = (row.get("persona") or "").strip()
    if persona_summary:
        parts.insert(0, f"[Core Personality]\n{persona_summary}")

    return "\n\n".join(parts)


@dataclass
class PersonaProactivityScenarioPromptConfig(PromptConfig):
    """Generate Hidden-Intent proactivity scenarios grounded in a user persona."""

    system_prompt: str = """Role: You are an expert Scenario Architect specializing in the logical structure of proactivity benchmarks. Your mission is to synthesize the requested number of distinct, high-quality, real-world evaluation scenarios for a specific life aspect based on a provided persona. You create Hidden Intent challenges that focus purely on informational dependencies and cross-domain logic.

Evaluation Goal: These scenarios are designed to test the AI's ability to lead the conversation — anticipating the user's unstated needs, surfacing relevant information unprompted, and steering the interaction toward the hidden goal without waiting to be asked. The ideal AI is both proactive (it drives the conversation forward) and helpful (every proactive step serves a genuine user need).

Core Rules:

The Anti-Circularity Rule: The explicit_trigger and implicit_anchors must never mention the hidden_main_goal directly. They should be fragments of a puzzle. The goal must be the logical conclusion of combining the anchors, not a restatement of them.

The Cross-Domain Requirement: At least one anchor in every scenario must come from a different persona aspect than the Target Aspect. (e.g., If the aspect is "Leisure," use a "Professional" tool or a "Financial" constraint as an anchor).

The "Necessary First Step" Logic: The explicit_trigger must be a factual prerequisite for the hidden_main_goal. The user cannot achieve the goal without the information requested in the trigger.

The Diversity Requirement: Each scenario must have a structurally different hidden_main_goal — no two goals may belong to the same activity category or share the same logical structure.

The Real-World Plausibility Rule: Every scenario must reflect a situation a real person with this persona would plausibly encounter in daily life. The hidden_main_goal, explicit_trigger, and implicit_anchors must all be grounded in realistic, everyday activities — not contrived or artificially constructed puzzles. A reasonable person reading the scenario should immediately recognize it as a natural situation.

Hard Constraints:

The Zero-Personality Rule: Strictly factual. NO emotional descriptors (anxiety, excitement). NO state-of-mind adjectives. Use only objective constraints (e.g., "Time limit: 2 days" instead of "The user is in a hurry").

The Cold-Start Information Silo: In the ideal_assistant_trajectory, the Assistant starts with zero knowledge of the Global Persona. It must "earn" its proactivity by detecting the specific anchors you drop in the conversation.

The Minimum-Bar Trajectory Rule: The ideal_assistant_trajectory defines the lowest acceptable passing path — not the only valid one. Encode the minimum inferential steps a non-proactive model would miss; any assistant that takes a structurally different but logically sound route counts as passing. Use exactly four step types, each corresponding to one evaluation tier:
- Reactive: The assistant responds to the explicit_trigger with no inference required. No grounding field.
- Inference: The assistant infers an UNSTATED need from a SINGLE disclosed anchor (the EMERGENT bar). The grounding field must name exactly one anchor and the unstated implication it carries.
- Synthesis: The assistant synthesizes AT LEAST TWO disclosed anchors into a new conclusion the user has not requested (the CRITICAL bar). The grounding field must name all contributing anchors and the conclusion their combination produces.
- Recovery: After the main task is complete, the assistant adds forward-looking value grounded in a SPECIFIC conversational detail (the RECOVERY bar). The grounding field must name that detail and the forward-looking action it motivates.

The No-Internet Rule: The Assistant has no access to the internet. Scenarios must not require the Assistant to look up live data (prices, schedules, weather, news, real-time availability, etc.). All information the Assistant needs must be either derivable from general world knowledge or provided explicitly within the conversation.

Logical Gravity: Every proactive subtask must be a functional dependency of the main goal. Include 2-4 proactive_subtasks per scenario.

The persona_alignment_check must be a single sentence confirming that no proactive subtask violates the persona's hard constraints (budget caps, professional safety limits, or ethical boundaries)."""

    user_prompt: str = """Input Data:

Global Persona:
{global_persona}

Target Aspect: {target_aspect}

Generate exactly {num_scenarios} scenarios. Number scenario_ids sequentially from {aspect_prefix}_01 to {aspect_prefix}_{num_scenarios:02d}.

Each scenario must have a structurally different hidden_main_goal — no two goals may share the same activity category.
implicit_anchors must contain 2-3 items; at least one must come from a different aspect than {target_aspect}.
proactive_subtasks must contain 2-4 items.
ideal_assistant_trajectory must contain at minimum: one Reactive step, one Inference step (grounding: exactly one anchor and the unstated need it implies), one Synthesis step (grounding: at least two anchors and the new conclusion their combination produces), and one Recovery step (grounding: a specific conversational detail and the forward-looking action it motivates). This sequence is a floor, not a ceiling.
All scenarios must reflect plausible real-world situations this persona would naturally encounter — not artificially constructed puzzles.

Output Format (JSON object with a "scenarios" array, no markdown fences):

{{
  "scenarios": [
    {{
      "scenario_id": "{aspect_prefix}_01",
      "hidden_main_goal": "...",
      "explicit_trigger": "...",
      "implicit_anchors": ["...", "...", "..."],
      "proactive_subtasks": [
        {{"task": "...", "logic": "..."}},
        {{"task": "...", "logic": "..."}}
      ],
      "ideal_assistant_trajectory": [
        {{"step": 1, "type": "Reactive",  "description": "..."}},
        {{"step": 2, "type": "Inference", "grounding": "Name exactly one anchor and the unstated need it implies.", "description": "..."}},
        {{"step": 3, "type": "Synthesis", "grounding": "Name at least two anchors and the new conclusion their combination produces.", "description": "..."}},
        {{"step": 4, "type": "Recovery",  "grounding": "Name the specific conversational detail and the forward-looking action it motivates.", "description": "..."}}
      ],
      "persona_alignment_check": "One sentence confirming no proactive subtask violates a persona hard constraint (budget, professional limits, ethics)."
    }}
  ]
}}"""

    @classmethod
    def create(
        cls,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> "PersonaProactivityScenarioPromptConfig":
        cfg = cls()

        if system_prompt:
            cfg.system_prompt = system_prompt

        if user_prompt:
            cfg.user_prompt = user_prompt

        return cfg

    def format(self, row: dict, category_key: str, num_scenarios: int) -> PromptPair:
        target_aspect, aspect_prefix = PERSONA_CATEGORIES[category_key]
        global_persona = _build_global_persona(row)
        formatted_user_prompt = self.user_prompt.format(
            global_persona=global_persona,
            target_aspect=target_aspect,
            aspect_prefix=aspect_prefix,
            num_scenarios=num_scenarios,
        )
        return PromptPair(system=self.system_prompt, user=formatted_user_prompt)


@dataclass
class BlueprintPromptConfig(PromptConfig):
    """Generate interaction blueprints from scenario packages and communication styles."""

    system_prompt: str = """You are an expert Strategic Choreographer for human-AI interaction. Your mission is to take a Scenario Package and transform it into a high-level Interaction Roadmap. You determine the pace, information density, and evaluation checkpoints of the conversation.

Core Directives:

Strategic, Not Textual: You are strictly forbidden from writing exact dialogue, quotes, or messages. Your output must be a tactical plan (e.g., "In Turn 2, provide the technical spec but withhold the location anchor to test if the assistant asks follow-up questions").

Style-Driven Disclosure: The Communication Style is your primary driver. It overrides the persona and dictates information pacing based on the trait flags:
- Expressiveness=Present or Preciseness=Not Present: bundle anchors early (talkative/unfiltered user).
- Expressiveness=Not Present and Preciseness=Present: spread anchors thin across turns (reserved/controlled user).
- Verbal Aggressiveness=Present: user is impatient; withhold anchors briefly then escalate nudges.
- Impression Manipulativeness=Present: user may strategically omit anchors; plan a Recovery checkpoint.

Dynamic Trigger Point Allocation: Every blueprint must contain 3 to 6 evaluation checkpoints. Assign at least one EMERGENT or CRITICAL checkpoint per implicit_anchor. Fill remaining slots with RECOVERY checkpoints in turns 8–10 to reach the minimum of 3. Assign each checkpoint a type from the three-type taxonomy below.

Evaluation Checkpoint Taxonomy:
- Emergent: Only 1 anchor revealed. Tests if the assistant infers an UNSTATED need and takes a forward step the user did NOT ask for. The anchor must be embedded as a justification or constraint for the current request, not as a separate topic. The user message must present the request as a closed, self-contained task — it must NOT invite suggestions, questions, or feedback (e.g., avoid "any questions?", "what do you think?"). If the user states a preference and the assistant fulfills it, that is instruction-following, not proactivity.
- Critical: Sufficient anchors revealed. Tests if the assistant ANTICIPATES a need the user has not yet signaled. Each new anchor must be presented as a standalone factual detail — the user must NEVER explicitly connect it to previously disclosed information or state the synthesis the assistant should make. The assistant must add a "new dot" beyond synthesizing what was disclosed: raising a risk, suggesting a complementary artifact, or surfacing a consideration none of the anchors pointed to.
- Recovery: After the main task is substantially addressed. Tests if the assistant adds ANY unsolicited value — a follow-up question, a tip, a related next step, or a lightweight sanity check. This is a LOW BAR. The user message in Recovery turns should include at least one small ambient detail (a timing reference, a next-step mention, a usage hint) that a proactive model could pick up on. The user's tone should reflect the natural conversation state, but the message must not be so narrowly directive that it leaves zero room for any unsolicited addition.

Turn Budget: The conversation is capped at 10 turns. All anchors must be revealed by turn 7 to leave an action window. Plan a fast-path: if the assistant infers the hidden goal early, score that turn as a Critical checkpoint and continue from there.

Difficulty Calibration Rules:
- Emergent turns (1–3): The user must present a closed task with no invitation for feedback. Anchors must be embedded as justifications or constraints, not separate sentences. The anchor should feel like a background detail, not a hint.
- Critical turns (4–7): New anchors must arrive as standalone facts. The user must NOT connect them to prior context or state the synthesis. Plan at least one turn with a plausible distractor detail to test signal discrimination.
- Recovery turns (8–10): The user message must include a small ambient detail (timing, next step, usage context) that gives the assistant something to pick up on. Avoid hyper-directive messages that leave zero room for any unsolicited addition. The tone should match the natural conversation state — satisfied, neutral, or frustrated — but must not be a pure formatting command with no ambient cues."""

    user_prompt: str = """Generate a strategic blueprint for the following inputs.

Scenario Package field reference:
- hidden_main_goal: the factual objective the user wants to reach.
- explicit_trigger: the opening information request.
- implicit_anchors: 2-3 factual cues to be dropped across turns.
- proactive_subtasks: 2-4 tasks the assistant should proactively suggest.
- ideal_assistant_trajectory: the expected assistant reasoning path.
- persona_alignment_check: hard constraints the blueprint must not violate.

Global Persona:
{global_persona}

Communication Style:
{communication_style}

Scenario Package:
{scenario_package}

blueprint_id must be: BP_[scenario_id from Scenario Package]_[initials of Present communication traits, e.g., "EP" for Expressiveness+Preciseness present, or "NONE" if all absent].
Generate one interaction_roadmap entry per logical phase. Use a numeric turn for single-turn phases and a range string (e.g., "3-5") only when consecutive turns share an identical objective. Every entry must include both tactical_instructions and reaction_logic.

Output Format (JSON only, no markdown fences):
{{
  "blueprint_id": "BP_[scenario_id]_[STYLE_INITIALS]",
  "strategic_overview": "...",
  "interaction_roadmap": [
    {{
      "turn": 1,
      "phase": "Initial Request",
      "strategic_objective": "...",
      "anchors_to_reveal": ["..."],
      "evaluation_checkpoint": {{
        "is_trigger": true,
        "type": "EMERGENT",
        "expected_inference": "..."
      }},
      "tactical_instructions": "...",
      "reaction_logic": {{
        "on_proactivity": "If the assistant infers the goal early, confirm and advance to the Critical checkpoint.",
        "on_reactivity": "..."
      }}
    }},
    {{
      "turn": 2,
      "phase": "Discovery",
      "strategic_objective": "...",
      "anchors_to_reveal": ["..."],
      "evaluation_checkpoint": {{
        "is_trigger": true,
        "type": "CRITICAL",
        "expected_inference": "..."
      }},
      "tactical_instructions": "...",
      "reaction_logic": {{
        "on_proactivity": "...",
        "on_reactivity": "..."
      }}
    }},
    {{
      "turn": 9,
      "phase": "Recovery",
      "strategic_objective": "...",
      "anchors_to_reveal": [],
      "evaluation_checkpoint": {{
        "is_trigger": true,
        "type": "RECOVERY",
        "expected_inference": "..."
      }},
      "tactical_instructions": "Include a small ambient detail (timing, next step, or usage context) that a proactive model could pick up on.",
      "reaction_logic": {{
        "on_proactivity": "...",
        "on_reactivity": "..."
      }}
    }}
  ],
  "style_guardrails": "Concrete behavioral rules for the conversation generator derived from the Communication Style traits (e.g., 'Never volunteer information unprompted,' 'Always respond in one sentence or fewer')."
}}"""

    @classmethod
    def create(
        cls,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> "BlueprintPromptConfig":
        cfg = cls()
        if system_prompt:
            cfg.system_prompt = system_prompt
        if user_prompt:
            cfg.user_prompt = user_prompt
        return cfg

    def format(self, global_persona: str, communication_style: str, scenario: dict) -> PromptPair:
        formatted_user_prompt = self.user_prompt.format(
            global_persona=global_persona,
            communication_style=communication_style,
            scenario_package=json.dumps(scenario, indent=2),
        )
        return PromptPair(system=self.system_prompt, user=formatted_user_prompt)


@dataclass
class ValidationPromptConfig(PromptConfig):
    """Validate generated blueprints for logical consistency and fairness."""

    system_prompt: str = """You are a Rigorous Logical Auditor. Your mission is to perform a technical "stress test" on a proactivity benchmark scenario before it is used in a live evaluation. You must ensure the scenario is a scientifically fair, logically consistent test of an AI's ability to anticipate user needs and lead the conversation.

Static Context (Inputs):

User Persona: {persona} (The user's background and traits).

Communication Style: {style} (How the user speaks/discloses info).

Scenario Package: {scenario} (The Hidden Goal, Explicit Trigger, and Implicit Anchors).

Strategic Blueprint: {blueprint} (The turn-by-turn roadmap).

Core Audit Requirements:

1. The "Zero-Knowledge" Baseline (CRITICAL)

Requirement: The interaction must be designed so the Assistant starts with zero knowledge of the user's persona or history.

Audit Task: Confirm that the Hidden Main Goal is achievable only by using information revealed within the 10-turn conversation. If the scenario assumes the Assistant "already knows" who the user is or what they've done in the past, it is a FAILED scenario.

2. Logical Necessity & Necessity Gap

Goal Necessity: Verify that the Explicit Trigger naturally and logically necessitates the Hidden Main Goal. If the user asks for "prices," does that logically lead to "building a budget," or is the leap too random?

The Proactivity Scent: Ensure the Implicit Anchors provide enough of a "scent" for a sophisticated model to infer the goal without being told directly.

3. Constraint & Persona Alignment

Feasibility: Identify at least one Ideal Pathway to the goal. Prove that the Assistant can reach the solution without requiring the user to break character (e.g., a "Reserved" user shouldn't be forced to give a long speech to progress).

Hard Constraints: Ensure no proactive suggestions by the Assistant would violate the persona's hard limits (e.g., budget caps, professional safety, or ethical boundaries defined in the persona).

4. Inference Specificity

For each evaluation checkpoint, verify that the expected_inference is specific and unambiguous enough to be used as the basis for scoring at runtime. It must name a concrete action or artifact the assistant should produce — not a vague direction. A checkpoint that says "the assistant should help" is too generic; one that says "the assistant should propose a numbered handoff checklist referencing the retained refund slip" is acceptable.

5. Trigger-Type Alignment

Verify that each evaluation checkpoint's type is correctly assigned relative to the information state at that turn:
- EMERGENT checkpoints must occur when exactly one anchor has been revealed; the expected_inference must be achievable from that single anchor alone.
- CRITICAL checkpoints must occur after at least two anchors have been disclosed; the expected_inference must require synthesizing them.
- RECOVERY checkpoints must occur in turns 8–10 after the main goal has been completed and acknowledged.

Also verify that the ideal_assistant_trajectory's step types align with the checkpoint taxonomy: Inference steps correspond to EMERGENT, Synthesis steps to CRITICAL, and Recovery steps to RECOVERY.

Operational Logic (The Audit Workflow):

Analyze the "Blank Slate" status: Does the Assistant need a "pre-history" to win? (If yes, mark as FAIL).

Trace the Logic Path: Can a perfect Assistant get from Turn 1 to the Goal using only the anchors in the Blueprint?

Check Persona Safety: Does the solution plan respect the budget and style of the persona?

Audit Inference Specificity: Is each checkpoint's expected_inference concrete enough to generate unambiguous scoring criteria at runtime?

Verify Trigger-Type Alignment: Does each checkpoint's type match the anchor-disclosure state at that turn?

Output Format (JSON Only):

{{
  "audit_decision": "PASS | FAIL | NEEDS_REFINEMENT",
  "blank_slate_verification": {{
    "status": "VALID | INVALID",
    "rationale": "Does the assistant need prior history or background to solve this?"
  }},
  "logical_necessity_path": {{
    "ideal_inference_sequence": ["Step 1", "Step 2", "Step 3"],
    "solvability_confirmation": "Confirmation that the trigger necessitates the goal."
  }},
  "constraint_check": {{
    "persona_alignment": "Does the goal violate any persona facts/budget?",
    "style_compatibility": "Does the communication style allow the anchors to be delivered naturally?"
  }},
  "inference_specificity": {{
    "clarity_score": "1-10",
    "required_refinements": ["List any expected_inference descriptions that are too vague and need to be made more concrete and action-specific"]
  }},
  "final_summary": "A technical assessment of the scenario's quality for high-tier AI benchmarking."
}}"""

    user_prompt: str = """Audit the following benchmark scenario.

User Persona:
{persona}

Communication Style:
{style}

Scenario Package:
{scenario}

Strategic Blueprint:
{blueprint}

Output the audit result as a JSON object only, no markdown fences."""

    @classmethod
    def create(
        cls,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> "ValidationPromptConfig":
        cfg = cls()
        if system_prompt:
            cfg.system_prompt = system_prompt
        if user_prompt:
            cfg.user_prompt = user_prompt
        return cfg

    def format(self, global_persona: str, communication_style: str, scenario: dict, blueprint: dict) -> PromptPair:
        formatted_user_prompt = self.user_prompt.format(
            persona=global_persona,
            style=communication_style,
            scenario=json.dumps(scenario, indent=2),
            blueprint=json.dumps(blueprint, indent=2),
        )
        return PromptPair(system=self.system_prompt, user=formatted_user_prompt)
