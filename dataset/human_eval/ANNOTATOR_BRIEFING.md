# Annotator Briefing — Proactivity Benchmark Human Validation

Thanks for helping with this study. This document explains **what you're rating, why it matters, and how to do it**. Plan ~80 minutes total. You'll see ~22–23 items. Read this end-to-end once before you start — it'll save you time.

---

## 1. What we're studying

We built a benchmark that measures how **proactively helpful** an AI assistant is in a conversation. Being proactive means doing more than literal instruction-following: noticing context the user hasn't spelled out, surfacing relevant suggestions or questions on its own, and helping the user reach their actual goal — even when they didn't fully describe it.

We used a GPT-5.4 as an **automated judge** to score 5 different models' responses across conversations.

**Your job is to validate that automated judge.** You'll see the same conversations and assistant responses, plus the same rubric the judge used, and you'll independently rate them. We'll then measure how well the judge agrees with humans.

You will **not** be told which model wrote which response, and you will **not** see the judge's score. Your judgment must be independent.

---

## 2. How the benchmark works

Each conversation in the benchmark is built from a **persona** and a **scenario**. The scenario has:

- A **hidden main goal** the user wants but never says aloud
  *("Plan a same-day go/no-go for a road trip" — the user only ever mentions fuel calculations.)*
- A handful of **implicit anchors** — small facts the user drops over several turns
  *("I prefer reusing old gear" / "the laptop will hold DSLR photos" / "we leave Saturday morning".)*
- An expected proactive behavior the model should produce

A scripted "user agent" plays the persona over up to 10 turns, leaking anchors gradually. At certain pre-declared turns (called **trigger points**) we score the assistant's response. There are three kinds of trigger:

### Trigger types

| Type | Where in the conversation | What it tests |
|---|---|---|
| **Emergent** | Early (turns 1–3) | The user reveals **one** anchor and asks a closed-sounding task. The model must infer an **unstated implication** from that single anchor and act on it without being asked. Pure instruction-following = fail. |
| **Critical** | Middle (turns 4–7) | Several anchors are now on the table. The model must **synthesize at least two of them** into a new conclusion or action the user did not request. Synthesis is the scored behavior. |
| **Recovery** | Late (turns 8–10) | The main task is finished and the user is wrapping up. The model must add **forward-looking value grounded in a specific detail** of *this* conversation. Generic "let me know if you need anything else" = fail. |

You will **not** be told which trigger type each item is. You'll just see the conversation, the response, and a rubric. The rubric is the source of truth — read it every time.

---

## 3. What you'll see for each item

Each screen has four parts:

1. **Conversation so far** — chat transcript. The user's last message has a gold border; that's what the assistant was responding to.
2. **Known information at this point** — the bulleted list of facts the assistant has been told (the anchors disclosed so far).
3. **The assistant's response** — green left border. *This is what you're rating.*
4. **Rating criteria** — three labeled paragraphs (Pass / Partial / Fail) written specifically for this item. They tell you exactly what counts as each rating for *this* response.

The rubric is **per item**, not a global standard. What earns a Pass on one item may be a Fail on another. **Always read the rubric.**

---

## 4. How to rate

Pick **one** of three labels:

- **Pass** — the response meets the Pass criteria. Typically: the model went beyond literal instruction-following and added something useful that's grounded in the disclosed anchors.
- **Partial** — the response shows some relevant awareness but doesn't quite get there. Typically: it asks a relevant clarifying question without proposing a concrete next step, or it gestures at the implication without acting on it.
- **Fail** — the response meets the Fail criteria. Typically: purely reactive, generic, ignores the anchors, or only restates information already given.

Then write **a short rationale** (one sentence is fine). Tell which criterion the rating maps to and what in the response makes it fit. Examples:

> Pass — explicitly adds a checksum verification step beyond the requested SMART check, citing the photo-batch use case. Matches Pass criterion.

> Partial — flags photo-storage reliability as a concern but doesn't propose a concrete integrity step, which Pass requires.

> Fail — gives a generic SMART/scan procedure with no reference to the photo anchor.

Finally, rate **confidence (1–5)**. 1 = not sure (rubric is borderline / response ambiguous). 3 = moderate. 5 = very confident. Use the lower end honestly — split judgments are useful data.

---

## 5. Common pitfalls — please avoid

- **Don't grade for general "proactiveness."** Use the rubric. The rubric defines proactivity for *this* response.
- **Don't reward instruction-following dressed up as proactivity.** Bullet lists, polished structure, polite hedging — none of these alone clear the bar. The model has to act on something it wasn't explicitly asked about.
- **Don't penalize style.** Casual, terse, or chatty replies can still pass. Fancy formatting can still fail.
- **Don't over-credit clarifying questions.** A clarifying question can be Partial if it touches the right concern, but Pass usually requires a concrete forward step, not just inquiry. (Exact bar is in the rubric.)
- **Don't carry context between items.** Each item stands alone. The same model response would be scored on its merits per the rubric for that item, regardless of what you saw earlier.
- **Don't talk to other annotators about specific items.** We need independent judgments. Talking about the *interface* or *the rubric in general* is fine; specific items are not.

---

## 6. Practical details

- **Time estimate:** ~3.5 min/item × 22–23 items ≈ ~80 minutes
- **Saving:** every submission is autosaved to the server *and* backed up in your browser; you can close the tab and resume any time using the same URL + ID + code
- **If a submit fails:** the button changes to "Retry submit" and your text is preserved — try again


# Thank you very much for participating!
