prompt_for_predict = """{practice_item_simulated_input}

---

## When completing the above tasks, you may refer to the experiential memory

The experiential memory consists of two types:

**Pattern Experience**
Records this user's preferences and behavioral habits. The subject is "this user."

**Calibration Experience**
Records error tendencies, blind spots, or things to pay special attention to when the model reasons about this user's behavior. The subject is "I (the model) when reasoning."

---

{exp_memory_text}

---

Based on the experiential memory and behavioral history above, predict the outcome for each practice task.

Before predicting each task, complete the following two reasoning steps:
1. List the relevant Pattern Experience entries and explain this user's behavioral tendencies in this type of scenario.
2. List the relevant Calibration Experience entries and explain what I need to watch out for when reasoning.

Then give your prediction based on the above reasoning.

Answer each task in order, with each task's output separate."""






prompt_for_reflect = """# Background
We are predicting behavior from a user's long history of behavioral sequences. The user has hundreds of historical behavior records, each containing a scene description and user response. The ultimate goal is to predict the user's behavior in new scenes.

To improve prediction accuracy, we use a Self-Practice method: construct simulated practice questions with ground-truth answers from the user's historical sequences, and let the model practice, reflect, and generalize repeatedly to accumulate two types of experiential memory before the final prediction:
- Pattern Experience: generalizations about this user's preferences and behavioral habits
- Calibration Experience: error tendencies and things to watch out for when the model reasons about this user's behavior

Each practice round has three steps: ① predict using current memory; ② reflect against ground truth and generate proposals; ③ every N rounds, review the proposal pool and consolidate well-supported proposals into formal memory. Each proposal must also pass two self-review checks before entering the pool: a groundedness check (ensures proposals genuinely depend on historical data) and a generalizability check (ensures proposals are not specific to a single scene).

Your current task is **Step ②: Reflection and Proposals** — deeply reflect on this prediction against the ground truth and propose memory corrections.

---

You are a rigorous behavior prediction quality analyst. Your task is to deeply reflect on the just-completed prediction and propose corrections to the experiential memory used.

# Experiential Memory Description

The experiential memory consists of two types:

**Pattern Experience**
Records this user's preferences and behavioral habits. The subject is "this user."
Correct example: "This user prefers live streams featuring outdoor natural scenes."
Incorrect example: "Check whether there is a natural scene when predicting." (This is a model reminder, not a user trait.)

**Calibration Experience**
Records error tendencies, blind spots, or things to pay special attention to when the model reasons about this user's behavior. The subject is "I (the model) when reasoning."
Correct example: "I tend to overestimate this user's interaction frequency with commerce content; actual interaction is lower."
Correct example: "When predicting this user's response to emotional content, pay extra attention to time-of-day factors."
Incorrect example: "This user dislikes commerce content." (This is a user trait; it belongs in Pattern Experience.)

---

# This Round's Practice Review

{exp_memory_text}

---

## Practice Input
{practice_item_simulated_input}

---

## Model Prediction
{predict_response}

---

## Ground Truth
{practice_item_simulated_label}

---

# Deep Reflection and Summary

## Step 1: Judge each task's prediction
For each task, determine whether the prediction is correct or has a significant deviation. Label "correct" or "incorrect/deviation."

## Step 2: Analyze each task
**Analysis is required regardless of whether the prediction was correct.**

- **Correct** predictions: summarize what worked. Focus on:
  - Did this success reveal a new user pattern or model reasoning insight?
  - Which memory entries directly supported the correct reasoning? These can be reinforced.
  - Was this a lucky correct answer (the reasoning logic was flawed but the result happened to be right)? If so, the memory still needs correction.
- **Incorrect** predictions: analyze the root cause. Focus on:
  - Was a Pattern Experience entry inaccurate or incomplete?
  - Was a Calibration Experience entry missing, causing an oversight or bias?
  - Does existing memory conflict with or not apply to this scene?

## Step 3: Generate correction proposals
Proposals may be generated for any task. Before writing each proposal, verify it meets all of the following quality requirements:

1. **Correct category**: clearly determine whether this proposal belongs to Pattern Experience or Calibration Experience; do not mix them up.
2. **Usability**: the memory must not depend on external information unavailable to the model at inference time (e.g., user identity data, real-time engagement statistics).
3. **Judgeability**: the conditions must be directly determinable from the input content description; avoid details that require inference to confirm.
4. **Avoid false precision**: do not use specific numbers without statistical basis (e.g., "correction factor 0.1", "extend by 300%"); use qualitative terms instead (e.g., "significantly longer", "slightly higher").
5. **Generalizability**: the memory should apply to similar scenes; do not write a single special observation directly as a memory entry.

---

Output your reflection in the following format:

## Per-task prediction assessment
(For each task, state whether the prediction was correct; label "correct" or "incorrect/deviation.")

## Per-task analysis and proposals
(For each task, provide analysis and list proposals; if no new proposals for a task, write "no new proposals.")
Each proposal on its own line as a single line of text, starting with "【提议】", in the following format:
【提议】brief trigger scene description | proposal type: add Pattern / add Calibration / modify existing / remove existing | specific content

(If there are no proposals at all this round, write "no proposals this round.")"""






prompt_for_review = """# Background
We are predicting behavior from a user's long history of behavioral sequences. The user has hundreds of historical behavior records, each containing a scene description and user response. The ultimate goal is to predict the user's behavior in new scenes.

To improve prediction accuracy, we use a Self-Practice method: construct simulated practice questions with ground-truth answers from the user's historical sequences, and let the model practice, reflect, and generalize repeatedly to accumulate two types of experiential memory before the final prediction:
- Pattern Experience: generalizations about this user's preferences and behavioral habits
- Calibration Experience: error tendencies and things to watch out for when the model reasons about this user's behavior

Each practice round has three steps: ① predict using current memory; ② reflect against ground truth and generate proposals; ③ every N rounds, review the proposal pool and consolidate well-supported proposals into formal memory. Each proposal must also pass two self-review checks before entering the pool: a groundedness check and a generalizability check.

Your current task is **Step ③: Memory Review** — carefully update the experiential memory based on the proposal pool. Only well-supported proposals should be incorporated.

---

You are an experience management expert. Your task is to carefully update the current experiential memory based on accumulated correction proposals.

## Experiential Memory Description

The experiential memory consists of two types:

**Pattern Experience**
Records this user's preferences and behavioral habits. The subject is "this user."

**Calibration Experience**
Records error tendencies, blind spots, or things to pay special attention to when the model reasons about this user's behavior. The subject is "I (the model) when reasoning."

---

## Current Experiential Memory

{exp_memory_text}

### Proposal Pool
{proposal_text}

---

## Review Task

### Step 1: Group the proposals
Group all proposals in the pool by content similarity. Identify which proposals express the same observation.

### Step 2: Apply adoption rules to each group
Execute the following rules strictly; do not skip any:

**Rules for adding or modifying memory:**
- An observation supported by 3 or more proposals → may be incorporated into memory or used to modify existing memory
- An observation supported by only 1–2 proposals → must remain in the proposal pool; do not incorporate into memory

**Rules for deleting or demoting existing memory:**
- An existing memory entry questioned by 3 or more proposals → modify that entry or demote it to the proposal pool
- An existing memory entry questioned by only 1–2 proposals → leave unchanged

**Note: The proposal pool is not a to-do list; it is a pool of candidate observations. Normally, most proposals should remain in the pool after each review.**

### Step 3: Check memory quality
Before incorporating a proposal into memory, verify each entry meets the following requirements:
1. **Correct category**: Pattern Experience describes the user; Calibration Experience describes model reasoning caveats. Do not mix them up.
2. **Usability**: must not depend on external information unavailable to the model at inference time.
3. **Judgeability**: conditions must be directly determinable from the input content.
4. **Avoid false precision**: do not use specific numbers without statistical basis; use qualitative terms instead.
5. **Generalizability**: must apply to similar scenes; must not be a single special observation.

---

## Output Requirements

Output strictly in the following two steps; do not merge or skip either step.

### Step 1: Review decision summary
First explain this round's review decisions in natural language:
- Which proposal groups were adopted (note corresponding proposal indices)
- Which existing memory entries were modified or demoted (note specific entries)
- Remaining proposals stay in the pool; no need to explain each one

### Step 2: Output the complete memory JSON
Output JSON strictly following these rules:
- user_exp and model_exp: output the complete post-review memory lists
- proposal: must include the full text of both: (1) all proposals judged "retained" in Step 1, copied verbatim; (2) any demoted existing memory entries

```json
{{
  "user_exp": [...],
  "model_exp": [...],
  "proposal": [...]
}}"""


prompt_for_evaluate_after_main = """You may refer to the experiential memory below to complete the above tasks. The experiential memory consists of two types:

**Pattern Experience**
Records this user's preferences and behavioral habits. The subject is "this user."

**Calibration Experience**
Records error tendencies, blind spots, or things to pay special attention to when the model reasons about this user's behavior. The subject is "I (the model) when reasoning."

Here is the experiential memory:
{memory}

Now complete the tasks above. Make sure to follow the output format required by those tasks.
"""




prompt_for_generate_perturbations = """# Background
We are predicting behavior from a user's long history of behavioral sequences. The user has hundreds of historical behavior records, each containing a scene description and user response. The ultimate goal is to predict the user's behavior in new scenes.

To improve prediction accuracy, we use a Self-Practice method: construct simulated practice questions with ground-truth answers from the user's historical sequences, and let the model practice, reflect, and generalize repeatedly to accumulate two types of experiential memory before the final prediction:
- Pattern Experience: generalizations about this user's preferences and behavioral habits
- Calibration Experience: error tendencies and things to watch out for when the model reasons about this user's behavior

Each practice round has three steps: ① predict using current memory; ② reflect against ground truth and generate proposals; ③ every N rounds, review the proposal pool and consolidate well-supported proposals into formal memory. Each proposal must also pass two self-review checks: a groundedness check (ensures proposals genuinely depend on historical data) and a generalizability check (ensures proposals are not specific to a single scene).

Your current task is **Self-Review · Groundedness Check Step 1: Generate perturbation plans** — design strong perturbations of the behavioral history to test whether proposals truly depend on the historical data.

---

You are a behavioral data perturbation expert. Your task is to design strong perturbation plans for a given user behavioral history, to test whether experiential proposals genuinely depend on the historical data.

# Perturbation Principles

The goal is to substantially modify the **response** fields in the historical behaviors so the perturbed history conveys user preference signals that are clearly different from the original:
- **Perturbations must be aggressive**: minor adjustments are not enough; the changes must be substantial reversals or inversions that produce a clearly different user profile.
- **Perturbations must cover many behaviors**: do not change only one record; each perturbation should cover all or nearly all records in the history to produce a systematic preference signal change.
- **Perturbations must have a clear direction**, for example:
  - Direction A: change all positive/active responses (clicks, purchases, likes, conversions) to negative/passive ones (skipped, ignored, viewed without interaction)
  - Direction B: change all high-intensity interactions (multiple likes, saves, comments) to zero interactions (impressions only, no clicks)
  - Direction C: change all purchase/conversion behaviors to abandonment, and change all likes to reports/"not interested"

# Perturbation Strength Self-Check
After designing each perturbation plan, ask yourself:
- If someone only saw the perturbed history, would their judgment of this user's preferences differ significantly from the original?
- If the answer is "not much difference," the perturbation is not strong enough and needs to be intensified.

# Original Behavioral History

{history_text}

---

# Output Requirements

Design {num_perturbations} strong perturbation plans. Each plan only needs to describe the perturbation approach; no need to output the full perturbed sequence. Output in JSON format as follows:

```json
[
  {{
    "perturbation_desc": "(describe the direction and theme of this perturbation)",
    "perturbation_details": "(using the specific content of the original sequence, explain which types of behavioral responses will be changed and how, with before/after examples)"
  }},
  ...
]```
"""





prompt_for_check_perturbation = """# Background
We are predicting behavior from a user's long history of behavioral sequences. The user has hundreds of historical behavior records, each containing a scene description and user response. The ultimate goal is to predict the user's behavior in new scenes.

To improve prediction accuracy, we use a Self-Practice method: construct simulated practice questions with ground-truth answers from the user's historical sequences, and let the model practice, reflect, and generalize repeatedly to accumulate two types of experiential memory before the final prediction:
- Pattern Experience: generalizations about this user's preferences and behavioral habits
- Calibration Experience: error tendencies and things to watch out for when the model reasons about this user's behavior

Each practice round has three steps: ① predict using current memory; ② reflect against ground truth and generate proposals; ③ every N rounds, review the proposal pool and consolidate well-supported proposals into formal memory. Each proposal must also pass two self-review checks: a groundedness check (ensures proposals genuinely depend on historical data) and a generalizability check (ensures proposals are not specific to a single scene).

Your current task is **Self-Review · Groundedness Check Step 2: Judge sensitivity** — based on the perturbation plan, infer whether each proposal would still hold if the history were perturbed, thereby filtering out model-fabricated proposals.

---

You are a proposal quality reviewer. Your task is to determine whether a batch of experiential proposals genuinely depend on the user's historical behavioral data, rather than being fabricated by the model.

# Review Logic

We have designed a purposeful perturbation plan for the user's behavioral history (not actually executed — only described). You must combine the original history and the perturbation plan to infer whether each proposal would still hold if the perturbation were applied:
- If the proposal **would no longer hold** after the perturbation, it genuinely depended on the specific content of the historical data — it is a grounded conclusion.
- If the proposal **would still hold** after the perturbation, it is unrelated to the specific content of the historical data — it is a model-fabricated conclusion.

# Review Materials

## Original Behavioral History
{history_text}

## Perturbation Plan
**Direction**: {perturbation_desc}
**Details**: {perturbation_details}

## All proposals to review
{original_proposals_text}

---

# Review Steps

## Step 1: Understand the perturbation effect
Combining the original history and the perturbation plan, construct a mental model of the perturbed user profile: what clearly different preference characteristics would the perturbed user exhibit?

## Step 2: Review each proposal
For each proposal, determine whether its conclusion is derived from specific behavioral features in the original sequence, or is a generic assertion unrelated to the historical data content. If the perturbed user profile contradicts the pattern described by the proposal, the proposal is sensitive (grounded). If the perturbed user profile remains compatible with the proposal, the proposal is insensitive (fabricated).

## Step 3: Output conclusions
Output a conclusion for each proposal in strictly the following format:

```json
[
  {{"proposal_idx": 0, "is_sensitive": true, "reason": "(brief explanation: which specific features in the original sequence support this proposal, and why the proposal no longer holds after those features disappear)"}},
  {{"proposal_idx": 1, "is_sensitive": false, "reason": "(brief explanation: why this proposal is unrelated to the specific content of the historical data and still holds after perturbation)"}},
  ...
]```

Note: proposal_idx corresponds one-to-one with the proposal numbering above, starting from 0."""






prompt_for_generate_virtual_scenes = """# Background
We are predicting behavior from a user's long history of behavioral sequences. The user has hundreds of historical behavior records, each containing a scene description and user response. The ultimate goal is to predict the user's behavior in new scenes.

To improve prediction accuracy, we use a Self-Practice method: construct simulated practice questions with ground-truth answers from the user's historical sequences, and let the model practice, reflect, and generalize repeatedly to accumulate two types of experiential memory before the final prediction:
- Pattern Experience: generalizations about this user's preferences and behavioral habits
- Calibration Experience: error tendencies and things to watch out for when the model reasons about this user's behavior

Each practice round has three steps: ① predict using current memory; ② reflect against ground truth and generate proposals; ③ every N rounds, review the proposal pool and consolidate well-supported proposals into formal memory. Each proposal must also pass two self-review checks: a groundedness check and a generalizability check (ensures proposals are not specific to a single scene).

Your current task is **Self-Review · Generalizability Check Step 1: Generate virtual scenes** — generate similar virtual scenes based on the real scene, to test whether proposals are overly dependent on a specific scene.

---

You are a scene generation expert. Your task is to generate several similar but distinct virtual scenes based on a real prediction scene, to test the generalizability of experiential proposals.

# Generation Principles

- Virtual scenes should belong to the same broad category as the real scene (e.g., both are e-commerce shopping scenes, both are ad recommendation scenes, etc.)
- Virtual scenes should identify the key entities in the real scene (e.g., product name, category, industry, prediction question) and replace them with similar but different entities
- The replaced virtual scenes should maintain the same structure and format as the real scene
- Virtual scenes should be sufficiently different from each other
- Do not modify the fixed format parts of the scene (e.g., "Scene details are as follows:", "Will this user add this item to their cart?" etc.)

# Real Scene

{real_scene_text}

---

# Output Requirements

Generate {num_virtual_scenes} virtual scenes, in JSON format as follows:

```json
[
  "(complete text of virtual scene 1, with exactly the same format as the real scene)",
  "(complete text of virtual scene 2)",
  ...
]```

Note: each virtual scene must be a complete scene text, including scene details and the prediction question."""





prompt_for_check_generalization = """# Background
We are predicting behavior from a user's long history of behavioral sequences. The user has hundreds of historical behavior records, each containing a scene description and user response. The ultimate goal is to predict the user's behavior in new scenes.

To improve prediction accuracy, we use a Self-Practice method: construct simulated practice questions with ground-truth answers from the user's historical sequences, and let the model practice, reflect, and generalize repeatedly to accumulate two types of experiential memory before the final prediction:
- Pattern Experience: generalizations about this user's preferences and behavioral habits
- Calibration Experience: error tendencies and things to watch out for when the model reasons about this user's behavior

Each practice round has three steps: ① predict using current memory; ② reflect against ground truth and generate proposals; ③ every N rounds, review the proposal pool and consolidate well-supported proposals into formal memory. Each proposal must also pass two self-review checks: a groundedness check and a generalizability check (ensures proposals are not specific to a single scene).

Your current task is **Self-Review · Generalizability Check Step 2: Judge generalizability** — determine whether each proposal is too specific and only applies to the current scene, and filter out proposals with insufficient generalizability.

---

You are a proposal generalizability reviewer. Your task is to determine whether a batch of experiential proposals are sufficiently general.

# Review Logic

A good experiential proposal should be generalizable — applicable to a variety of similar scenes, not just one specific scene.
- If a proposal's description is too specific and tightly bound to a particular scene, the proposal can precisely identify its corresponding real scene.
- If a proposal is well-generalized and describes a general pattern, the proposal content alone cannot determine which specific scene it corresponds to.

# Review Materials

## Candidate scenes (A is the real scene; the rest are similar virtual scenes)
{options_text}

## All proposals to review
{proposals_text}

---

# Review Steps

## Step 1: Understand the differences between candidate scenes
Identify the key distinguishing entities or features in each candidate scene.

## Step 2: Review each proposal
For each proposal, determine whether its content alone can precisely identify real scene A (i.e., whether the proposal is too specific).
- If it can precisely identify A, the proposal has poor generalizability and does not pass.
- If it cannot precisely identify A, the proposal has good generalizability and passes.

## Step 3: Output conclusions
Output a conclusion for each proposal in strictly the following format:

```json
[
  {{"proposal_idx": 0, "generalization_passed": true, "reason": "(brief explanation)"}},
  {{"proposal_idx": 1, "generalization_passed": false, "reason": "(brief explanation)"}},
  ...
]```

Note: proposal_idx corresponds one-to-one with the proposal numbering above, starting from 0. The real scene is fixed as option A; the model does not need to guess which one it is."""


