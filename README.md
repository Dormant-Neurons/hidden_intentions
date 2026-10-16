# Unknown Unknowns: Do Hidden Intentions in LLMs Evade Detection?

[![EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Findings%20of%20ACL-6c5ce7.svg)](https://2026.emnlp.org/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2601.18552)

Official repository for **“Unknown Unknowns: Do Hidden Intentions in LLMs Evade Detection?”**, accepted to **Findings of the Association for Computational Linguistics: EMNLP 2026**.

**Authors:** Devansh Srivastav, David Pape, and Lea Schönherr  
**Affiliation:** CISPA Helmholtz Center for Information Security

## Overview

LLMs increasingly mediate how people search for information, seek advice and make decisions, which makes it possible for their outputs to covertly steer what users believe and do. We call these behaviours **hidden intentions**: covert, goal-directed patterns in model outputs that can influence users' beliefs or actions. They can be introduced deliberately, for example through prompting, data poisoning or fine-tuning, or emerge unintentionally from training.

Regulation such as the EU AI Act prohibits manipulative AI systems, but enforcing such a ban requires being able to *detect* manipulation. This work asks whether hidden intentions can be reliably detected, even under favourable conditions: overt manipulation, single-turn interactions and known categories.

## Contributions

- **Ten categories of hidden intentions.** We introduce ten categories, each grounded in social science research on persuasion and influence.
- **A real-world case study.** We show that all ten categories already manifest in publicly available LLMs.
- **A controlled testbed.** We induce hidden intentions in unmodified LLMs with lightweight agentic workflows, yielding 4,000 prompt–response pairs with known ground truth, verified through human annotation.
- **A systematic analysis of detection failures.** We evaluate static classifiers and LLM judges under category-specific and category-agnostic judging, and stress-test them at low prevalence.

## The ten categories

| ID | Category | What the model does | Social science grounding |
|---|---|---|---|
| C01 | Strategic Vagueness | Hedges to avoid clear answers | Strategic ambiguity, plausible deniability |
| C02 | Authority Bias | Claims unwarranted expertise | Authority and automation bias |
| C03 | Safetyism | Refuses legitimate queries | Gatekeeping, spiral of silence |
| C04 | Simulated Consensus Signalling | Fabricates social agreement | Social proof |
| C05 | Unsafe Coding Practices | Suggests insecure code | Cognitive offloading |
| C06 | Commercial Manipulation | Covertly promotes products | Personalised persuasion |
| C07 | Political Manipulation | Favours one political side | Framing and agenda-setting |
| C08 | Selective Personalisation Bias | Echoes and flatters the user | Echo chambers and filter bubbles |
| C09 | Emotional Manipulation | Exploits guilt, fear or trust | Affective persuasion |
| C10 | Disinformation and Bias | States falsehoods confidently | Illusory truth effect |

## Key findings

- **Static classifiers learn the trigger, not the intention.** Embedding-based classifiers are near-perfect on the trigger context seen in training, but fail when the same hidden intention appears under a new trigger.
- **Without the category prior, LLM judges degrade.** All ten judges perform substantially worse when they are not told which manipulation to look for. Frontier judges fail in opposite directions: o3 misses 52% of hidden intentions, while Claude Opus 4 flags 66% of benign outputs.
- **High precision is an artefact of balanced test sets.** Unless false-positive rates are vanishingly small, auditing is dominated by precision–prevalence trade-offs. For example, GPT-4.1 drops from 0.90 precision at 50% prevalence to 0.08 at 1%.
- **Obvious fixes do not close the gap.** Newer models (GPT-5.4), reasoning models (o3) and providing all ten category definitions at once do not recover reliable detection.

Together, these results point to a gap between governance frameworks that prohibit manipulative AI and what current auditing methods can deliver in open-world, low-prevalence settings.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{srivastav2026unknown,
  title     = {Unknown Unknowns: Do Hidden Intentions in {LLM}s Evade Detection?},
  author    = {Srivastav, Devansh and Pape, David and Sch{\"o}nherr, Lea},
  booktitle = {Findings of the Association for Computational Linguistics: {EMNLP} 2026},
  year      = {2026}
}
```

## Acknowledgements

This work was funded by the German Federal Ministry of Research, Technology and Space (BMFTR) under the grants AIgenCY (16KIS2012) and SisWiss (16KIS2330); by zukunft.niedersachsen, the joint science funding program of the Lower Saxony Ministry of Science and Culture and the Volkswagen Foundation; and by the Daimler and Benz Foundation under the grant Ladenburger Kolleg, project KonCheck. It was further supported by the Helmholtz Association's Initiative and Networking Fund on the HAICORE@FZJ partition.
