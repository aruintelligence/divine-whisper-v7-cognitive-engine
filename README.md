> [!NOTE]
> **Divine Whisper lineage repository.** This is an experimental stage. See the [ecosystem index](https://github.com/aruintelligence/divine-whisper-ecosystem) for the complete version map and research scope.

# Divine Whisper v7: contextual routing experiment

The original extensionless v7 script failed at import because several names were undefined. Its "self-optimization" gave all five agents the same random success or failure on each task. That feedback cannot identify which agent performed better. This repository now has a runnable, dependency-free Python experiment in `divine_whisper_v7_cognitive_engine.py`.

Five named **scripted** agents have fixed, different success probabilities across five generated task categories. Each episode samples a category and a potential binary outcome for each agent. The router chooses one agent and observes **only that agent's outcome**. No language model, real expert, safety reviewer, or external task is involved.

Three policies see the **same precomputed episodes** for each seed:

| Policy | Decision rule |
| --- | --- |
| Uniform | Choose a random agent, ignoring feedback |
| Global reputation | UCB on successes per agent across all categories |
| Contextual reputation | UCB on successes per agent within the current category |

The UCB routers try each agent once per applicable context, then use empirical success rate plus `0.6 × sqrt(log(total + 1) / agent_tries)`. Each policy learns only from its own selected outcomes. The benchmark reports realized successes and expected pseudo-regret, defined as the sum of `best scripted success probability − chosen agent's scripted probability` on each episode. It also exports per-seed outcomes, routing counts, and sampled learning curves.

## Run

```bash
python divine_whisper_v7_cognitive_engine.py --seed 42 --episodes 500 --replicates 16 --out-dir artifacts
python -m unittest discover -s tests -v
```

The command writes `artifacts/routing_comparison.json`; identical arguments produce the same file. At the default setting, mean realized successes per 500 episodes are **299.31** uniform, **295.38** global reputation, and **375.44** contextual reputation. Mean expected pseudo-regret is **123.67**, **123.73**, and **49.23** respectively. Global reputation does not improve on uniform in this setting; category-specific feedback does.

The skill table is invented and visible in the source and JSON. This result tests whether a router can exploit **known, synthetic category structure** under partial feedback. It does not demonstrate agent intelligence, self-improvement on real tasks, consensus, generalization, or a breakthrough. Different skill tables, exploration settings, and longer runs can change the ranking.

Original Divine Whisper concept: Daniel Jacob Read IV and Shane Travis Horman. MIT licensed.

---

### Inward Physics research navigation

[Complete Inward Physics index](https://github.com/aruintelligence/aruintelligence/blob/main/INWARD_PHYSICS.md) · [Daniel Jacob Read IV’s ORCID](https://orcid.org/0009-0000-6133-1872) · [Zenodo DOI series (record index)](https://github.com/aruintelligence/aruintelligence/blob/main/INWARD_PHYSICS.md#zenodo-record-index)

These links index project-authored proposals and simulations; they do not imply independent validation of a physical model.
