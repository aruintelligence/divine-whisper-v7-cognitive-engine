"""Seeded contextual routing experiment for the Divine Whisper v7 lineage.

All agents, categories, and success probabilities are scripted. The experiment
tests routing and partial-feedback learning on this toy environment only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


CATEGORIES = ("analysis", "research", "communication", "correction", "arbitration")
AGENTS = ("Michael", "Uriel", "Gabriel", "Raphael", "Raguel")
# Invented probabilities, fixed before any comparison. They are not evaluations
# of models, people, or the named roles.
SKILLS = {
    "Michael": {"analysis": .85, "research": .45, "communication": .50, "correction": .70, "arbitration": .55},
    "Uriel": {"analysis": .60, "research": .85, "communication": .45, "correction": .55, "arbitration": .50},
    "Gabriel": {"analysis": .50, "research": .55, "communication": .85, "correction": .45, "arbitration": .60},
    "Raphael": {"analysis": .65, "research": .50, "communication": .50, "correction": .85, "arbitration": .50},
    "Raguel": {"analysis": .55, "research": .50, "communication": .60, "correction": .55, "arbitration": .85},
}


@dataclass(frozen=True)
class Episode:
    category: str
    outcomes: dict[str, int]


def generate_episodes(seed: int, count: int) -> list[Episode]:
    """Draw category and all potential outcomes once for a paired replay."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    episodes = []
    for _ in range(count):
        category = CATEGORIES[rng.randrange(len(CATEGORIES))]
        outcomes = {agent: int(rng.random() < SKILLS[agent][category]) for agent in AGENTS}
        episodes.append(Episode(category, outcomes))
    return episodes


class UniformRouter:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def choose(self, category: str) -> str:
        return AGENTS[self.rng.randrange(len(AGENTS))]

    def observe(self, category: str, chosen: str, reward: int) -> None:
        pass


class ReputationRouter:
    """UCB routing with global feedback or separate feedback per category."""

    def __init__(self, contextual: bool, exploration: float = 0.6):
        if exploration < 0:
            raise ValueError("exploration must be nonnegative")
        self.contextual = contextual
        self.exploration = exploration
        self.counts = {key: {agent: 0 for agent in AGENTS} for key in self.keys}
        self.rewards = {key: {agent: 0 for agent in AGENTS} for key in self.keys}

    @property
    def keys(self) -> tuple[str, ...]:
        return CATEGORIES if self.contextual else ("all",)

    def key(self, category: str) -> str:
        if category not in CATEGORIES:
            raise ValueError("unknown category")
        return category if self.contextual else "all"

    def choose(self, category: str) -> str:
        key = self.key(category)
        # Explore each agent once in each context before using its observations.
        for agent in AGENTS:
            if self.counts[key][agent] == 0:
                return agent
        total = sum(self.counts[key].values())
        return max(
            AGENTS,
            key=lambda agent: self.rewards[key][agent] / self.counts[key][agent]
            + self.exploration * math.sqrt(math.log(total + 1) / self.counts[key][agent]),
        )

    def observe(self, category: str, chosen: str, reward: int) -> None:
        key = self.key(category)
        if chosen not in AGENTS or reward not in (0, 1):
            raise ValueError("invalid feedback")
        # Only the routed agent receives feedback; unchosen potential outcomes
        # remain hidden from the policy.
        self.counts[key][chosen] += 1
        self.rewards[key][chosen] += reward


def evaluate(router: UniformRouter | ReputationRouter, episodes: list[Episode]) -> dict:
    reward_total = 0
    expected_regret = 0.0
    selections = {category: {agent: 0 for agent in AGENTS} for category in CATEGORIES}
    reward_curve = [0]
    regret_curve = [0.0]
    for episode in episodes:
        chosen = router.choose(episode.category)
        reward = episode.outcomes[chosen]
        router.observe(episode.category, chosen, reward)
        selections[episode.category][chosen] += 1
        reward_total += reward
        best_expected = max(SKILLS[agent][episode.category] for agent in AGENTS)
        expected_regret += best_expected - SKILLS[chosen][episode.category]
        reward_curve.append(reward_total)
        regret_curve.append(expected_regret)
    return {
        "realized_rewards": reward_total,
        "expected_pseudo_regret": expected_regret,
        "selections": selections,
        "reward_curve": reward_curve,
        "regret_curve": regret_curve,
    }


def benchmark(seed: int = 42, episodes: int = 500, replicates: int = 16) -> dict:
    if episodes < 1 or replicates < 1:
        raise ValueError("episodes and replicates must be positive")
    methods = ("uniform", "global_reputation", "contextual_reputation")
    by_method = {name: [] for name in methods}
    sample_points = sorted(set([0, episodes] + list(range(50, episodes, 50))))

    for replicate in range(replicates):
        trial_seed = seed + replicate
        trials = generate_episodes(trial_seed, episodes)
        policies = {
            "uniform": UniformRouter(trial_seed ^ 0x9E3779B9),
            "global_reputation": ReputationRouter(contextual=False),
            "contextual_reputation": ReputationRouter(contextual=True),
        }
        for name, policy in policies.items():
            result = evaluate(policy, trials)
            by_method[name].append({
                "seed": trial_seed,
                "realized_rewards": result["realized_rewards"],
                "expected_pseudo_regret": result["expected_pseudo_regret"],
                "selections": result["selections"],
                "reward_at_sample": [result["reward_curve"][n] for n in sample_points],
                "regret_at_sample": [result["regret_curve"][n] for n in sample_points],
            })

    return {
        "scope": "synthetic contextual bandit with partial feedback and scripted agent skills",
        "seed": seed,
        "episodes_per_replicate": episodes,
        "replicates": replicates,
        "categories": list(CATEGORIES),
        "agents": list(AGENTS),
        "skill_probabilities": SKILLS,
        "sample_at_episode": sample_points,
        "methods": {
            name: {
                "mean_realized_rewards": sum(run["realized_rewards"] for run in by_method[name]) / replicates,
                "mean_expected_pseudo_regret": sum(run["expected_pseudo_regret"] for run in by_method[name]) / replicates,
                "mean_reward_at_sample": [
                    sum(run["reward_at_sample"][i] for run in by_method[name]) / replicates
                    for i in range(len(sample_points))
                ],
                "runs": by_method[name],
            }
            for name in methods
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.episodes < 1 or args.replicates < 1:
        parser.error("--episodes and --replicates must be positive")
    report = benchmark(args.seed, args.episodes, args.replicates)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "routing_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "episodes_per_replicate": args.episodes,
        "replicates": args.replicates,
        "mean_realized_rewards": {
            name: method["mean_realized_rewards"] for name, method in report["methods"].items()
        },
        "mean_expected_pseudo_regret": {
            name: round(method["mean_expected_pseudo_regret"], 3)
            for name, method in report["methods"].items()
        },
        "output_dir": str(args.out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
