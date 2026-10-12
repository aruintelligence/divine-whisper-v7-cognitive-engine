import json
import tempfile
import unittest
from pathlib import Path

from divine_whisper_v7_cognitive_engine import (
    AGENTS,
    CATEGORIES,
    ReputationRouter,
    UniformRouter,
    benchmark,
    evaluate,
    generate_episodes,
)


class RoutingTests(unittest.TestCase):
    def test_episode_stream_is_seeded_and_paired(self):
        episodes = generate_episodes(42, 100)
        self.assertEqual(episodes, generate_episodes(42, 100))
        self.assertNotEqual(episodes, generate_episodes(43, 100))
        one = evaluate(UniformRouter(1), episodes)
        two = evaluate(ReputationRouter(contextual=True), episodes)
        self.assertEqual(episodes, generate_episodes(42, 100))
        for result in (one, two):
            self.assertEqual(sum(sum(row.values()) for row in result["selections"].values()), 100)
            self.assertEqual(result["reward_curve"][-1], result["realized_rewards"])

    def test_only_selected_agent_receives_feedback(self):
        router = ReputationRouter(contextual=True, exploration=0.0)
        chosen = router.choose("research")
        router.observe("research", chosen, 1)
        self.assertEqual(sum(router.counts["research"].values()), 1)
        self.assertEqual(sum(router.rewards["research"].values()), 1)
        self.assertEqual(sum(router.counts["analysis"].values()), 0)
        self.assertEqual(router.choose("analysis"), AGENTS[0])

    def test_contextual_router_learns_distinct_specialists(self):
        router = ReputationRouter(contextual=True, exploration=0.0)
        for category, specialist in (("research", "Uriel"), ("communication", "Gabriel")):
            for _ in AGENTS:
                chosen = router.choose(category)
                router.observe(category, chosen, int(chosen == specialist))
        self.assertEqual(router.choose("research"), "Uriel")
        self.assertEqual(router.choose("communication"), "Gabriel")

    def test_default_comparison_is_reproducible_and_exposes_ablation(self):
        report = benchmark(42, 500, 16)
        self.assertEqual(report, benchmark(42, 500, 16))
        self.assertEqual(report["categories"], list(CATEGORIES))
        self.assertEqual(report["sample_at_episode"][-1], 500)
        methods = report["methods"]
        self.assertLess(
            methods["contextual_reputation"]["mean_expected_pseudo_regret"],
            methods["uniform"]["mean_expected_pseudo_regret"],
        )
        self.assertGreater(
            methods["contextual_reputation"]["mean_realized_rewards"],
            methods["global_reputation"]["mean_realized_rewards"],
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            target.write_text(json.dumps(report))
            self.assertEqual(json.loads(target.read_text()), report)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            generate_episodes(42, 0)
        with self.assertRaises(ValueError):
            benchmark(42, 10, 0)
        with self.assertRaises(ValueError):
            ReputationRouter(contextual=True, exploration=-1)


if __name__ == "__main__":
    unittest.main()
