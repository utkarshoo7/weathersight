"""Unit tests for the current deterministic decision policy.

Run from the repository root with:
    python -m unittest discover -s tests -v
"""

import unittest

from code.pipeline.decision_policy import decide_action_mode


class DecisionPolicyTests(unittest.TestCase):
    def test_low_confidence_is_untrusted_caution(self):
        self.assertEqual(
            decide_action_mode("clear", 0.39, "medium"),
            {
                "mode": "caution",
                "trusted": False,
                "reason": "low_weather_confidence",
            },
        )

    def test_confidence_threshold_is_inclusive_at_point_four(self):
        result = decide_action_mode("clear", 0.40, "medium")
        self.assertTrue(result["trusted"])
        self.assertEqual(result["mode"], "normal")

    def test_low_lane_visibility_takes_precedence_over_weather(self):
        result = decide_action_mode("rainy", 0.95, "low")
        self.assertEqual(result["mode"], "slow")
        self.assertFalse(result["trusted"])
        self.assertEqual(result["reason"], "poor_lane_visibility")

    def test_unknown_lane_visibility_is_treated_as_medium(self):
        result = decide_action_mode("clear", 0.90, "unknown")
        self.assertEqual(result["mode"], "normal")
        self.assertTrue(result["trusted"])

    def test_none_lane_visibility_is_treated_as_medium(self):
        result = decide_action_mode("clear", 0.90, None)
        self.assertEqual(result["mode"], "normal")
        self.assertTrue(result["trusted"])

    def test_snowy_weather_returns_slow_and_trusted(self):
        result = decide_action_mode("snowy", 0.90, "high")
        self.assertEqual(result["mode"], "slow")
        self.assertTrue(result["trusted"])
        self.assertEqual(result["reason"], "snow_conditions")

    def test_rainy_weather_returns_normal_and_trusted(self):
        result = decide_action_mode("rainy", 0.90, "high")
        self.assertEqual(result["mode"], "normal")
        self.assertTrue(result["trusted"])
        self.assertEqual(result["reason"], "rain_conditions")

    def test_clear_weather_returns_normal_and_trusted(self):
        result = decide_action_mode("clear", 0.90, "high")
        self.assertEqual(result["mode"], "normal")
        self.assertTrue(result["trusted"])
        self.assertEqual(result["reason"], "clear_conditions")


if __name__ == "__main__":
    unittest.main()
