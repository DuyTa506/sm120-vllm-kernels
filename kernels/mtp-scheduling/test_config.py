import unittest
from mtp_schedule_config import config


class ConfigTests(unittest.TestCase):
    def test_only_schedule_differs(self):
        fixed, dynamic = config("fixed2"), config("batch21")
        schedule = dynamic.pop("num_speculative_tokens_per_batch_size")
        self.assertEqual(fixed, dynamic)
        self.assertEqual(schedule, [[1, 8, 2], [9, 64, 1]])
        self.assertEqual(fixed["num_speculative_tokens"], 2)

    def test_reject_ambiguous_or_unsupported_profile(self):
        for value in ("nospec", "depth0", "scalar", ""):
            with self.assertRaises(ValueError):
                config(value)
