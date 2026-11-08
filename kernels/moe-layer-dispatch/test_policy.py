import unittest
from policy import choose, select, cache_identity


class PolicyTests(unittest.TestCase):
    def test_requires_two_orders_and_accuracy(self):
        row = dict(valid=True, passes=[dict(cutlass_us=90, scalar_us=100)] * 2)
        self.assertEqual(choose(row), "cutlass")
        row["passes"][1] = dict(cutlass_us=99, scalar_us=100)
        self.assertEqual(choose(row), "scalar")
        row["passes"] = [dict(cutlass_us=80, scalar_us=100)]
        self.assertEqual(choose(row), "scalar")
        row.update(valid=False, passes=row["passes"] * 2)
        self.assertEqual(choose(row), "scalar")

    def test_guard_fallback_and_cache_identity(self):
        policy = dict(source_sha256="abc", entries={"19:1024": "cutlass"})
        self.assertEqual(select(policy, 19, 1024, "abc"), "cutlass")
        self.assertEqual(select(policy, 19, 1023, "abc"), "existing")
        self.assertEqual(select(policy, 0, 1024, "abc"), "scalar")
        self.assertEqual(select(policy, 19, 1025, "abc"), "scalar")
        with self.assertRaises(ValueError):
            select(policy, 19, 1024, "changed")
        identity = cache_identity(policy)
        policy["entries"]["0:2048"] = "cutlass"
        self.assertNotEqual(identity, cache_identity(policy))


if __name__ == "__main__":
    unittest.main()
