import unittest

from terrain_baker.errors import MemoryBudgetExceeded
from terrain_baker.memory import MemoryBudget


class MemoryBudgetTests(unittest.TestCase):
    def test_acquire_release_and_peak(self):
        budget = MemoryBudget(100)
        budget.acquire(40)
        budget.acquire(30)
        self.assertEqual(budget.current, 70)
        self.assertEqual(budget.peak, 70)
        budget.release(30)
        self.assertEqual(budget.current, 40)
        self.assertEqual(budget.peak, 70)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["limit_bytes"], 100)
        self.assertEqual(snapshot["peak_live_bytes"], 70)

    def test_over_budget_raises_and_keeps_state(self):
        budget = MemoryBudget(64)
        budget.acquire(40)
        with self.assertRaises(MemoryBudgetExceeded):
            budget.acquire(30)
        self.assertEqual(budget.current, 40)

    def test_release_clamps_at_zero(self):
        budget = MemoryBudget(10)
        budget.acquire(5)
        budget.release(9)
        self.assertEqual(budget.current, 0)

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            MemoryBudget(0)


if __name__ == "__main__":
    unittest.main()
