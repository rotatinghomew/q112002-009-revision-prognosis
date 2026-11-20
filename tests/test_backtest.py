"""回测可重现性：滚动评估、逐周容量重放与审计追溯。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scheduler.records import RecordStore
from scheduler.model import train_for_term
from scheduler.backtest import rolling_evaluation, replay_weeks
from scheduler.features import training_rows


class BacktestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = RecordStore.from_folder("records")
        cls.store.assert_contract()

    def test_rolling_evaluation_covers_terms_2_to_6(self):
        evaluations = rolling_evaluation(self.store)
        self.assertEqual([ev.term for ev in evaluations], [2, 3, 4, 5, 6])
        cutoff = self.store.current_cutoff()
        for ev in evaluations:
            self.assertLess(max(ev.training_terms), ev.term)
            # 训练集只含更早的期次；每项计划在周龄 1..4 各贡献一行。
            self.assertEqual(ev.n_training, 36 * (ev.term - 1) * 4)
            term_deadlines = [
                p["start"] + 4
                for p in self.store.plans
                if p["term"] == ev.term
            ]
            if max(term_deadlines) + 1 <= cutoff:
                # 观察期完全结束的期次：36 项 × 4 个周龄全部可评估。
                self.assertEqual(ev.metrics["n"], 36 * 4)
                self.assertEqual(
                    {m["n"] for m in ev.channel_metrics.values()},
                    {12 * 4, 24 * 4},
                )
            else:
                # 在途期次：只有截点前已到达的锚点周可入模，且不得补全。
                self.assertLess(ev.metrics["n"], 36 * 4)
                self.assertGreater(ev.metrics["n"], 0)

    def test_replay_uses_only_capacity_recorded_by_as_of(self):
        # lag=1（排下周）：第 42 周容量 recorded=42，在截点 41 不可见。
        replay = replay_weeks(self.store, [42], lag=1)
        self.assertEqual(replay[0]["allocation"]["capacity_snapshot"], [])
        # lag=0（当周初排当周）：截点 42 可见当周容量。
        replay0 = replay_weeks(self.store, [42], lag=0)
        self.assertEqual(
            len(replay0[0]["allocation"]["capacity_snapshot"]), 3
        )

    def test_replay_reproduces_commitments(self):
        replay = replay_weeks(self.store, [42], lag=0)[0]["allocation"]
        held = {(c["student"], c["state"]) for c in replay["held_commitments"]}
        self.assertEqual(
            held,
            {("s3", "accepted"), ("s6", "accepted"), ("s7", "invited")},
        )
        # s7 的换目标在当周得知 → 待核实释放请求被重现。
        self.assertEqual(
            [r["student"] for r in replay["release_requests"]], ["s7"]
        )

    def test_replay_training_rows_grow_with_history(self):
        replay = replay_weeks(self.store, [9, 17, 25], lag=0)
        counts = [w["n_training_rows"] for w in replay]
        self.assertEqual(counts, sorted(counts))
        self.assertGreater(counts[-1], counts[0])

    def test_audit_payload_records_information_basis(self):
        replay = replay_weeks(self.store, [44], lag=0)[0]
        self.assertEqual(replay["as_of_week"], 44)
        self.assertIn("forecast_summary", replay)
        self.assertIn("risk_ranking", replay["forecast_summary"])
        allocation = replay["allocation"]
        for inv in allocation["invited"]:
            self.assertIn("incomplete_risk", inv)
            self.assertIn("credible_intervals", inv)

    def test_forecast_summary_four_state_counts(self):
        replay = replay_weeks(self.store, [44], lag=0)[0]
        states = replay["forecast_summary"]["states"]
        # 第 44 周截点：term6 计划里同时存在待观察、完成、换目标。
        self.assertIn("pending_observation", states)
        self.assertIn("completed", states)
        self.assertIn("switched", states)


if __name__ == "__main__":
    unittest.main()
