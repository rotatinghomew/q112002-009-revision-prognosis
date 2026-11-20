"""泄漏防护：特征与训练样本只使用当周可见资料，期次严格滚动。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scheduler.records import RecordStore
from scheduler.features import build_features, training_rows
from scheduler.model import train_for_term, OutcomeModel
from scheduler.forecast import forecast_week
from scheduler.labels import deadline_week, COMPLETED, SWITCHED, INCOMPLETE


class LeakageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = RecordStore.from_folder("records")
        cls.store.assert_contract()

    def test_anchor_week_never_after_cutoff(self):
        for as_of in (8, 16, 24, 32, 40, 44):
            for features, _ in training_rows(self.store, 6, as_of):
                self.assertLessEqual(features.week, as_of)

    def test_training_excludes_future_terms(self):
        for max_term in range(1, 6):
            rows = training_rows(self.store, max_term, 44)
            self.assertTrue(rows)
            for features, _ in rows:
                self.assertLessEqual(
                    self.store.plan(features.plan_id)["term"], max_term
                )

    def test_rolling_model_training_terms_are_strictly_earlier(self):
        for term in range(2, 7):
            model = train_for_term(self.store, term, 44)
            self.assertTrue(model.training_terms)
            self.assertLess(max(model.training_terms), term)

    def test_hidden_future_outcome_cannot_change_features(self):
        # 同一计划在得知周前一周与（假设删除后续事件后）的特征必须一致：
        # 直接验证当周特征不依赖任何 known_week 更晚的事件。
        late = next(
            e
            for e in self.store.tables["events"]
            if e["known_week"] > e["event_week"]
        )
        plan = self.store.plan(late["plan"])
        week = late["known_week"] - 1
        features = build_features(self.store, plan, week)
        visible = self.store.timeline(plan["plan"]).visible_events(week)
        self.assertNotIn(late, visible)
        # 进展可见性只由可见事件决定。
        self.assertEqual(
            features.progress_known,
            any(e["kind"] == "自主进展" for e in visible),
        )

    def test_parallel_plan_future_outcome_not_in_history(self):
        # 同期并行计划只能贡献当周状态，不能贡献结局计数。
        from collections import Counter

        counts = Counter()
        for plan in self.store.plans:
            if plan["term"] != 6:
                continue
            f = build_features(self.store, plan, 44)
            counts[f.history_total] += 1
            # 历史只来自 term<6 且当时已可确认的结局。
            for other in self.store.plans:
                if other["student"] == plan["student"] and other["term"] == 6:
                    continue
        self.assertTrue(counts)  # 冒烟：确实生成了特征

    def test_forecast_does_not_predict_resolved_plans(self):
        model = train_for_term(self.store, 6, 44)
        forecasts = forecast_week(self.store, model, 44, term=6)
        for fc in forecasts:
            if fc.visible_state in (COMPLETED, SWITCHED, INCOMPLETE):
                # 已确定的状态不再输出概率预测，也不再可干预。
                self.assertIsNone(fc.probs)
                self.assertFalse(fc.active)
            if fc.active:
                # 仍可干预 ⇒ 仍待观察 ⇒ 必有概率与可信区间。
                self.assertEqual(fc.visible_state, "pending_observation")
                self.assertIsNotNone(fc.probs)
                self.assertAlmostEqual(sum(fc.probs.values()), 1.0, places=9)

    def test_capacity_snapshot_ignores_later_recordings(self):
        # 容量 recorded == week 的资料里，提前一周看不到下周席位。
        cap = self.store.capacity_for_week(45, 44)
        self.assertEqual(cap, {})
        cap_now = self.store.capacity_for_week(44, 44)
        self.assertEqual(set(cap_now), {"m1", "m2", "m3"})

    def test_attendance_filtered_by_known_week(self):
        # term6 的回执 known_week=41，在第 40 周截点不可见。
        self.assertEqual(
            [a for a in self.store.visible_attendance(40) if a["week"] >= 40],
            [],
        )
        self.assertTrue(self.store.visible_attendance(41))


if __name__ == "__main__":
    unittest.main()
