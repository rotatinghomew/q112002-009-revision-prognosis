"""业务时间与四态判定测试：补记事件不可提前可见，四态互斥。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scheduler.records import RecordStore
from scheduler.labels import (
    COMPLETED,
    SWITCHED,
    PENDING,
    INCOMPLETE,
    FORECAST_STATES,
    deadline_week,
)


class LabelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = RecordStore.from_folder("records")
        cls.store.assert_contract()

    def test_deadline_is_start_plus_four(self):
        for plan in self.store.plans:
            self.assertEqual(deadline_week(plan), plan["start"] + 4)

    def test_four_states_are_exhaustive_and_exclusive(self):
        for week in range(0, 49):
            for plan in self.store.plans:
                state = self.store.timeline(plan["plan"]).status_at(week)
                self.assertIn(state, FORECAST_STATES)

    def test_late_known_event_invisible_before_known_week(self):
        # 数据中存在得知周晚于发生周的补记事件。
        late = [
            e
            for e in self.store.tables["events"]
            if e["known_week"] > e["event_week"]
        ]
        self.assertTrue(late)
        for event in late[:20]:
            tl = self.store.timeline(event["plan"])
            before = tl.visible_events(event["known_week"] - 1)
            self.assertNotIn(event, before)
            after = tl.visible_events(event["known_week"])
            self.assertIn(event, after)

    def test_completion_and_switch_are_distinct_outcomes(self):
        # 换目标不能被压成“未完成”或“完成”。
        outcomes = {
            self.store.timeline(p["plan"]).realized_outcome()
            for p in self.store.plans
        }
        self.assertEqual(outcomes, set(FORECAST_STATES) - {PENDING})

    def test_switch_plan_tracks_separately(self):
        switched_plan = next(
            p
            for p in self.store.plans
            if self.store.timeline(p["plan"]).realized_outcome() == SWITCHED
        )
        tl = self.store.timeline(switched_plan["plan"])
        terminal = next(
            e for e in tl._events if e["kind"] == "换目标"
        )
        # 得知之前仍是待观察，得知当周转 switched，而不是 completed/incomplete。
        self.assertEqual(tl.status_at(terminal["known_week"] - 1), PENDING)
        self.assertEqual(tl.status_at(terminal["known_week"]), SWITCHED)
        self.assertFalse(tl.is_active(terminal["known_week"]))

    def test_incomplete_only_after_deadline_without_terminal(self):
        # 截止周含在观察期内：截止周当周仍为待观察，之后才可确认未完成。
        for plan in self.store.plans:
            tl = self.store.timeline(plan["plan"])
            if tl.realized_outcome() == INCOMPLETE:
                self.assertEqual(
                    tl.status_at(deadline_week(plan)), PENDING
                )
                self.assertEqual(
                    tl.status_at(deadline_week(plan) + 1), INCOMPLETE
                )
                self.assertIsNone(tl.outcome_known_by(deadline_week(plan)))
                self.assertEqual(
                    tl.outcome_known_by(deadline_week(plan) + 1), INCOMPLETE
                )

    def test_active_window(self):
        plan = self.store.plans[0]
        tl = self.store.timeline(plan["plan"])
        self.assertFalse(tl.is_active(plan["start"] - 1))
        if tl.realized_outcome() == INCOMPLETE:
            self.assertTrue(tl.is_active(deadline_week(plan)))
            self.assertFalse(tl.is_active(deadline_week(plan) + 1))


if __name__ == "__main__":
    unittest.main()
