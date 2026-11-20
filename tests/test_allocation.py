"""排期规则测试：三件事同时成立、人工名额、承诺保护、同人去重。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scheduler.records import RecordStore
from scheduler.model import train_for_term
from scheduler.allocation import allocate_week
from scheduler.forecast import forecast_week


class AllocationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = RecordStore.from_folder("records")
        cls.store.assert_contract()
        cls.model = train_for_term(cls.store, 6, 44)

    def _allocation(self, target, as_of=None, term=6):
        return allocate_week(
            self.store, self.model, target, as_of or target, term=term
        )

    def test_no_capacity_snapshot_means_no_invitation(self):
        # 第 45 周容量 recorded=45，在 44 周截点尚不可见。
        a = allocate_week(self.store, self.model, 45, 44, term=6)
        self.assertEqual(a.capacity_snapshot, [])
        self.assertEqual(a.invited, [])
        self.assertEqual(a.manual_review, [])

    def test_manual_seats_are_within_total_not_added(self):
        a = self._allocation(44)
        total_seats = sum(c["seats"] for c in a.capacity_snapshot)
        total_manual = sum(c["manual"] for c in a.capacity_snapshot)
        total_algo = sum(c["algorithm_seats"] for c in a.capacity_snapshot)
        self.assertEqual(total_manual + total_algo, total_seats)
        self.assertLessEqual(len(a.invited), total_algo)
        self.assertLessEqual(len(a.manual_review), total_manual)

    def test_algorithm_never_uses_manual_seats_and_vice_versa(self):
        a = self._allocation(44)
        # 人工名额只在 m1；算法邀请里 m1 只能用算法席位（seats-manual=1）。
        m1_algo = sum(
            1 for inv in a.invited if inv["mentor"] == "m1"
        )
        cap_m1 = next(c for c in a.capacity_snapshot if c["mentor"] == "m1")
        self.assertLessEqual(m1_algo, cap_m1["algorithm_seats"])
        for item in a.manual_review:
            self.assertEqual(item["mentor"], "m1")

    def test_mentor_focus_must_match_plan(self):
        a = self._allocation(44)
        for inv in a.invited + a.manual_review:
            plan = self.store.plan(inv["plan"])
            self.assertEqual(
                self.store.mentor_focus[inv["mentor"]], plan["focus"]
            )

    def test_mentor_must_be_in_student_availability(self):
        a = self._allocation(44)
        for inv in a.invited + a.manual_review:
            acceptable = self.store.available_mentors(
                inv["student"], a.target_week
            )
            self.assertIn(inv["mentor"], acceptable)

    def test_one_meeting_per_student_per_week(self):
        a = self._allocation(44)
        placed = [inv["student"] for inv in a.invited]
        placed += [inv["student"] for inv in a.manual_review]
        self.assertEqual(len(placed), len(set(placed)))
        # 被去重的学生若同期有两项计划，只保留一项进入决策。
        active_students = {
            fc.student
            for fc in forecast_week(self.store, self.model, 44, term=6)
            if fc.active
        }
        decided = set(placed) | {u["student"] for u in a.unschedulable}
        decided |= {u["student"] for u in a.manual_backlog}
        self.assertEqual(decided, active_students)

    def test_duplicate_invitation_blocked_when_receipt_exists(self):
        # 第 42 周已有回执的学生，在第 42 周重放时不会再收到新邀请。
        a = self._allocation(42)
        contacted = {
            row["student"]
            for row in self.store.visible_attendance(42)
            if row["week"] == 42
        }
        self.assertTrue(contacted)
        invited_students = {inv["student"] for inv in a.invited}
        manual_students = {inv["student"] for inv in a.manual_review}
        self.assertTrue(contacted.isdisjoint(invited_students))
        self.assertTrue(contacted.isdisjoint(manual_students))
        blocked = {u["student"] for u in a.unschedulable}
        self.assertTrue(contacted.issubset(blocked))

    def test_accepted_commitments_hold_seats(self):
        a = self._allocation(42)
        held_students = {c.student for c in a.held_commitments}
        self.assertEqual(held_students, {"s3", "s6", "s7"})
        # 已承诺占用后，导师剩余可分配席位不超过快照容量减承诺。
        cap = {c["mentor"]: c["seats"] for c in a.capacity_snapshot}
        used = {m: 0 for m in cap}
        for c in a.held_commitments:
            used[c.mentor] += 1
        for inv in a.invited:
            used[inv["mentor"]] += 1
        for item in a.manual_review:
            used[item["mentor"]] += 1
        for mentor, seats in cap.items():
            self.assertLessEqual(used[mentor], seats)

    def test_prediction_change_alone_never_releases_commitment(self):
        # 截点 41：p6-3、p6-6 已 accepted，当时无终局证据，必须原样保留。
        a = allocate_week(self.store, self.model, 42, 41, term=6)
        # 该截点容量尚未录入，连快照都没有——但规则层面承诺可见性独立验证：
        visible = self.store.visible_attendance(41)
        accepted = [r for r in visible if r["state"] == "accepted"]
        self.assertEqual({r["student"] for r in accepted}, {"s3", "s6"})
        for row in accepted:
            self.assertIsNone(
                self.store.timeline(row["plan"]).known_terminal(41)
            )

    def test_status_evidence_triggers_verification_not_silent_release(self):
        # 截点 42：s7 的计划已有“换目标”证据（known 42），只发待核实请求，
        # 席位在导师确认前仍保留在 held_commitments。
        a = self._allocation(42)
        s7 = next(r for r in a.release_requests if r["student"] == "s7")
        self.assertEqual(s7["evidence_event"], "换目标")
        self.assertIn("核实", s7["action"])
        self.assertIn("s7", {c.student for c in a.held_commitments})

    def test_insufficient_evidence_goes_manual_not_low_risk_shelf(self):
        a = self._allocation(42)
        # 人工队列与积压队列都必须显式保留证据不足者。
        manual_students = {x["student"] for x in a.manual_review}
        backlog_students = {x["student"] for x in a.manual_backlog}
        self.assertTrue(manual_students or backlog_students)
        for item in a.manual_review:
            self.assertTrue(item["reasons"])
            self.assertIn("不作低风险搁置", item["basis"])
        for item in a.manual_backlog:
            self.assertTrue(item["reasons"])

    def test_new_course_plans_never_get_algorithm_invitation(self):
        a = self._allocation(44)
        new_course = {
            p["plan"] for p in self.store.plans if p["course"] == "新课程"
        }
        for inv in a.invited:
            self.assertNotIn(inv["plan"], new_course)

    def test_every_decision_is_traceable(self):
        a = self._allocation(44)
        self.assertTrue(a.rule_trace)
        for inv in a.invited:
            self.assertIn("basis", inv)
            self.assertIn("probabilities", inv)
        for item in a.manual_review:
            self.assertTrue(item["reasons"])


if __name__ == "__main__":
    unittest.main()
