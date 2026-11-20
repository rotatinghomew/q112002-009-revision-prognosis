"""登记表的只读存储与契约校验，所有取数入口都强制“当周可见”。

设计原则：原始表原样保留，不在加载时合并业务记录；任何按周视图都必须
显式传入截点周，由本模块统一按 ``known_week`` / ``recorded`` 过滤，避免
调用方在特征、模型、回测里各自实现一遍而引入泄漏。
"""

from dataclasses import dataclass
from pathlib import Path
import json

from .labels import PlanTimeline, deadline_week

EXPECTED_CATALOG = {
    "plans": ["plan", "student", "term", "start", "channel", "focus", "course"],
    "events": ["plan", "kind", "event_week", "known_week", "note"],
    "mentors": ["mentor", "focus"],
    "capacity": ["week", "mentor", "seats", "manual", "recorded"],
    "availability": ["student", "week", "mentors"],
    "attendance": ["plan", "student", "mentor", "week", "state", "known_week"],
}

# 会谈回执的业务状态（数据里另有英文枚举，这里保留原文）。
ATTENDANCE_STATES = {"invited", "accepted", "attended", "declined"}
# 已经占住导师席位、不能被新邀请顶替的承诺状态。
COMMITTED_STATES = {"invited", "accepted", "attended"}


def load_tables(folder):
    """读取并校验一个资料目录（与 record_reader 的格式契约一致）。"""
    folder = Path(folder)
    catalog = json.loads((folder / "catalog.json").read_text(encoding="utf-8"))
    if catalog != EXPECTED_CATALOG:
        raise ValueError("catalog.json 字段契约与本系统内置契约不一致")
    tables = {}
    for table, columns in catalog.items():
        rows = json.loads((folder / f"{table}.json").read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{table} 不是行数组")
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict) or set(row) != set(columns):
                raise ValueError(f"{table} 第 {index} 行字段与目录不符")
        tables[table] = rows
    return tables


@dataclass(frozen=True)
class CapacityRow:
    week: int
    mentor: str
    seats: int
    manual: int
    recorded: int


class RecordStore:
    """带按周可见性过滤的登记表访问层。"""

    def __init__(self, tables):
        self.tables = tables
        self.plans = tables["plans"]
        self._plan_by_id = {p["plan"]: p for p in self.plans}
        self._timelines = {
            p["plan"]: PlanTimeline(p, tables["events"]) for p in self.plans
        }
        self.mentor_focus = {m["mentor"]: m["focus"] for m in tables["mentors"]}
        self._capacity = [
            CapacityRow(**{k: row[k] for k in CapacityRow.__dataclass_fields__})
            for row in tables["capacity"]
        ]
        self._availability = tables["availability"]
        self._attendance = tables["attendance"]

    @classmethod
    def from_folder(cls, folder):
        return cls(load_tables(folder))

    # ---- 计划与时间线 -------------------------------------------------
    def plan(self, plan_id):
        return self._plan_by_id[plan_id]

    def timeline(self, plan_id):
        return self._timelines[plan_id]

    def plans_in_term(self, term):
        return [p for p in self.plans if p["term"] == term]

    # ---- 按周可见的事件 / 回执 ---------------------------------------
    def visible_attendance(self, week):
        """当周截点看得见的全部会谈回执（按得知周过滤）。"""
        return [a for a in self._attendance if a["known_week"] <= week]

    def attendance_for_student(self, student, week):
        """该学生截至当周可见的全部回执（跨计划、跨周）。"""
        return [
            a
            for a in self._attendance
            if a["student"] == student and a["known_week"] <= week
        ]

    # ---- 容量快照（回测必须重放当周快照） ----------------------------
    def known_capacity_rows(self, as_of_week):
        """截点周已知的全部容量登记（``recorded <= as_of_week``）。

        后来才录入的席位不会出现，回测无法拿新增资源冒充早就可用。
        同一目标周、同一导师若有多条快照，保留录入周最新的一条。
        """
        latest = {}
        for row in self._capacity:
            if row.recorded > as_of_week:
                continue
            key = (row.week, row.mentor)
            if key not in latest or row.recorded > latest[key].recorded:
                latest[key] = row
        return list(latest.values())

    def capacity_for_week(self, target_week, as_of_week):
        """目标排期周在截点 ``as_of_week`` 已知的导师容量（按导师索引）。"""
        return {
            row.mentor: row
            for row in self.known_capacity_rows(as_of_week)
            if row.week == target_week
        }

    # ---- 学生可用时间 -------------------------------------------------
    def available_mentors(self, student, week):
        """学生在目标周可接收的导师名单（无登记即视为当周不可约谈）。"""
        for row in self._availability:
            if row["student"] == student and row["week"] == week:
                return list(row["mentors"])
        return []

    # ---- 全集工具 -----------------------------------------------------
    def all_weeks_with_capacity(self):
        return sorted({row.week for row in self._capacity})

    def current_cutoff(self):
        """资料中事件登记的最大得知周，作为“今天”的默认截点。"""
        known = [e["known_week"] for e in self.tables["events"]]
        known += [a["known_week"] for a in self._attendance]
        return max(known)

    def assert_contract(self):
        """超出字段层面的业务引用与取值校验，测试与 CLI 启动时调用。"""
        plan_ids = set(self._plan_by_id)
        students = {p["student"] for p in self.plans}
        for event in self.tables["events"]:
            if event["plan"] not in plan_ids:
                raise ValueError(f"events 引用了未知计划 {event['plan']}")
            if event["known_week"] < event["event_week"]:
                raise ValueError(f"事件得知周早于发生周: {event}")
        for row in self._attendance:
            if row["plan"] not in plan_ids:
                raise ValueError(f"attendance 引用了未知计划 {row['plan']}")
            if row["state"] not in ATTENDANCE_STATES:
                raise ValueError(f"未知会谈状态 {row['state']}")
            if row["mentor"] not in self.mentor_focus:
                raise ValueError(f"attendance 引用了未知导师 {row['mentor']}")
        for row in self._capacity:
            if row.mentor not in self.mentor_focus:
                raise ValueError(f"capacity 引用了未知导师 {row['mentor']}")
            if not 0 <= row.manual <= row.seats:
                raise ValueError(f"人工名额超出总席位: {row}")
        for row in self._availability:
            if row["student"] not in students:
                raise ValueError(f"availability 引用了未知学生 {row['student']}")
            unknown = set(row["mentors"]) - set(self.mentor_focus)
            if unknown:
                raise ValueError(f"availability 含未知导师 {unknown}")
        return True

    def deadline_week(self, plan_id):
        return deadline_week(self._plan_by_id[plan_id])
