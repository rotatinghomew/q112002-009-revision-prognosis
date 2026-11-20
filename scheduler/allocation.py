"""会谈席位分配：同一时刻成立的三件事 + 人工甄别名额 + 承诺保护。

分配一个新席位必须**同时**满足：

1. 计划仍可干预：在 ``start..deadline`` 内且当周未见终局事件；
2. 学生在目标周有可接收时间，且名单中存在能力方向匹配的导师；
3. 预测依据足够：模型适用性通过且学生个体历史达标。

证据不足者不是低风险，他们进入**人工甄别队列**，只使用容量里预留的
``manual`` 席位；算法席位（``seats - manual``）不得占用人工名额，反之
人工名额也不会被算法建议填满。

承诺保护（与业务约定一致）：

* 已接受/已邀请/已出席的承诺先留着，新分配只能使用剩余席位；
* 预测变化但没有计划状态证据，不能撤掉已接受席位；
* 只有当周可见的新状态确认完成或退出时，才把该承诺标记为
  “待导师核实”，核实取消或转谈后才由调用方释放席位。
"""

from dataclasses import dataclass, field

from .labels import COMPLETED, SWITCHED
from .records import COMMITTED_STATES
from .forecast import forecast_week
from .features import MIN_STUDENT_HISTORY


@dataclass
class Commitment:
    plan_id: str
    student: str
    mentor: str
    week: int
    state: str
    known_week: int


@dataclass
class Allocation:
    target_week: int
    as_of_week: int
    invited: list = field(default_factory=list)        # 算法新邀请
    manual_review: list = field(default_factory=list)  # 已占人工名额
    manual_backlog: list = field(default_factory=list)  # 待人工甄别但名额/导师不足
    held_commitments: list = field(default_factory=list)
    release_requests: list = field(default_factory=list)
    unschedulable: list = field(default_factory=list)
    capacity_snapshot: list = field(default_factory=list)
    rule_trace: list = field(default_factory=list)     # 每条决策的依据

    def to_dict(self):
        return {
            "target_week": self.target_week,
            "as_of_week": self.as_of_week,
            "capacity_snapshot": self.capacity_snapshot,
            "held_commitments": [c.__dict__ for c in self.held_commitments],
            "release_requests": self.release_requests,
            "invited": self.invited,
            "manual_review": self.manual_review,
            "manual_backlog": self.manual_backlog,
            "unschedulable": self.unschedulable,
            "rule_trace": self.rule_trace,
        }


def _visible_commitments(store, target_week, as_of_week):
    """目标周当周截点已存在、且仍占席位的承诺。"""
    commitments = []
    for row in store.visible_attendance(as_of_week):
        if row["week"] != target_week:
            continue
        commitments.append(
            Commitment(
                plan_id=row["plan"],
                student=row["student"],
                mentor=row["mentor"],
                week=row["week"],
                state=row["state"],
                known_week=row["known_week"],
            )
        )
    return commitments


def _prior_contact_students(store, as_of_week, target_week):
    """截至当周已可见、且针对目标周或更早周次发出过邀请的学生。

    同人重复邀约规则：同一学生在同一目标周已有任何回执（含 declined），
    不再重复发邀请；更早周次的历史会谈不阻止新一周的邀请（每周独立）。
    """
    return {
        row["student"]
        for row in store.visible_attendance(as_of_week)
        if row["week"] == target_week
    }


def allocate_week(store, model, target_week, as_of_week, term=None):
    """生成目标周的完整排期决策（纯函数，不写回登记表）。"""
    result = Allocation(target_week=target_week, as_of_week=as_of_week)
    capacity = store.capacity_for_week(target_week, as_of_week)
    if not capacity:
        result.rule_trace.append(
            f"目标周 {target_week} 在截点 {as_of_week} 无任何容量快照，"
            "不产生邀请（不能用后来录入的资源）"
        )
        return result

    result.capacity_snapshot = [
        {
            "mentor": m,
            "focus": store.mentor_focus[m],
            "seats": row.seats,
            "manual": row.manual,
            "algorithm_seats": row.seats - row.manual,
            "recorded": row.recorded,
        }
        for m, row in sorted(capacity.items())
    ]

    # 1) 重放承诺，先占席位；并按当周新状态生成“待核实释放”。
    commitments = _visible_commitments(store, target_week, as_of_week)
    already_contacted = _prior_contact_students(
        store, as_of_week, target_week
    )
    pending_release = set()
    for commitment in commitments:
        result.held_commitments.append(commitment)
        timeline = store.timeline(commitment.plan_id)
        terminal = timeline.known_terminal(as_of_week)
        if terminal is not None and terminal["kind"] in ("确认完成", "换目标"):
            # 有计划状态证据：请求导师核实，席位在核实前继续保留。
            pending_release.add(
                (commitment.plan_id, commitment.mentor, commitment.student)
            )
            result.release_requests.append(
                {
                    "plan": commitment.plan_id,
                    "student": commitment.student,
                    "mentor": commitment.mentor,
                    "state": commitment.state,
                    "evidence_event": terminal["kind"],
                    "event_week": terminal["event_week"],
                    "known_week": terminal["known_week"],
                    "action": "请导师核实取消或转谈；核实前席位不释放",
                }
            )

    # 2) 当周预测，筛出可干预计划并按学生去重（多个计划同人每周最多一谈）。
    forecasts = forecast_week(store, model, as_of_week, term=term)
    candidates = []
    for fc in forecasts:
        if not fc.active:
            continue
        plan = store.plan(fc.plan_id)
        if plan["student"] in already_contacted:
            result.unschedulable.append(
                {
                    "plan": fc.plan_id,
                    "student": fc.student,
                    "reason": "目标周已有会谈回执，按同人不重复邀约规则跳过",
                }
            )
            continue
        candidates.append(fc)

    # 同一学生同期多项计划：保留未完成风险最高（证据不足优先人工）的一项。
    by_student = {}
    for fc in candidates:
        keep = by_student.get(fc.student)
        if keep is None or _student_priority(fc) > _student_priority(keep):
            by_student[fc.student] = fc
    candidates = sorted(
        by_student.values(),
        key=lambda fc: _student_priority(fc),
        reverse=True,
    )

    # 3) 导师可行集：能力方向匹配 ∩ 学生当周可接收名单 ∩ 有剩余席位。
    #    每个承诺都先占住一个席位（含待核实者——导师确认前不释放）：
    #    优先占用算法席位，算法席位不足时才占用人工预留。
    algo_seats = {m: row.seats - row.manual for m, row in capacity.items()}
    manual_seats = {m: row.manual for m, row in capacity.items()}
    for commitment in commitments:
        mentor = commitment.mentor
        if algo_seats.get(mentor, 0) > 0:
            algo_seats[mentor] -= 1
        elif manual_seats.get(mentor, 0) > 0:
            manual_seats[mentor] -= 1

    for fc in candidates:
        plan = store.plan(fc.plan_id)
        acceptable = store.available_mentors(fc.student, target_week)
        eligible_mentors = [
            m
            for m in acceptable
            if m in capacity and store.mentor_focus[m] == plan["focus"]
        ]
        if not acceptable:
            reason = "学生目标周无可用时间登记，当周不可约谈"
        elif not eligible_mentors:
            reason = (
                f"可接收导师 {acceptable} 中无方向匹配 {plan['focus']!r} 者"
            )
        else:
            reason = None
        if reason is not None:
            result.unschedulable.append(
                {"plan": fc.plan_id, "student": fc.student, "reason": reason}
            )
            result.rule_trace.append(
                f"{fc.plan_id}/{fc.student}: {reason}"
            )
            continue

        if not fc.evidence_sufficient:
            # 先过与算法队列相同的可约谈门槛，再占人工名额；
            # 无可用时间/无匹配导师属于客观不可安排，不能算“已甄别”。
            if not acceptable:
                result.unschedulable.append(
                    {
                        "plan": fc.plan_id,
                        "student": fc.student,
                        "reason": "证据不足且学生目标周无可用时间登记",
                    }
                )
                continue
            if not eligible_mentors:
                result.unschedulable.append(
                    {
                        "plan": fc.plan_id,
                        "student": fc.student,
                        "reason": (
                            "证据不足且可接收导师 "
                            f"{acceptable} 中无方向匹配 {plan['focus']!r} 者"
                        ),
                    }
                )
                continue
            placement = _place_manual(
                fc, plan, eligible_mentors, manual_seats, store
            )
            if placement is not None:
                result.manual_review.append(placement)
            else:
                # 人工名额是总席位内的硬预留；满员后进积压队列，
                # 这些人仍然不是“低风险”，下周需优先复核。
                result.manual_backlog.append(
                    {
                        "plan": fc.plan_id,
                        "student": fc.student,
                        "eligible_mentors": eligible_mentors,
                        "reasons": fc.support_reasons
                        or [f"个体历史不足（阈值 {MIN_STUDENT_HISTORY}）"],
                        "reason": "人工甄别名额已满，保留为人工复核积压",
                    }
                )
            continue

        placement = _place_algorithm(
            fc, plan, eligible_mentors, algo_seats, store
        )
        if placement is not None:
            result.invited.append(placement)
        else:
            # 算法席位不足不自动挤占人工名额。
            result.unschedulable.append(
                {
                    "plan": fc.plan_id,
                    "student": fc.student,
                    "reason": "方向匹配导师的算法席位已用尽（人工名额保留给甄别）",
                }
            )

    result.rule_trace.append(
        f"分配结束：算法邀请 {len(result.invited)}，"
        f"人工甄别 {len(result.manual_review)}，"
        f"人工积压 {len(result.manual_backlog)}，"
        f"保留承诺 {len(result.held_commitments)}，"
        f"待核实释放 {len(result.release_requests)}，"
        f"不可安排 {len(result.unschedulable)}"
    )
    return result


def _student_priority(fc):
    """学生级排序：证据不足者先进人工队列；其余按未完成风险降序。"""
    if not fc.evidence_sufficient:
        return (1, fc.incomplete_risk or 1.0)
    return (0, fc.incomplete_risk or 0.0)


def _mentor_order(eligible_mentors, seats):
    """剩余席位多的导师优先，便于留出整段容量。"""
    return sorted(eligible_mentors, key=lambda m: (-seats.get(m, 0), m))


def _place_algorithm(fc, plan, eligible_mentors, seats, store):
    for mentor in _mentor_order(eligible_mentors, seats):
        if seats.get(mentor, 0) > 0:
            seats[mentor] -= 1
            return {
                "plan": fc.plan_id,
                "student": fc.student,
                "mentor": mentor,
                "focus": plan["focus"],
                "incomplete_risk": round(fc.incomplete_risk, 4),
                "basis": "证据充分；按 P(可确认未完成) 降序占用算法席位",
                "probabilities": {o: round(p, 4) for o, p in fc.probs.items()},
                "credible_intervals": {
                    o: [round(lo, 4), round(hi, 4)]
                    for o, (lo, hi) in fc.intervals.items()
                },
            }
    return None


def _place_manual(fc, plan, eligible_mentors, seats, store):
    for mentor in _mentor_order(eligible_mentors, seats):
        if seats.get(mentor, 0) > 0:
            seats[mentor] -= 1
            return {
                "plan": fc.plan_id,
                "student": fc.student,
                "mentor": mentor,
                "focus": plan["focus"],
                "reasons": fc.support_reasons
                or [f"个体历史不足（阈值 {MIN_STUDENT_HISTORY}）"],
                "basis": "证据不足，不作低风险搁置；占用人工甄别名额",
            }
    return None
