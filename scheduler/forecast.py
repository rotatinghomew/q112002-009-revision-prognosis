"""当周预测：四种当周状态与截止结局概率的逐周快照。

对每个计划每周输出两类信息：

1. **当周状态**（确定性，由当时可见登记判定）：完成 / 换目标 /
   到截止时还没来得及观察 / 可确认的未完成——四者互斥、可逐周回看；
2. **截止结局概率**（仅对仍可干预的计划）：模型给出的三结局概率、
   95% 可信区间和适用性标记。已见终局或已过截止周的计划不再预测。
"""

from dataclasses import dataclass, asdict

from .labels import (
    COMPLETED,
    SWITCHED,
    PENDING,
    INCOMPLETE,
    FORECAST_STATES,
    deadline_week,
)
from .features import build_features, MIN_STUDENT_HISTORY


@dataclass
class PlanForecast:
    plan_id: str
    student: str
    term: int
    week: int
    deadline: int
    visible_state: str          # 四态之一
    active: bool                # 当周是否仍可干预
    probs: dict | None          # 三结局概率（仅 active 时）
    intervals: dict | None
    incomplete_risk: float | None   # P(可确认未完成)，预警主排序键
    supported: bool             # 模型建议是否受历史表现支持
    support_reasons: list
    stratum_samples: int
    evidence_sufficient: bool   # 个体历史 + 分箱证据是否足够
    reasons: list               # 判定依据（审计追溯用）

    def to_dict(self):
        return asdict(self)


def evidence_sufficient(features, model_prediction):
    """证据是否足够支撑自动化建议。

    两条任一不足即转人工甄别：
    * 模型侧：分箱/适用性未通过（新课程、样本不足）；
    * 个体侧：该学生可确认的历史计划少于 ``MIN_STUDENT_HISTORY``。
    注意“证据不足”不是“低风险”，排期模块会把他们保留进人工名额。
    """
    reasons = list(model_prediction["support"]["reasons"])
    if features.history_total < MIN_STUDENT_HISTORY:
        reasons.append(
            f"学生可确认历史计划 {features.history_total} "
            f"< 门槛 {MIN_STUDENT_HISTORY}"
        )
    return (len(reasons) == 0), reasons


def forecast_plan(store, model, plan, week):
    """单个计划的当周预测快照。"""
    timeline = store.timeline(plan["plan"])
    state = timeline.status_at(week)
    features = build_features(store, plan, week)
    deadline = deadline_week(plan)
    reasons = [
        f"截点周 {week}，计划开始周 {plan['start']}，截止周 {deadline}",
        f"当周可见状态={state}",
    ]

    probs = intervals = None
    risk = None
    supported = True
    support_reasons = []
    stratum_samples = 0
    sufficient = False

    if state in (COMPLETED, SWITCHED):
        terminal = timeline.known_terminal(week)
        reasons.append(
            f"已见终局事件 {terminal['kind']}（得知周 {terminal['known_week']}），不再预测"
        )
        sufficient = True  # 无需模型：状态已确定
    elif state == INCOMPLETE:
        reasons.append("截止周已过且未见终局事件，可确认未完成")
        sufficient = True
    else:  # PENDING：仍在观察/干预窗口内
        prediction = model.predict(features)
        probs = prediction["probs"]
        intervals = prediction["intervals"]
        risk = probs[INCOMPLETE]
        supported = prediction["supported"]
        support_reasons = prediction["support"]["reasons"]
        stratum_samples = prediction["support"]["stratum_samples"]
        sufficient, ev_reasons = evidence_sufficient(features, prediction)
        reasons.extend(ev_reasons)
        reasons.append(
            "分箱概率: "
            + ", ".join(f"{o}={probs[o]:.3f}" for o in probs)
        )
        if not supported:
            reasons.append("模型建议不受历史表现支持，须人工甄别")

    return PlanForecast(
        plan_id=plan["plan"],
        student=plan["student"],
        term=plan["term"],
        week=week,
        deadline=deadline,
        visible_state=state,
        active=timeline.is_active(week),
        probs=probs,
        intervals=intervals,
        incomplete_risk=risk,
        supported=supported,
        support_reasons=support_reasons,
        stratum_samples=stratum_samples,
        evidence_sufficient=sufficient,
        reasons=reasons,
    )


def forecast_week(store, model, week, term=None):
    """该周全部（或指定期次）计划的预测快照列表。"""
    plans = store.plans_in_term(term) if term is not None else store.plans
    return [forecast_plan(store, model, plan, week) for plan in plans]
