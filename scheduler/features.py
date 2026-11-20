"""当周特征工程。

每个特征函数都只接收“截至截点周可见”的资料，调用方负责先按
``known_week`` 过滤（``RecordStore`` 已封装）。这里绝不直接读取事件的
发生周来推断当周是否已经发生——补记事件在得知之前对当周不存在。

特征分两层：

1. 计划层：报名渠道、能力方向、课程、相对观察周龄、是否已见自主进展；
2. 学生历史层：该学生在更早期次里、当周已可确认结局的计划表现。
   同期并行的另一项计划只贡献“当周可见状态”，不贡献它未来的结局。
"""

from dataclasses import dataclass, field

from .labels import (
    COMPLETED,
    SWITCHED,
    INCOMPLETE,
    PENDING,
    deadline_week,
)

# 建模使用的离散特征列（顺序即稳定的列顺序，便于审计与跨期对比）。
MODEL_FEATURES = ("channel", "focus", "course", "progress_known", "age_band")

AGE_BANDS = ("pre_start", "w1", "w2", "w3", "w4", "post_deadline")

# 历史层最小样本：低于该数的学生视为“个体证据不足”，进入人工甄别候选。
MIN_STUDENT_HISTORY = 3
# 模型分箱（渠道×方向×课程）最少带标签训练样本，否则该箱不给模型建议。
MIN_STRATUM_SAMPLES = 5


def age_band(plan, week):
    """相对开始周的观察周龄分箱。"""
    age = week - plan["start"]
    if age < 0:
        return "pre_start"
    if age >= 1 and age <= 4:
        return f"w{age}"
    return "post_deadline"


@dataclass
class PlanFeatures:
    plan_id: str
    student: str
    week: int
    channel: str
    focus: str
    course: str
    age_band: str
    progress_known: bool
    visible_state: str
    active: bool
    # 学生在更早期次、且当周已可确认结局的历史计划计数（按结局）。
    history_counts: dict = field(default_factory=dict)
    history_total: int = 0
    # 同期并行计划在当周的可见状态（不含本计划自身的未来信息）。
    parallel_states: tuple = ()

    def model_key(self):
        """模型分箱键：渠道×方向×课程×周龄×进展可见性。"""
        return (
            self.channel,
            self.focus,
            self.course,
            self.age_band,
            self.progress_known,
        )

    def support_key(self):
        """适用性键：新课程/新字段出现时，该键在训练期没有历史支撑。"""
        return (self.channel, self.focus, self.course)


def build_features(store, plan, week):
    """构造单个计划在 ``week`` 截点的特征（点内正确，无未来信息）。"""
    timeline = store.timeline(plan["plan"])
    visible_state = timeline.status_at(week)

    history_counts = {COMPLETED: 0, SWITCHED: 0, INCOMPLETE: 0}
    parallel = []
    for other in store.plans:
        if other["student"] != plan["student"] or other["plan"] == plan["plan"]:
            continue
        other_timeline = store.timeline(other["plan"])
        if other["term"] < plan["term"]:
            # 更早期次：只计入当周已可确认的结局，未确认的不进历史。
            outcome = other_timeline.outcome_known_by(week)
            if outcome is not None:
                history_counts[outcome] += 1
        elif other["term"] == plan["term"]:
            # 同期并行：只取当周可见状态，绝不读取其后续结局。
            parallel.append(other_timeline.status_at(week))

    return PlanFeatures(
        plan_id=plan["plan"],
        student=plan["student"],
        week=week,
        channel=plan["channel"],
        focus=plan["focus"],
        course=plan["course"],
        age_band=age_band(plan, week),
        progress_known=timeline.progress_known(week),
        visible_state=visible_state,
        active=timeline.is_active(week),
        history_counts=history_counts,
        history_total=sum(history_counts.values()),
        parallel_states=tuple(sorted(parallel)),
    )


def features_for_week(store, week, term=None):
    """该周全部（或指定期次）计划的当周特征。"""
    plans = store.plans_in_term(term) if term is not None else store.plans
    return [build_features(store, plan, week) for plan in plans]


def training_rows(store, max_term, as_of_week):
    """生成纵向训练样本：截至 ``as_of_week`` 结局已可确认的、``max_term``
    及之前期次的计划，在其观察窗口内**每个周龄**各贡献一行。

    每行是 ``(特征, 真实结局)``。特征锚定在“当时那个周龄周末”的可见
    信息上（周龄 1..4，即开始周后第一周到截止周），结局来自完整重放但
    只取当周已可确认者；锚点晚于 ``as_of_week`` 的行不生成。这样部署时
    任意周龄的预测都能在训练集里找到同周龄分箱，且训练与部署使用同一
    信息时点，杜绝结局泄漏进特征。
    """
    rows = []
    for plan in store.plans:
        if plan["term"] > max_term:
            continue
        outcome = store.timeline(plan["plan"]).outcome_known_by(as_of_week)
        if outcome is None or outcome == PENDING:
            continue
        for age in range(1, 5):
            anchor_week = plan["start"] + age
            if anchor_week > as_of_week:
                continue  # 锚点尚未到达，不能假装当时已有这些特征
            rows.append((build_features(store, plan, anchor_week), outcome))
    return rows
