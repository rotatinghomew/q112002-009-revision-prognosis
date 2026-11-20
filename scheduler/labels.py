"""业务时间与四种计划状态的判定。

术语（与 README 对齐）：

* 计划开始周 ``start``；观察期限为“开始后第四周末（含当周）”，
  故截止周 ``deadline = start + 4``，可干预区间为 ``start..deadline``。
* 事件有两个时间：发生周 ``event_week`` 与当时看得见的得知周 ``known_week``。
  当周判定只认真实可见的登记（``known_week <= 当周``）；补记的事件在它
  被得知之前，对当周而言等于不存在。
* 四种当周状态互斥：

  ``completed``           确认完成（当时已可见）
  ``switched``            换目标提前退出（当时已可见）
  ``pending_observation`` 到截止时还没来得及观察（截止周未到且未见终局）
  ``confirmed_incomplete``可确认的未完成（截止周已过且未见任何终局事件）

  换目标与完成是不同的业务出口，绝不允许压成“最终交没交”。
"""

# 开始后第四周末（含当周）
DEADLINE_OFFSET = 4

COMPLETED = "completed"
SWITCHED = "switched"
PENDING = "pending_observation"
INCOMPLETE = "confirmed_incomplete"

TERMINAL_KINDS = {
    "确认完成": COMPLETED,
    "换目标": SWITCHED,
}
PROGRESS_KIND = "自主进展"

FORECAST_STATES = (COMPLETED, SWITCHED, PENDING, INCOMPLETE)
# 回测里用于对照的真实结局（不含“尚未观察”，那是时间未到而不是结局）。
REALIZED_OUTCOMES = (COMPLETED, SWITCHED, INCOMPLETE)


def deadline_week(plan):
    """计划观察截止周 = 开始周 + 4。"""
    return plan["start"] + DEADLINE_OFFSET


def intervention_open(plan, week):
    """计划在该周是否仍处于可干预区间（不考虑已知终局）。"""
    return plan["start"] <= week <= deadline_week(plan)


class PlanTimeline:
    """单个计划按得知周可见的事件视图。"""

    def __init__(self, plan, events):
        self.plan = plan
        self._events = sorted(
            (e for e in events if e["plan"] == plan["plan"]),
            key=lambda e: (e["known_week"], e["event_week"]),
        )

    def visible_events(self, week):
        """当周 ``week`` 看得见的事件（含当周得知者）。"""
        return [e for e in self._events if e["known_week"] <= week]

    def known_terminal(self, week):
        """当周已可见的终局事件；没有则返回 ``None``。

        数据契约保证一个计划至多一条终局事件；这里仍做防御性处理，
        始终取得知最早的一条，保证判定稳定。
        """
        for event in self._events:
            if event["known_week"] <= week and event["kind"] in TERMINAL_KINDS:
                return event
        return None

    def progress_known(self, week):
        """当周是否已看见至少一条自主进展登记。"""
        return any(
            e["known_week"] <= week and e["kind"] == PROGRESS_KIND
            for e in self._events
        )

    def status_at(self, week):
        """计划在 ``week`` 周末截点的四态判定。

        截止周含在观察期内：截止周当周仍可能出现终局事件，因此未见终局
        时只能判“到截止时还没来得及观察”；只有截止周**之后**仍无终局，
        才可确认未完成。这样保证“可干预”与“已确认未完成”不会同时成立。
        """
        terminal = self.known_terminal(week)
        if terminal is not None:
            return TERMINAL_KINDS[terminal["kind"]]
        if week <= deadline_week(self.plan):
            return PENDING
        return INCOMPLETE

    def is_active(self, week):
        """当周计划是否仍可干预：在窗口内且未见终局。"""
        return intervention_open(self.plan, week) and self.known_terminal(week) is None

    def realized_outcome(self):
        """用完整登记重放的真实结局（回测对照用，不能用于当周特征）。

        终局事件若发生在截止周之后则不计入观察期（本资料中不存在，
        但规则必须显式），此时按未完成处理。
        """
        deadline = deadline_week(self.plan)
        for event in self._events:
            if (
                event["kind"] in TERMINAL_KINDS
                and event["event_week"] <= deadline
            ):
                return TERMINAL_KINDS[event["kind"]]
        return INCOMPLETE

    def outcome_known_by(self, week):
        """真实结局是否在 ``week`` 当周已可确认（训练样本入模门槛）。

        截止周含在观察期内，故“可确认未完成”最早只能在截止周之后的
        下一周成立；截止周当周仍允许终局事件出现，不能提前判定。
        """
        deadline = deadline_week(self.plan)
        terminal = self.known_terminal(week)
        if terminal is not None and terminal["event_week"] <= deadline:
            return TERMINAL_KINDS[terminal["kind"]]
        if week > deadline:
            return INCOMPLETE
        return None
