"""逐周回测：在每个历史截点只用当时可见的资料重放预测与排期。

回测回答两个问题：

1. **预测质量**：滚动期次训练（训练期都早于评估期），在各期锚点周
   （截止周前一周）评估对数损失、Brier、基准对照与逐结局校准；
2. **排期可重现性**：对每个历史目标周，用 ``recorded <= 截点`` 的容量
   快照和 ``known_week <= 截点`` 的回执重放分配，记录当时席位、已承诺
   会谈、实际邀请与每条建议的信息依据。后来增加的容量在回测中不可见。

本模块不计算任何“约谈提升完成率”的量：被约谈者由历史优先规则选择，
处理组/对照组不可比，做差会把选择偏差误报成干预效果。
"""

from dataclasses import dataclass, field

from .labels import (
    COMPLETED,
    SWITCHED,
    INCOMPLETE,
    PENDING,
    deadline_week,
)
from .model import OutcomeModel
from .features import training_rows
from .allocation import allocate_week
from .forecast import forecast_week


@dataclass
class TermEvaluation:
    term: int
    training_terms: tuple
    n_training: int
    metrics: dict
    unsupported_share: float
    channel_metrics: dict


def rolling_evaluation(store, as_of_week=None):
    """期次级滚动评估：用 t 之前的期次训练，在 t 上评估。

    每个历史期次在“该期全部结局都已可确认”的最早截点上评估，避免全局
    截点把尚未走完观察期的期次错误地截掉样本；逐周重放仍由
    :func:`replay_weeks` 按各自截点严格过滤，两者不混用时间视角。
    """
    if as_of_week is None:
        as_of_week = store.current_cutoff()
    terms = sorted({p["term"] for p in store.plans})
    evaluations = []
    for target_term in terms:
        prior_terms = [t for t in terms if t < target_term]
        if not prior_terms:
            continue
        term_deadlines = [
            deadline_week(p)
            for p in store.plans
            if p["term"] == target_term
        ]
        # 截止周之后一周才能确认“无终局=未完成”，取该期最大确认截点，
        # 再受全局截点约束（未来期次不可能超纲）。
        term_cutoff = min(max(term_deadlines) + 1, as_of_week)
        # 训练集严格来自更早的期次（滚动泛化，不把往期累积样本混进
        # 评估分母）；截至该截点结局尚不可确认的计划自然没有训练行。
        train = [
            row
            for row in training_rows(store, max(prior_terms), term_cutoff)
            if store.plan(row[0].plan_id)["term"] in prior_terms
        ]
        model = OutcomeModel(train, prior_terms)
        eval_rows = [
            row
            for row in training_rows(store, target_term, term_cutoff)
            if store.plan(row[0].plan_id)["term"] == target_term
        ]
        if not eval_rows:
            continue
        metrics = model.evaluate(eval_rows)
        unsupported = sum(
            1 for features, _ in eval_rows
            if not model.support_for(features)["supported"]
        )
        by_channel = {}
        for channel in sorted({f.channel for f, _ in eval_rows}):
            subset = [(f, o) for f, o in eval_rows if f.channel == channel]
            by_channel[channel] = model.evaluate(subset)
        evaluations.append(
            TermEvaluation(
                term=target_term,
                training_terms=tuple(prior_terms),
                n_training=len(train),
                metrics=metrics,
                unsupported_share=unsupported / len(eval_rows),
                channel_metrics=by_channel,
            )
        )
    return evaluations


def replay_weeks(store, weeks, term=None, lag=0):
    """对一组历史目标周重放完整排期，返回逐周审计记录。

    ``lag`` 决定截点约定：``lag=1`` 模拟“周二排下周”（截点=目标周前一周），
    ``lag=0`` 模拟“当周初排当周”（截点=目标周，容量与回执按当周已录入、
    已得知者重放）。无论哪种约定，模型都只在当时已结束、结局已可确认的
    期次上训练；容量严格按 ``recorded <= 截点`` 重放。
    """
    replay = []
    for target_week in weeks:
        as_of = target_week - lag
        # 当时已可训练的期次：截止周已过且结局在 as_of 可确认。
        trainable_terms = sorted(
            {
                p["term"]
                for p in store.plans
                if deadline_week(p) <= as_of
            }
        )
        if trainable_terms:
            rows = training_rows(
                store, max(trainable_terms), as_of
            )
            model = OutcomeModel(rows, trainable_terms)
        else:
            model = OutcomeModel([], ())
        allocation = allocate_week(
            store, model, target_week, as_of, term=term
        )
        forecasts = forecast_week(store, model, as_of, term=term)
        replay.append(
            {
                "target_week": target_week,
                "as_of_week": as_of,
                "training_terms": model.training_terms,
                "n_training_rows": model.n_rows,
                "allocation": allocation.to_dict(),
                "forecast_summary": _summarize_forecasts(forecasts),
            }
        )
    return replay


def _summarize_forecasts(forecasts):
    from collections import Counter

    states = Counter(fc.visible_state for fc in forecasts)
    return {
        "states": dict(states),
        "active": sum(1 for fc in forecasts if fc.active),
        "evidence_insufficient": sum(
            1 for fc in forecasts if fc.active and not fc.evidence_sufficient
        ),
        "unsupported": sum(
            1 for fc in forecasts if fc.active and not fc.supported
        ),
        "risk_ranking": [
            {
                "plan": fc.plan_id,
                "student": fc.student,
                "incomplete_risk": (
                    round(fc.incomplete_risk, 4)
                    if fc.incomplete_risk is not None
                    else None
                ),
                "evidence_sufficient": fc.evidence_sufficient,
            }
            for fc in sorted(
                (fc for fc in forecasts if fc.active),
                key=lambda fc: (
                    fc.evidence_sufficient,
                    -(fc.incomplete_risk or 0.0),
                ),
            )
        ],
    }
