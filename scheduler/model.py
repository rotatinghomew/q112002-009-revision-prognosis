"""多结局概率模型：带 Dirichlet 收缩的分箱估计与适用性门槛。

为什么不用更复杂的模型：

* 样本量小（每期 36 项计划、五个可训练期次共 180 项），高方差模型在逐周
  回测里会放大不稳定性；分箱 + 层次收缩给出可审计、可复现的概率与
  解析的后验不确定性。
* 这里估计的是**可观察结局的条件分布** P(结局 | 当周可见特征)，不是
  干预的因果效应。历史邀请由辅导员的优先规则产生，被约谈者与未约谈者
  不可比，因此本模块刻意不提供任何“再约一次的完成率提升”估计量；
  会谈回执只用于容量/承诺管理，不进入结局模型的特征。

模型结构（层次收缩）：

* 基础率：全渠道基础上的 Dirichlet 后验，作为收缩目标；
* 分箱率：``渠道×方向×课程×周龄×进展可见`` 分箱的 Dirichlet 后验；
* 后验均值 = (分箱计数 + 收缩权重×基础率) / (分箱样本 + 收缩权重)。

样本不足或课程/字段在训练期从未出现时，``supported=False``，调用方必须
把该计划转入人工甄别，而不是当成低风险搁置。
"""

import math

from .labels import COMPLETED, SWITCHED, INCOMPLETE, REALIZED_OUTCOMES
from .features import MODEL_FEATURES, MIN_STRATUM_SAMPLES

OUTCOMES = REALIZED_OUTCOMES
# Dirichlet 先验强度（对称先验），等价于每结局 ALPHA 个伪观测。
ALPHA = 1.0
# 收缩权重：分箱样本达到该数时，分箱证据与基础率各占一半权重。
SHRINK_STRENGTH = 10.0
# 可信区间水平（95%）。
CREDIBLE_MASS = 0.95


def _beta_critical_quantiles(total_alpha):
    """用正态近似返回 Beta 分布的 95% 双侧分位点系数。

    标准库没有 Beta 逆 CDF；对后验样本量（含先验）≥ 3 的分箱，正态近似
    足够保守。区间仅用于不确定性展示与门槛判定，不参与排序。
    """
    z = 1.959963984540054
    return z


def credible_interval(count, total, prior_mass):
    """单结局 Beta(count+ALPHA, total-count+prior_mass) 的近似 95% 区间。"""
    a = count + ALPHA
    b = total - count + ALPHA * (len(OUTCOMES) - 1)
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    half = _beta_critical_quantiles(a + b) * math.sqrt(var)
    return max(0.0, mean - half), min(1.0, mean + half)


class OutcomeModel:
    """在一组带标签训练行上拟合的三结局收缩模型。"""

    def __init__(self, training_rows, training_terms):
        self.training_terms = tuple(sorted(training_terms))
        self.n_rows = len(training_rows)
        self._base_counts = {o: 0 for o in OUTCOMES}
        self._strat_counts = {}
        self._support_counts = {}   # 渠道×方向×课程 粒度的样本数
        self._channels = set()
        self._courses = set()
        self._feature_values = {name: set() for name in MODEL_FEATURES}

        for features, outcome in training_rows:
            if outcome not in OUTCOMES:
                continue
            self._base_counts[outcome] += 1
            key = features.model_key()
            bucket = self._strat_counts.setdefault(
                key, {o: 0 for o in OUTCOMES}
            )
            bucket[outcome] += 1
            support = features.support_key()
            self._support_counts[support] = self._support_counts.get(support, 0) + 1
            self._channels.add(features.channel)
            self._courses.add(features.course)
            self._feature_values["channel"].add(features.channel)
            self._feature_values["focus"].add(features.focus)
            self._feature_values["course"].add(features.course)
            self._feature_values["progress_known"].add(features.progress_known)
            self._feature_values["age_band"].add(features.age_band)

        self.base_total = sum(self._base_counts.values())
        self.base_probs = self._dirichlet_mean(self._base_counts, self.base_total)

    @staticmethod
    def _dirichlet_mean(counts, total):
        denom = total + ALPHA * len(OUTCOMES)
        return {o: (counts[o] + ALPHA) / denom for o in OUTCOMES}

    def support_for(self, features):
        """适用性结论：该计划的特征组合是否仍受历史表现支持。"""
        reasons = []
        key = features.model_key()
        support_key = features.support_key()
        stratum_total = sum(self._strat_counts.get(key, {}).values())
        support_total = self._support_counts.get(support_key, 0)

        if features.course not in self._courses:
            reasons.append(f"课程 {features.course!r} 在训练期次从未出现")
        if features.channel not in self._channels:
            reasons.append(f"渠道 {features.channel!r} 无训练样本")
        if support_total < MIN_STRATUM_SAMPLES:
            reasons.append(
                f"渠道×方向×课程样本 {support_total} < 门槛 {MIN_STRATUM_SAMPLES}"
            )
        if stratum_total < MIN_STRATUM_SAMPLES:
            reasons.append(
                f"完整分箱样本 {stratum_total} < 门槛 {MIN_STRATUM_SAMPLES}，仅基础率可用"
            )
        return {
            "supported": not reasons,
            "reasons": reasons,
            "stratum_samples": stratum_total,
            "support_samples": support_total,
        }

    def predict(self, features):
        """返回三结局概率、95% 可信区间与适用性标记。"""
        support = self.support_for(features)
        key = features.model_key()
        counts = self._strat_counts.get(key, {o: 0 for o in OUTCOMES})
        total = sum(counts.values())

        probs = {}
        intervals = {}
        if total == 0:
            # 分箱完全无样本：退到基础率（仍然标注 unsupported）。
            for outcome in OUTCOMES:
                probs[outcome] = self.base_probs[outcome]
                intervals[outcome] = credible_interval(
                    self._base_counts[outcome], self.base_total, ALPHA
                )
        else:
            shrink = SHRINK_STRENGTH
            denom = total + shrink
            for outcome in OUTCOMES:
                p_stratum = (counts[outcome] + ALPHA) / (
                    total + ALPHA * len(OUTCOMES)
                )
                probs[outcome] = (
                    total * p_stratum + shrink * self.base_probs[outcome]
                ) / denom
                intervals[outcome] = credible_interval(counts[outcome], total, ALPHA)

        # 数值清理：收缩加权后归一化，保证概率和为 1。
        scale = sum(probs.values())
        probs = {o: probs[o] / scale for o in OUTCOMES}
        return {
            "probs": probs,
            "intervals": intervals,
            "supported": support["supported"],
            "support": support,
        }

    # ---- 回测指标 -----------------------------------------------------
    def evaluate(self, eval_rows):
        """在带标签评估行上计算多分类对数损失、Brier 与基准对照。"""
        if not eval_rows:
            return None
        ll = 0.0
        brier = 0.0
        base_ll = 0.0
        per_outcome = {o: {"n": 0, "correct_prob": []} for o in OUTCOMES}
        for features, actual in eval_rows:
            pred = self.predict(features)["probs"]
            p = min(max(pred[actual], 1e-12), 1.0)
            ll -= math.log(p)
            brier += sum(
                (pred[o] - (1.0 if o == actual else 0.0)) ** 2 for o in OUTCOMES
            )
            base_ll -= math.log(min(max(self.base_probs[actual], 1e-12), 1.0))
            per_outcome[actual]["n"] += 1
            per_outcome[actual]["correct_prob"].append(pred[actual])
        n = len(eval_rows)
        return {
            "n": n,
            "log_loss": ll / n,
            "base_log_loss": base_ll / n,
            "brier": brier / n,
            "outcome_rates": {
                o: {
                    "n": v["n"],
                    "mean_predicted_prob": (
                        sum(v["correct_prob"]) / len(v["correct_prob"])
                        if v["correct_prob"]
                        else None
                    ),
                }
                for o, v in per_outcome.items()
            },
        }


def train_for_term(store, target_term, as_of_week):
    """滚动训练：只使用目标期次之前、且结局在截点已可确认的计划。"""
    from .features import training_rows

    terms = tuple(t for t in {p["term"] for p in store.plans} if t < target_term)
    rows = training_rows(store, max(terms) if terms else 0, as_of_week)
    return OutcomeModel(rows, terms)
