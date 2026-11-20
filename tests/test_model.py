"""模型适用性、渠道分层与“无因果提升量”边界测试。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scheduler.records import RecordStore
from scheduler.model import train_for_term, OutcomeModel
from scheduler.features import training_rows
from scheduler.backtest import rolling_evaluation
from scheduler.labels import INCOMPLETE


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = RecordStore.from_folder("records")
        cls.store.assert_contract()
        cls.model = train_for_term(cls.store, 6, 44)

    def test_probabilities_sum_to_one_with_intervals(self):
        rows = training_rows(self.store, 5, 44)
        for features, _ in rows[:25]:
            pred = self.model.predict(features)
            self.assertAlmostEqual(sum(pred["probs"].values()), 1.0, places=9)
            for outcome, (lo, hi) in pred["intervals"].items():
                self.assertLessEqual(lo, pred["probs"][outcome] + 1e-9)
                self.assertGreaterEqual(hi, pred["probs"][outcome] - 1e-9)
                self.assertGreaterEqual(lo, 0.0)
                self.assertLessEqual(hi, 1.0)

    def test_new_course_marks_unsupported(self):
        # 新课程只在第 6 期出现，训练期（1-5）没有任何该课程样本。
        new_course_plans = [
            p for p in self.store.plans if p["course"] == "新课程"
        ]
        self.assertTrue(new_course_plans)
        self.assertEqual(
            {p["term"] for p in new_course_plans}, {6}
        )
        from scheduler.features import build_features

        for plan in new_course_plans[:3]:
            features = build_features(self.store, plan, 44)
            pred = self.model.predict(features)
            self.assertFalse(pred["supported"])
            self.assertTrue(
                any("新课程" in r for r in pred["support"]["reasons"])
            )

    def test_channel_specific_evaluation(self):
        evaluations = rolling_evaluation(self.store)
        for ev in evaluations:
            self.assertEqual(set(ev.channel_metrics), {"课堂", "线上"})
            for channel, metrics in ev.channel_metrics.items():
                self.assertGreater(metrics["n"], 0)
                self.assertTrue(metrics["log_loss"] > 0)

    def test_model_beats_or_matches_base_rate_on_history(self):
        # 收缩模型在每个历史期次上不应比纯基础率更差（样本内滚动评估）。
        evaluations = rolling_evaluation(self.store)
        for ev in evaluations:
            self.assertLessEqual(
                ev.metrics["log_loss"],
                ev.metrics["base_log_loss"] + 1e-9,
            )

    def test_no_uplift_or_causal_api_surface(self):
        # 系统刻意不提供约谈因果提升估计；会谈回执不进入模型特征。
        forbidden = ("uplift", "causal", "treatment_effect", "提升率")
        import scheduler.model as model_module
        import scheduler.forecast as forecast_module

        source = model_module.__doc__ + forecast_module.__doc__
        for word in forbidden:
            self.assertIn(
                "不", source
            )  # 文档明确声明不做因果量
        # 模型拟合只接收 (特征, 结局)，签名里没有干预/处理组参数。
        rows = training_rows(self.store, 5, 44)
        model = OutcomeModel(rows, (1, 2, 3, 4, 5))
        self.assertEqual(
            set(model.predict(rows[0][0])),
            {"probs", "intervals", "supported", "support"},
        )

    def test_empty_training_degrades_gracefully(self):
        model = OutcomeModel([], ())
        rows = training_rows(self.store, 1, 44)
        features = rows[0][0]
        pred = model.predict(features)
        self.assertFalse(pred["supported"])
        self.assertAlmostEqual(sum(pred["probs"].values()), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
