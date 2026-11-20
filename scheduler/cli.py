"""命令行入口：契约检查、当周预测、下周排期、滚动回测与审计导出。

用法（Python 3.11 标准库，无需第三方包）：

    python3 -m scheduler.cli check
    python3 -m scheduler.cli forecast --week 44 --term 6
    python3 -m scheduler.cli schedule --target-week 45 [--term 6]
    python3 -m scheduler.cli backtest [--weeks 1..44] [--out report.json]
    python3 -m scheduler.cli explain --plan p6-1 --week 44

资料目录通过 --records 传入；读取与计算分离，输出只写到指定位置。
"""

import argparse
import json
import sys
from pathlib import Path

from .records import RecordStore
from .model import train_for_term
from .forecast import forecast_week, forecast_plan
from .allocation import allocate_week
from .backtest import rolling_evaluation, replay_weeks
from .labels import deadline_week

DEFAULT_RECORDS = Path(__file__).resolve().parent.parent / "records"


def _store(path):
    store = RecordStore.from_folder(path)
    store.assert_contract()
    return store


def cmd_check(args):
    store = _store(args.records)
    print("资料契约检查通过")
    for table, rows in store.tables.items():
        print(f"  {table}: {len(rows)} 条登记")
    print(f"当前事件截点周: {store.current_cutoff()}")
    return 0


def _model_for(store, week, term):
    if term is None:
        term = max(p["term"] for p in store.plans)
    return train_for_term(store, term, week), term


def cmd_forecast(args):
    store = _store(args.records)
    model, term = _model_for(store, args.week, args.term)
    forecasts = forecast_week(store, model, args.week, term=term)
    payload = {
        "as_of_week": args.week,
        "term": term,
        "training_terms": model.training_terms,
        "n_training_rows": model.n_rows,
        "forecasts": [fc.to_dict() for fc in forecasts],
    }
    _emit(payload, args.out)
    if not args.out:
        for fc in forecasts:
            risk = (
                f"P(未完成)={fc.incomplete_risk:.3f}"
                if fc.incomplete_risk is not None
                else f"状态={fc.visible_state}"
            )
            flag = "" if fc.evidence_sufficient else " [证据不足→人工甄别]"
            print(f"{fc.plan_id} {fc.student} {risk}{flag}")
    return 0


def cmd_schedule(args):
    store = _store(args.records)
    as_of = args.as_of_week if args.as_of_week is not None else args.target_week - 1
    model, term = _model_for(store, as_of, args.term)
    allocation = allocate_week(
        store, model, args.target_week, as_of, term=term
    )
    forecasts = forecast_week(store, model, as_of, term=term)
    payload = {
        "target_week": args.target_week,
        "as_of_week": as_of,
        "term": term,
        "training_terms": model.training_terms,
        "forecasts": [fc.to_dict() for fc in forecasts],
        "allocation": allocation.to_dict(),
    }
    _emit(payload, args.out)
    if not args.out:
        _print_allocation(allocation)
    return 0


def cmd_backtest(args):
    store = _store(args.records)
    weeks = _parse_weeks(args.weeks, store) if args.weeks else [
        w for w in store.all_weeks_with_capacity()
        if w <= store.current_cutoff() + 1
    ]
    evaluations = rolling_evaluation(store)
    replay = replay_weeks(store, weeks, lag=args.lag)
    payload = {
        "cutoff_week": store.current_cutoff(),
        "as_of_convention": (
            f"截点=目标周-{args.lag}（0=当周初排当周，1=周二排下周）"
        ),
        "rolling_evaluation": [
            {
                "term": ev.term,
                "training_terms": ev.training_terms,
                "n_training": ev.n_training,
                "unsupported_share": round(ev.unsupported_share, 4),
                "metrics": ev.metrics,
                "channel_metrics": ev.channel_metrics,
            }
            for ev in evaluations
        ],
        "weekly_replay": replay,
    }
    _emit(payload, args.out)
    if not args.out:
        print("滚动期次评估（对数损失 vs 基础率）：")
        for ev in evaluations:
            m = ev.metrics
            print(
                f"  term {ev.term}: n={m['n']} "
                f"log_loss={m['log_loss']:.4f} "
                f"base={m['base_log_loss']:.4f} "
                f"brier={m['brier']:.4f} "
                f"unsupported={ev.unsupported_share:.0%}"
            )
        print(f"逐周重放: {len(replay)} 周（详见 --out JSON）")
    return 0


def cmd_explain(args):
    store = _store(args.records)
    plan = store.plan(args.plan)
    model, _ = _model_for(store, args.week, plan["term"])
    fc = forecast_plan(store, model, plan, args.week)
    payload = fc.to_dict()
    payload["visible_events"] = [
        e for e in store.timeline(args.plan).visible_events(args.week)
    ]
    payload["deadline"] = deadline_week(plan)
    _emit(payload, args.out)
    if not args.out:
        print(f"计划 {args.plan} 在第 {args.week} 周截点的可追溯依据：")
        for reason in fc.reasons:
            print(f"  - {reason}")
        for event in payload["visible_events"]:
            print(
                f"  - 可见事件 {event['kind']} 发生周{event['event_week']} "
                f"得知周{event['known_week']}"
            )
    return 0


def _print_allocation(allocation):
    a = allocation
    print(
        f"目标周 {a.target_week}（截点 {a.as_of_week}） "
        f"容量快照: {a.capacity_snapshot}"
    )
    print(f"保留承诺 {len(a.held_commitments)} 个：")
    for c in a.held_commitments:
        print(f"  {c.student} {c.mentor} {c.state}")
    for req in a.release_requests:
        print(
            f"  待核实释放: {req['plan']}/{req['student']} "
            f"因 {req['evidence_event']}（得知周{req['known_week']}）→ {req['action']}"
        )
    print(f"算法新邀请 {len(a.invited)} 个：")
    for inv in a.invited:
        print(
            f"  {inv['student']} → {inv['mentor']} 风险={inv['incomplete_risk']}"
        )
    print(f"人工甄别 {len(a.manual_review)} 个：")
    for item in a.manual_review:
        print(f"  {item['student']} → {item['mentor']} 依据={item['reasons']}")
    if a.manual_backlog:
        print(f"人工积压 {len(a.manual_backlog)} 个（不作低风险搁置）：")
        for item in a.manual_backlog:
            print(f"  {item['student']} {item['reasons']}")
    if a.unschedulable:
        print(f"不可安排 {len(a.unschedulable)} 个：")
        for item in a.unschedulable:
            print(f"  {item['student']}: {item['reason']}")
    for line in a.rule_trace:
        print(f"  规则: {line}")


def _parse_weeks(spec, store):
    if ".." in spec:
        lo, hi = spec.split("..")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x]


def _emit(payload, out):
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"已写出 {out}")
    else:
        print(text)


def build_parser():
    parser = argparse.ArgumentParser(description="自主修订计划预警与会谈排期")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check").set_defaults(func=cmd_check)

    p = sub.add_parser("forecast", help="某周截点的四态预测")
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--term", type=int)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("schedule", help="生成目标周排期建议")
    p.add_argument("--target-week", type=int, required=True)
    p.add_argument("--as-of-week", type=int)
    p.add_argument("--term", type=int)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("backtest", help="滚动评估与逐周重放")
    p.add_argument("--weeks", help="如 9..16 或 9,10,11")
    p.add_argument(
        "--lag",
        type=int,
        default=0,
        choices=(0, 1),
        help="截点相对目标周的滞后：0=当周初排当周（默认，与容量录入约定一致），1=排下周",
    )
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("explain", help="追溯单个计划当周建议依据")
    p.add_argument("--plan", required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_explain)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
