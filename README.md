# 自主修订周排期资料

`week` 是连续业务周，`term` 是教学期。计划观察期限为开始后第四周末（含当周）；完成或转出有独立的业务时间与当时可见时间。每周截点只允许使用 `known_week` 不晚于该周的资料；未来期不能拿来确定历史特征或模型选择。
`events` 可能由学生补记，类型是自主进展、确认完成、换目标。`attendance` 是历史邀请及处理回执；已接受或已出席不可并作“没有干预”。回测所有历史资源以 `capacity` 当时快照为准，`manual` 是在总席位内保留的人工甄别数量，不累加到总量。`availability` 记当期学生接受的导师；导师只能接自己的能力方向。多个计划不能令学生在同一周被重复约谈。

## 资料格式

资料是匿名合成登记，不包含真实学生、真实成绩或外部访问地址。UTF-8 JSON，每张表是对象数组；`null` 表示该字段无已知值。`records/catalog.json` 记录每张表的字段。

- `plans`：历期自主修订报名资料，学生可跨期出现。 字段：`plan`、`student`、`term`、`start`、`channel`、`focus`、`course`。
- `events`：发生周和得知周不同。 字段：`plan`、`kind`、`event_week`、`known_week`、`note`。
- `mentors`：导师能力方向。 字段：`mentor`、`focus`。
- `capacity`：过去与当前排期席位，recorded 为录入周。 字段：`week`、`mentor`、`seats`、`manual`、`recorded`。
- `availability`：当周学生的可参加时间已转为对应导师名单。 字段：`student`、`week`、`mentors`。
- `attendance`：实发邀请、接受和出席回执。 字段：`plan`、`student`、`mentor`、`week`、`state`、`known_week`。

## 本地查看

使用 Python 3.11 标准库，无需安装包。用 `python3 record_reader.py` 检查并列出收到的表行数；也可传入同格式资料目录。用 `python3 -m unittest discover -s tests -v` 检查资料契约。读取入口不会计算分析结果。

## 预警与排期系统（`scheduler/`）

同样只用标准库；建议始终以隔离模式运行，避免从资料目录拾取被植入的
模块（`python3 -I run_scheduler.py ...`）。

### 业务判定（`labels.py`）

- 截止周 = `start + 4`，截止周**含在**观察期内：截止周当周仍可能出现
  终局事件，未见终局时只能判 `pending_observation`；截止周之后仍无终局
  才可判 `confirmed_incomplete`。“可干预”与“已确认未完成”不会同时成立。
- 四种当周状态互斥：`completed` / `switched` /
  `pending_observation` / `confirmed_incomplete`。换目标与完成是不同
  出口，不压成“最终交没交”。
- 事件按 `known_week` 可见：补记事件在得知前对当周不存在。

### 预测与适用性（`features.py`、`model.py`、`forecast.py`）

- 训练样本是**纵向**的：每个结局已可确认的历史计划，在周龄 1..4 各
  贡献一行，特征只取该锚点周可见资料；期次严格滚动，未来期不入选。
- 模型是渠道×方向×课程×周龄×进展可见性分箱上的 Dirichlet 收缩估计，
  输出三结局概率、95% 近似可信区间和基础率对照（对数损失、Brier）。
- 分箱或 `渠道×方向×课程` 样本不足、新课程/新字段在训练期从未出现时，
  标记 `supported=false` 并写明原因；这些计划只能进人工甄别。
- 模型只估计可观察结局的条件分布，**不提供**“再约谈一次提升多少
  完成率”的因果量；会谈回执不进入结局特征（历史优先约谈带来选择偏差）。

### 排期（`allocation.py`）

新邀请必须同时满足：计划仍可干预、学生当周有可接收时间且存在方向匹配
导师、预测依据足够。`manual` 是总席位内的人工甄别预留（不累加总量），
算法席位不得占用人工名额；证据不足者进入人工队列，名额不足时进
`manual_backlog`，不会被当作低风险搁置。同一学生每周最多一谈，已有
回执者不重复邀约。已接受/已邀请席位先保留；只有当周可见的新状态证据
确认完成或退出时才生成“待导师核实”请求，核实前不释放，预测变化本身
不能撤席。

### 回测与审计（`backtest.py`、`cli.py`）

- `backtest`：期次滚动评估（训练期严格早于评估期）+ 逐周重放；容量按
  `recorded <= 截点`、回执按 `known_week <= 截点` 重放，后来增加的资源
  不会出现在历史周。`--lag 0` 为当周初排当周，`--lag 1` 为排下周。
- `explain --plan ... --week ...`：重现单个计划当周可见事件、状态、
  概率区间与每条建议依据，支撑“当时为什么发这条邀请”的追溯。

```
python3 -I run_scheduler.py check
python3 -I run_scheduler.py forecast --week 44 --term 6
python3 -I run_scheduler.py schedule --target-week 44 --term 6
python3 -I run_scheduler.py backtest --out report.json
python3 -I run_scheduler.py explain --plan p6-4 --week 44
```
