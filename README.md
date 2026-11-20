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
