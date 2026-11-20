#!/usr/bin/env python3
"""排期系统命令行入口。

以隔离模式运行（``sys.flags.isolated``），避免从资料目录或当前目录拾取
被植入的模块；脚本自身把项目根目录加入 ``sys.path``，因此在任意工作
目录都可调用：

    python3 -I run_scheduler.py check
    python3 -I run_scheduler.py schedule --target-week 45 --term 6
    python3 -I run_scheduler.py backtest --out report.json
"""

import sys
from pathlib import Path

if not sys.flags.isolated:
    # 重新以隔离模式启动自身，防止 PYTHONPATH/当前目录注入。
    raise SystemExit(
        __import__("subprocess").call(
            [sys.executable, "-I", str(Path(__file__).resolve()), *sys.argv[1:]]
        )
    )

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scheduler.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
