"""
主入口
======

用法:
    python main.py                  # 跑一次完整流程
    python main.py --mode scan      # 只扫描
    python main.py --mode report    # 只生成报告(不重新跑)
    python main.py --human          # 带人工审核的流程
    python main.py --no-proxy       # 禁用代理(解决 clash 代理对 push2.eastmoney.com 的兼容问题)
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# 让 import 能找到项目根目录
sys.path.insert(0, str(Path(__file__).parent))

# 必须先 parse args,才能在 import akshare 之前决定是否禁用代理
_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument("--no-proxy", action="store_true", help="禁用代理")
_args, _remaining = _arg_parser.parse_known_args()

if _args.no_proxy:
    from tools.disable_proxy import disable_proxy
    disable_proxy()

import config
from agents import (
    Orchestrator, ScannerAgent, AnomalyDetector,
    HotspotDetector, TopicModeler, ResearchAgent,
    WriterAgent, ReviewerAgent,
)
from evaluation import get_evaluator
from memory import get_working_memory, get_episodic_memory


def setup_logging():
    log_file = config.LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 根 logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # 清掉已有 handler
    root.handlers.clear()
    
    # 控制台
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, config.LOG_LEVEL))
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)
    
    # 文件
    file_h = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
    ))
    root.addHandler(file_h)
    
    return log_file


def run_full(orchestrator: Orchestrator) -> dict:
    """跑完整流程"""
    print("\n" + "=" * 60)
    print("  财经热点发现与内容生产 Agent 系统")
    print("  Finance Hotspot Discovery & Content Production Agent")
    print("=" * 60)
    print(f"  开始时间:{datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60 + "\n")
    
    t0 = time.time()
    result = orchestrator.run({})
    duration = time.time() - t0
    
    print(f"\n{'='*60}")
    print(f" 流程结束,耗时 {duration:.1f}s,状态: {result.metrics}")
    print(f"{'='*60}\n")
    
    return result.to_dict()


def run_with_human(orchestrator: Orchestrator) -> dict:
    """带人工审核的流程"""
    print("\n[模式] 带人工审核的流程\n")
    result = orchestrator.run_with_human_review()
    return result.to_dict()


def run_scan_only() -> dict:
    """只跑扫描 + 异动检测 + 热点识别(轻量模式)"""
    print("\n[模式] 轻量扫描(只到热点识别)\n")
    
    scanner = ScannerAgent()
    anomaly = AnomalyDetector()
    hotspot = HotspotDetector()
    
    r1 = scanner.run({})
    print(f"  [scanner] {r1.metrics}")
    
    r2 = anomaly.run({})
    print(f"  [anomaly] {r2.metrics}")
    
    r3 = hotspot.run({})
    print(f"  [hotspot] {r3.metrics}")
    
    # 打印 top 候选
    wm = get_working_memory()
    candidates = wm.get("hotspot_candidates", [])
    print(f"\n  Top 候选:")
    for c in candidates[:10]:
        print(f"    [{c['confidence']:4s}] {c['subject']:20s} score={c['score']} - {c['description'][:60]}")
    
    return {
        "scan": r1.to_dict(),
        "anomaly": r2.to_dict(),
        "hotspot": r3.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="财经热点 Agent 系统")
    parser.add_argument("--mode", choices=["full", "scan", "report", "human"],
                        default="full", help="运行模式")
    parser.add_argument("--no-eval", action="store_true", help="不生成评估报告")
    parser.add_argument("--no-proxy", action="store_true",
                        help="禁用代理(解决 clash 代理对 push2.eastmoney.com 的兼容问题)")
    args = parser.parse_args()
    
    log_file = setup_logging()
    logger = logging.getLogger("main")
    logger.info(f"开始运行,模式={args.mode}, no_proxy={args.no_proxy}")
    
    wm = get_working_memory()
    orchestrator = Orchestrator()
    
    if args.mode == "full":
        result = run_full(orchestrator)
    elif args.mode == "scan":
        result = run_scan_only()
    elif args.mode == "human":
        result = run_with_human(orchestrator)
    elif args.mode == "report":
        # 不重新跑,只对当前 WM 生成报告
        pass
    else:
        result = {}
    
    # 生成评估报告
    if not args.no_eval and args.mode != "scan":
        print("\n" + "="*60)
        print("  生成评估报告...")
        print("="*60 + "\n")
        evaluator = get_evaluator()
        eval_result = evaluator.evaluate_run()
        report_path = evaluator.generate_report(eval_result)
        print(f"  报告: {report_path}")
        
        # 打印 summary
        print(f"\n  整体健康度: {eval_result['summary']}")
        print(f"  信号: {eval_result['discovery']['signal_count']}")
        print(f"  候选: {eval_result['discovery']['candidate_count']}")
        print(f"  通过审核: {eval_result['business']['published_count']}")
        print()
    
    logger.info(f"运行结束,日志: {log_file}")
    return result


if __name__ == "__main__":
    main()
