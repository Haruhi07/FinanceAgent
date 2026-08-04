"""
Anomaly Detector - 异动检测
============================

职责:在信号流中发现"异常模式"

方法:
- 跨源共振:同一主体(个股/板块)在多个信号源同时出现
- 量价异动:量比突增 + 涨幅异常
- 板块共振:同板块多只个股同时涨停
"""

from __future__ import annotations
import logging
from collections import defaultdict, Counter
from datetime import datetime

from .base import BaseAgent, AgentResult
from tools import (
    SIGNAL_LIMIT_UP, SIGNAL_LIMIT_DOWN,
    SIGNAL_BOARD_CHANGE, SIGNAL_FUND_FLOW,
)
from tools.persist import save_anomalies
import config

logger = logging.getLogger(__name__)


class AnomalyDetector(BaseAgent):
    name = "anomaly_detector"
    
    def _run(self, input_data: dict | None) -> AgentResult:
        """input_data: 不需要,从 Working Memory 读 signals"""
        signals = self.wm.get("signals", [])
        if not signals:
            return AgentResult(
                agent_name=self.name, success=False,
                errors=["no signals in working memory"],
            )
        
        anomalies: list[dict] = []
        
        # ---- 1. 板块级共振(同板块多只个股涨停) ----
        anomalies.extend(self._detect_board_resonance(signals))
        
        # ---- 2. 板块+资金流共振(异动板块同时资金流入) ----
        anomalies.extend(self._detect_board_with_fundflow(signals))
        
        # ---- 3. 异动+涨跌停共振(板块异动频繁 + 涨停多) ----
        anomalies.extend(self._detect_change_with_limitup(signals))
        
        # ---- 4. 跌停+异动共振(风险信号) ----
        anomalies.extend(self._detect_limitdown_with_change(signals))
        
        # 写入 WM
        for a in anomalies:
            self.wm.append("anomalies", a)

        # 2026-08 新增:落盘 JSON 供 Orchestrator 读 + 留 trace
        path = save_anomalies(anomalies, extra={
            "input_signal_count": len(signals),
            "by_type": dict(Counter(a.get("anomaly_type", "?") for a in anomalies)),
            "source": "anomaly_detector",
        })

        return AgentResult(
            agent_name=self.name,
            success=len(anomalies) > 0,
            items=anomalies,
            metrics={
                "anomaly_count": len(anomalies),
                "input_signal_count": len(signals),
                "by_type": dict(Counter(a.get("anomaly_type", "?") for a in anomalies)),
                "json_path": str(path),
            },
        )
    
    def _detect_board_resonance(self, signals: list[dict]) -> list[dict]:
        """同板块多只个股涨停 -> 板块级热点(2026-08-04 修复:个股去重)

        3 天扫描时,同一只股可能在多天都有涨停信号,
        需要按 name 去重,避免 symbols 列表里堆重复名。
        """
        board_to_names: dict[str, list[str]] = defaultdict(list)
        for s in signals:
            if s.get("signal_type") == SIGNAL_LIMIT_UP and s.get("board"):
                name = (s.get("name") or "").strip()
                if name:
                    board_to_names[s["board"]].append(name)

        anomalies = []
        for board, names in board_to_names.items():
            # 按出现频次排序,保留顺序去重(dict 保序)
            unique_names = list(dict.fromkeys(names))
            unique_count = len(unique_names)
            if unique_count >= 3:  # 同板块 3 只以上**不同**个股涨停算共振
                anomalies.append({
                    "anomaly_type": "board_resonance",
                    "board": board,
                    "limit_up_count": unique_count,           # 修复:用去重后数量
                    "total_signals": len(names),              # 2026-08-04 新增:原始信号数(去重前)
                    "symbols": unique_names[:10],              # 修复:去重后的名字
                    "score_hint": min(0.5 + unique_count * 0.05, 0.95),
                    "description": f"【板块共振】{board} 板块 {unique_count} 只个股涨停",
                    "ts": _now(),
                })
        return anomalies
    
    def _detect_board_with_fundflow(self, signals: list[dict]) -> list[dict]:
        """板块异动 + 资金流入共振"""
        # 异动板块
        change_boards = set()
        for s in signals:
            if s.get("signal_type") == SIGNAL_BOARD_CHANGE and s.get("name"):
                change_boards.add(s["name"])
        
        # 资金流入 Top 板块
        fundflow_boards = set()
        for s in signals:
            if s.get("signal_type") == SIGNAL_FUND_FLOW and s.get("name"):
                # 只看净流入
                try:
                    amt = float(s.get("raw", {}).get("主力净流入-净额", 0) or 0)
                    if amt > 0:
                        fundflow_boards.add(s["name"])
                except (ValueError, TypeError):
                    pass
        
        # 共振:同时在异动 + 资金流入
        resonance_boards = change_boards & fundflow_boards
        
        anomalies = []
        for board in resonance_boards:
            anomalies.append({
                "anomaly_type": "board_change_with_fundflow",
                "board": board,
                "score_hint": 0.85,
                "description": f"【资金+异动共振】{board} 板块异动频繁且主力资金净流入",
                "ts": _now(),
            })
        return anomalies
    
    def _detect_change_with_limitup(self, signals: list[dict]) -> list[dict]:
        """板块异动频繁 + 该板块多只涨停 -> 强信号"""
        # 板块异动次数
        board_change_count: dict[str, int] = Counter()
        for s in signals:
            if s.get("signal_type") == SIGNAL_BOARD_CHANGE and s.get("name"):
                change_count = s.get("raw", {}).get("板块异动总次数", 0)
                try:
                    board_change_count[s["name"]] += int(change_count or 0)
                except (ValueError, TypeError):
                    pass
        
        # 板块涨停数
        board_limitup: dict[str, int] = Counter()
        for s in signals:
            if s.get("signal_type") == SIGNAL_LIMIT_UP and s.get("board"):
                board_limitup[s["board"]] += 1
        
        anomalies = []
        for board, change_cnt in board_change_count.items():
            lu_cnt = board_limitup.get(board, 0)
            if change_cnt >= 5 and lu_cnt >= 2:
                anomalies.append({
                    "anomaly_type": "change_with_limitup",
                    "board": board,
                    "change_count": change_cnt,
                    "limit_up_count": lu_cnt,
                    "score_hint": 0.9,
                    "description": f"【强共振】{board} 板块异动 {change_cnt} 次,涨停 {lu_cnt} 只",
                    "ts": _now(),
                })
        return anomalies
    
    def _detect_limitdown_with_change(self, signals: list[dict]) -> list[dict]:
        """跌停 + 板块异动 = 风险信号"""
        board_limitdown: dict[str, int] = Counter()
        for s in signals:
            if s.get("signal_type") == SIGNAL_LIMIT_DOWN and s.get("board"):
                board_limitdown[s["board"]] += 1
        
        anomalies = []
        for board, cnt in board_limitdown.items():
            if cnt >= 3:
                anomalies.append({
                    "anomaly_type": "risk_concentration",
                    "board": board,
                    "limit_down_count": cnt,
                    "score_hint": 0.6,  # 风险信号评分低一些
                    "description": f"【风险集中】{board} 板块 {cnt} 只个股跌停",
                    "ts": _now(),
                })
        return anomalies


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
