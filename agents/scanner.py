"""
Scanner Agent - 多源信号扫描
==============================

职责:7×24 多源实时监听,产出标准化 Signal

数据源(全部走 akshare):
- 涨停/跌停股池(主动发现"最热"信号的最快途径)
- 板块异动详情
- 盘口异动(大单买卖)
- 板块资金流
- 业绩预告/业绩报表
- 公司动态
- 全球财经快讯
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta

from .base import BaseAgent, AgentResult
from tools import (
    get_tools, Signal,
    SIGNAL_LIMIT_UP, SIGNAL_LIMIT_DOWN,
    SIGNAL_BOARD_CHANGE, SIGNAL_FUND_FLOW,
    SIGNAL_ANNOUNCEMENT, SIGNAL_NEWS,
)

logger = logging.getLogger(__name__)


class ScannerAgent(BaseAgent):
    name = "scanner"
    
    def __init__(self):
        super().__init__()
        self.tools = get_tools()
    
    def _run(self, input_data: dict | None) -> AgentResult:
        """input_data:
          - since: ISO 字符串(保留兼容)
          - limit: 每源每日的记录上限
          - scan_days: 扫描最近 N 个工作日(默认 config.SCANNER_SCAN_DAYS,2026-08 改造)
        """
        if input_data is None:
            input_data = {}
        since = input_data.get("since")
        limit = input_data.get("limit", 50)
        # 2026-08 改造:扫最近 N 个工作日(方案 A - 每次拉 3 天合并)
        scan_days = input_data.get("scan_days") or getattr(__import__("config"), "SCANNER_SCAN_DAYS", 3)
        scan_dates = self._recent_trade_dates(scan_days)
        self.logger.info(f"扫描最近 {scan_days} 个工作日: {scan_dates}")

        signals: list[Signal] = []
        errors: list[str] = []
        seen_keys: set[tuple] = set()  # 去重(代码+日期+类型)

        # ---- 1. 涨停股池(最热信号) - 按日期扫 ----
        for date in scan_dates:
            try:
                records = self.tools.get_limit_up_pool(date=date)
                for r in records[:limit]:
                    sym = str(r.get("代码", ""))
                    key = (sym, date, "limit_up")
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    sig = self._from_limit_up(r)
                    sig.timestamp = f"{date}T{_now()[11:]}"  # 标注实际日期
                    signals.append(sig)
                self.logger.info(f"limit_up {date}: {len(records)} records")
            except Exception as e:
                errors.append(f"limit_up {date}: {e}")

        # ---- 2. 跌停股池(风险信号) - 按日期扫 ----
        for date in scan_dates:
            try:
                records = self.tools.get_limit_down_pool(date=date)
                for r in records[:limit]:
                    sym = str(r.get("代码", ""))
                    key = (sym, date, "limit_down")
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    sig = self._from_limit_down(r)
                    sig.timestamp = f"{date}T{_now()[11:]}"
                    signals.append(sig)
                self.logger.info(f"limit_down {date}: {len(records)} records")
            except Exception as e:
                errors.append(f"limit_down {date}: {e}")

        # ---- 3. 板块异动(板块级共振) - 当天,接口不支持日期 ----
        try:
            records = self.tools.get_board_change()
            for r in records[:limit]:
                signals.append(self._from_board_change(r))
            self.logger.info(f"board_change: {len(records)} records")
        except Exception as e:
            errors.append(f"board_change: {e}")

        # ---- 4. 板块资金流(主力动向) - 改用 "3日" indicator 拿 3 日累计 ----
        try:
            records = self.tools.get_sector_fund_flow(indicator="3日", sector_type="行业资金流")
            records = sorted(
                records,
                key=lambda x: float(x.get("主力净流入-净额") or 0),
                reverse=True,
            )[:20]
            for r in records:
                signals.append(self._from_sector_fund_flow(r))
            self.logger.info(f"sector_fund_flow(3日) top: {len(records)}")
        except Exception as e:
            errors.append(f"sector_fund_flow(3日): {e}")

        # ---- 5. 业绩预告(基本面异动) - 多个报告期 ----
        try:
            for date in self._recent_report_dates():
                records = self.tools.get_yjyg(date)
                for r in records[:30]:
                    signals.append(self._from_yjyg(r))
        except Exception as e:
            errors.append(f"yjyg: {e}")

        # ---- 6. 全局快讯(宏观/突发) - 当天 ----
        try:
            records = self.tools.get_global_news()
            for r in records[:20]:
                signals.append(self._from_global_news(r))
            self.logger.info(f"global_news: {len(records)} records")
        except Exception as e:
            errors.append(f"global_news: {e}")

        # 写入 Working Memory
        for s in signals:
            self.wm.append("signals", s.to_dict())

        return AgentResult(
            agent_name=self.name,
            success=len(signals) > 0,
            data={
                "since": since,
                "scan_at": datetime.now().isoformat(timespec="seconds"),
                "scan_dates": scan_dates,
            },
            items=[s.to_dict() for s in signals],
            metrics={
                "total_signals": len(signals),
                "by_type": self._count_by_type(signals),
                "scan_days": len(scan_dates),
                "scan_dates": scan_dates,
            },
            errors=errors,
        )

    def _recent_trade_dates(self, n: int) -> list[str]:
        """生成最近 N 个工作日(YYYYMMDD),跳过周末

        简单实现:从今天往前数,周一-五算工作日,周六日跳过
        注:生产应该对接交易日历(节假日),但当前先做简化
        """
        from datetime import timedelta
        dates = []
        d = datetime.now()
        while len(dates) < n:
            # weekday(): 周一=0, 周日=6
            if d.weekday() < 5:
                dates.append(d.strftime("%Y%m%d"))
            d -= timedelta(days=1)
        # 返回从早到晚的顺序
        return list(reversed(dates))
    
    def _recent_report_dates(self) -> list[str]:
        """最近几个已发布的报告期(避免未来日期)"""
        # 业绩预告按报告期发布,不能查未来
        # 当前是 2026 年,但 akshare 的业绩预告数据通常到 2024 年底
        # 真实使用时应该用今天日期回推
        candidates = [
            "20241231",  # 2024 年报
            "20240930",  # 2024 三季报
            "20240630",  # 2024 中报
            "20240331",  # 2024 一季报
        ]
        return candidates
    
    # ---- 各源到 Signal 的转换 ----
    
    def _from_limit_up(self, r: dict) -> Signal:
        return Signal(
            signal_type=SIGNAL_LIMIT_UP,
            symbol=str(r.get("代码", "")),
            name=str(r.get("名称", "")),
            board=str(r.get("所属行业", "")),
            timestamp=_now(),
            raw=r,
            score_hint=0.8,  # 涨停是强信号
        )
    
    def _from_limit_down(self, r: dict) -> Signal:
        return Signal(
            signal_type=SIGNAL_LIMIT_DOWN,
            symbol=str(r.get("代码", "")),
            name=str(r.get("名称", "")),
            board=str(r.get("所属行业", "")),
            timestamp=_now(),
            raw=r,
            score_hint=0.7,
        )
    
    def _from_board_change(self, r: dict) -> Signal:
        return Signal(
            signal_type=SIGNAL_BOARD_CHANGE,
            symbol="",
            name=str(r.get("板块名称", "")),
            board=str(r.get("板块名称", "")),
            timestamp=_now(),
            raw=r,
            score_hint=0.6,
        )
    
    def _from_sector_fund_flow(self, r: dict) -> Signal:
        return Signal(
            signal_type=SIGNAL_FUND_FLOW,
            symbol="",
            name=str(r.get("板块名称", "")),
            board=str(r.get("板块名称", "")),
            timestamp=_now(),
            raw=r,
            score_hint=0.5,
        )
    
    def _from_yjyg(self, r: dict) -> Signal:
        return Signal(
            signal_type=SIGNAL_ANNOUNCEMENT,
            symbol=str(r.get("股票代码", "")),
            name=str(r.get("股票名称", "")),
            board="",
            timestamp=_now(),
            raw=r,
            score_hint=0.5,
        )
    
    def _from_global_news(self, r: dict) -> Signal:
        return Signal(
            signal_type=SIGNAL_NEWS,
            symbol="",
            name=str(r.get("标题", r.get("内容", "")))[:50],
            board="",
            timestamp=_now(),
            raw=r,
            score_hint=0.4,
        )
    
    def _count_by_type(self, signals: list[Signal]) -> dict:
        from collections import Counter
        return dict(Counter(s.signal_type for s in signals))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
