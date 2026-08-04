"""
akshare 工具封装
================

核心设计:
1. 缓存:相同请求短时间内只跑一次,文件级持久化
2. 重试:网络异常自动重试,带退避
3. 标准化:把 akshare 各种"奇奇怪怪"的列名统一成内部 schema
4. 异常隔离:任何一个接口失败都不会拖垮整个流程
"""

from __future__ import annotations

import json
import time
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)

# 注意: 旧的 `_disable_proxy_for_akshare()` 已删除
# 代理控制现在统一由 tools.disable_proxy 处理(支持白名单)
# 使用方法: 在 main.py 入口处,根据 --no-proxy flag 决定是否调用 disable_proxy()

# 信号类型常量(内部用)
SIGNAL_PRICE_ANOMALY = "price_anomaly"
SIGNAL_VOLUME_ANOMALY = "volume_anomaly"
SIGNAL_LIMIT_UP = "limit_up"
SIGNAL_LIMIT_DOWN = "limit_down"
SIGNAL_BOARD_CHANGE = "board_change"
SIGNAL_LHB = "lhb"
SIGNAL_FUND_FLOW = "fund_flow"
SIGNAL_RESEARCH = "research"
SIGNAL_NEWS = "news"
SIGNAL_ANNOUNCEMENT = "announcement"


@dataclass
class Signal:
    """统一的信号数据结构 - 所有 Scanner 输出的信号都长这样"""
    signal_type: str            # 见上方常量
    symbol: str = ""            # 股票代码,如 "600519"
    name: str = ""              # 名称,如 "贵州茅台"
    board: str = ""             # 所属行业/板块
    timestamp: str = ""         # ISO 时间
    raw: dict = field(default_factory=dict)  # 原始 payload
    score_hint: float = 0.0     # 来源给出的"强度提示",后续 Detector 还会再算

    def to_dict(self) -> dict:
        return asdict(self)


class _FileCache:
    """简单的文件缓存 - 避免重复请求 akshare"""
    def __init__(self, cache_dir: Path, ttl_map: dict):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl_map = ttl_map

    def _key(self, name: str, params: Any) -> str:
        s = json.dumps({"name": name, "params": str(params)}, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()

    def get(self, name: str, params: Any, ttl_category: str = "intraday") -> Optional[Any]:
        p = self.dir / f"{name}__{self._key(name, params)}.json"
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > self.ttl_map.get(ttl_category, 300):
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, name: str, params: Any, value: Any) -> None:
        p = self.dir / f"{name}__{self._key(name, params)}.json"
        try:
            p.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning(f"cache write failed: {e}")


def _retry(func, *args, **kwargs):
    """带指数退避的重试"""
    last_err = None
    for i in range(config.REQUEST_MAX_RETRY):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            if i < config.REQUEST_MAX_RETRY - 1:
                delay = config.REQUEST_RETRY_DELAY * (2 ** i)
                logger.warning(
                    f"{func.__name__} failed (attempt {i+1}/{config.REQUEST_MAX_RETRY}): {e}. "
                    f"retry in {delay}s"
                )
                time.sleep(delay)
    raise last_err


def _df_to_records(df: Optional[pd.DataFrame]) -> list:
    """DataFrame -> 字典列表,容错处理"""
    if df is None or df.empty:
        return []
    # 替换 NaN
    df = df.astype(object).where(pd.notnull(df), None)
    return df.to_dict(orient="records")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


class AkShareTools:
    """akshare 接口的统一封装
    
    所有方法都返回 list[dict],失败返回空列表(不抛异常到上层)
    """

    def __init__(self):
        self._cache = _FileCache(config.CACHE_DIR, config.CACHE_TTL_SECONDS)
        # 延迟导入,启动更快 + 避免 akshare 启动时副作用
        import akshare as ak
        self._ak = ak

    # ========== 异动类(主动发现的核心) ==========

    def get_limit_up_pool(self, date: Optional[str] = None) -> list[dict]:
        """涨停股池 - 主动发现'最热'信号的最快途径"""
        date = date or _today()
        cached = self._cache.get("limit_up", date, "intraday")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_zt_pool_em, date=date)
            records = _df_to_records(df)
            self._cache.set("limit_up", date, records)
            return records
        except Exception as e:
            logger.error(f"get_limit_up_pool({date}) failed: {e}")
            return []

    def get_limit_down_pool(self, date: Optional[str] = None) -> list[dict]:
        """跌停股池(部分 akshare 版本接口名不同)"""
        date = date or _today()
        cached = self._cache.get("limit_down", date, "intraday")
        if cached is not None:
            return cached
        # 兼容不同版本的接口名
        for fn_name in ["stock_zt_pool_dtgc_em", "stock_dt_pool_em", "stock_zt_pool_zbgc_em"]:
            fn = getattr(self._ak, fn_name, None)
            if fn is None:
                continue
            try:
                df = _retry(fn, date=date)
                records = _df_to_records(df)
                self._cache.set("limit_down", date, records)
                return records
            except Exception as e:
                logger.warning(f"{fn_name}({date}) failed: {e}")
        return []

    def get_board_change(self) -> list[dict]:
        """板块异动详情 - 板块级共振信号"""
        cached = self._cache.get("board_change", "today", "intraday")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_board_change_em)
            records = _df_to_records(df)
            self._cache.set("board_change", "today", records)
            return records
        except Exception as e:
            logger.error(f"get_board_change failed: {e}")
            return []

    def get_changes_realtime(self, symbol: str = "大笔买入") -> list[dict]:
        """盘口异动 - 大单买卖、火箭发射、跌停打开等"""
        cached = self._cache.get("changes_rt", symbol, "realtime")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_changes_em, symbol=symbol)
            records = _df_to_records(df)
            self._cache.set("changes_rt", symbol, records)
            return records
        except Exception as e:
            logger.error(f"get_changes_realtime({symbol}) failed: {e}")
            return []

    # ========== 资金流向 ==========

    def get_sector_fund_flow(self, indicator: str = "今日", sector_type: str = "行业资金流") -> list[dict]:
        """板块资金流 - 看主力在买什么"""
        # 检查是否在禁用列表
        if "get_sector_fund_flow" in getattr(config, "DISABLED_TOOLS", set()):
            return []
        cached = self._cache.get("sector_ff", f"{indicator}_{sector_type}", "intraday")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_sector_fund_flow_rank, indicator=indicator, sector_type=sector_type)
            records = _df_to_records(df)
            self._cache.set("sector_ff", f"{indicator}_{sector_type}", records)
            return records
        except Exception as e:
            logger.error(f"get_sector_fund_flow failed: {e}")
            return []

    def get_individual_fund_flow(self, stock: str, market: str = "sh") -> list[dict]:
        """个股资金流 - 最近 100 个交易日"""
        cached = self._cache.get("individual_ff", f"{stock}_{market}", "daily")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_individual_fund_flow, stock=stock, market=market)
            records = _df_to_records(df)
            self._cache.set("individual_ff", f"{stock}_{market}", records)
            return records
        except Exception as e:
            logger.error(f"get_individual_fund_flow({stock}) failed: {e}")
            return []

    # ========== 龙虎榜 ==========

    def get_lhb_detail(self, start_date: str, end_date: str) -> list[dict]:
        """龙虎榜详情 - 看机构动向"""
        cached = self._cache.get("lhb", f"{start_date}_{end_date}", "daily")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_lhb_detail_em, start_date=start_date, end_date=end_date)
            records = _df_to_records(df)
            self._cache.set("lhb", f"{start_date}_{end_date}", records)
            return records
        except Exception as e:
            logger.error(f"get_lhb_detail failed: {e}")
            return []

    # ========== 行情数据 ==========

    def get_zh_a_spot(self) -> list[dict]:
        """全 A 实时行情 - 用于异动检测"""
        cached = self._cache.get("zh_a_spot", "today", "realtime")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_zh_a_spot_em)
            records = _df_to_records(df)
            self._cache.set("zh_a_spot", "today", records)
            return records
        except Exception as e:
            logger.error(f"get_zh_a_spot failed: {e}")
            return []

    def get_zh_a_hist(self, symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> list[dict]:
        """个股历史 K 线 - 用于计算均线、波动率等"""
        cached = self._cache.get("zh_a_hist", f"{symbol}_{start_date}_{end_date}_{adjust}", "daily")
        if cached is not None:
            return cached
        try:
            df = _retry(
                self._ak.stock_zh_a_hist,
                symbol=symbol, period="daily",
                start_date=start_date, end_date=end_date, adjust=adjust,
            )
            records = _df_to_records(df)
            self._cache.set("zh_a_hist", f"{symbol}_{start_date}_{end_date}_{adjust}", records)
            return records
        except Exception as e:
            logger.error(f"get_zh_a_hist({symbol}) failed: {e}")
            return []

    def get_individual_info(self, symbol: str) -> dict:
        """个股基本信息"""
        cached = self._cache.get("indiv_info", symbol, "fundamental")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_individual_info_em, symbol=symbol)
            records = _df_to_records(df)
            info = {r.get("item"): r.get("value") for r in records}
            self._cache.set("indiv_info", symbol, info)
            return info
        except Exception as e:
            logger.error(f"get_individual_info({symbol}) failed: {e}")
            return {}

    # ========== 财务 / 公告 ==========

    def get_yjyg(self, date: str) -> list[dict]:
        """业绩预告 - 提前发现基本面异动"""
        cached = self._cache.get("yjyg", date, "daily")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_yjyg_em, date=date)
            records = _df_to_records(df)
            self._cache.set("yjyg", date, records)
            return records
        except Exception as e:
            logger.error(f"get_yjyg({date}) failed: {e}")
            return []

    def get_yjbb(self, date: str) -> list[dict]:
        """业绩报表"""
        cached = self._cache.get("yjbb", date, "daily")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_yjbb_em, date=date)
            records = _df_to_records(df)
            self._cache.set("yjbb", date, records)
            return records
        except Exception as e:
            logger.error(f"get_yjbb({date}) failed: {e}")
            return []

    def get_financial_report(self, stock: str, report_type: str = "利润表") -> list[dict]:
        """三大报表 - 利润表/资产负债表/现金流量表"""
        cached = self._cache.get("fin_report", f"{stock}_{report_type}", "fundamental")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_financial_report_sina, stock=stock, symbol=report_type)
            records = _df_to_records(df)
            self._cache.set("fin_report", f"{stock}_{report_type}", records)
            return records
        except Exception as e:
            logger.error(f"get_financial_report({stock}, {report_type}) failed: {e}")
            return []

    def get_gsdt(self, symbol: str = "") -> list[dict]:
        """公司动态 - 公告类(按日期,不是按 symbol)"""
        # stock_gsrl_gsdt_em 在新版本里改成了按日期查询
        # 所以这里返回的是当日所有公司动态
        date = _today()
        cached = self._cache.get("gsdt", date, "daily")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_gsrl_gsdt_em, date=date)
            records = _df_to_records(df)
            self._cache.set("gsdt", date, records)
            return records
        except Exception as e:
            logger.error(f"get_gsdt({date}) failed: {e}")
            return []

    def get_research_report(self, symbol: str) -> list[dict]:
        """个股研报
        注: stock_research_report_em 1.18+ 需要传中文股票名(不是 symbol)
        """
        cached = self._cache.get("research", symbol, "daily")
        if cached is not None:
            return cached
        try:
            # 先查个股信息获取中文名
            info = self.get_individual_info(symbol)
            name = info.get("股票简称", symbol)
            df = _retry(self._ak.stock_research_report_em, symbol=name)
            records = _df_to_records(df)
            self._cache.set("research", symbol, records)
            return records
        except Exception as e:
            logger.warning(f"get_research_report({symbol}) failed: {e}")
            return []

    def get_news(self, symbol: str) -> list[dict]:
        """个股新闻"""
        cached = self._cache.get("news", symbol, "intraday")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_news_em, symbol=symbol)
            records = _df_to_records(df)
            self._cache.set("news", symbol, records)
            return records
        except Exception as e:
            logger.error(f"get_news({symbol}) failed: {e}")
            return []

    # ========== 行业 / 板块 / 宏观 ==========

    def get_industry_boards(self) -> list[dict]:
        """行业板块列表"""
        cached = self._cache.get("industry_boards", "all", "fundamental")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_board_industry_name_em)
            records = _df_to_records(df)
            self._cache.set("industry_boards", "all", records)
            return records
        except Exception as e:
            logger.error(f"get_industry_boards failed: {e}")
            return []

    def get_global_news(self) -> list[dict]:
        """全球财经快讯"""
        cached = self._cache.get("global_news", "latest", "realtime")
        if cached is not None:
            return cached
        try:
            df = _retry(self._ak.stock_info_global_em)
            records = _df_to_records(df)
            self._cache.set("global_news", "latest", records)
            return records
        except Exception as e:
            logger.error(f"get_global_news failed: {e}")
            return []


# 单例
_instance: Optional[AkShareTools] = None


def get_tools() -> AkShareTools:
    global _instance
    if _instance is None:
        _instance = AkShareTools()
    return _instance
