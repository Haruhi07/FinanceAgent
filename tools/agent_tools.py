"""
Tool Registry - 把所有可调用的"工具"统一管理
================================================

2026-08 改造:在原 13 个金融数据工具基础上,新增 3 个子 Agent 工具
(让 Orchestrator 可以通过 function-call 调 Researcher/Writer/Reviewer)。

设计:
- 金融数据工具(build_finance_tools):14 个 akshare 接口的封装,LLM 用
- 子 Agent 工具(build_subagent_tools):3 个,R/W/R 子 Agent,Orchestrator 用
- 统一通过 ToolRegistry 管理,to_openai_tools() 暴露给 LLM
"""
from __future__ import annotations
import asyncio
import json
import logging
import re
from typing import Any, Callable, Optional

import config
from .akshare_tools import get_tools, _df_to_records

logger = logging.getLogger(__name__)


def _truncate_list(result, max_items=20):
    """对返回的 list 截断,保留前 max_items 条 + 标记"""
    if isinstance(result, list) and len(result) > max_items:
        return result[:max_items] + [{"_truncated": f"original {len(result)} items, kept first {max_items}"}]
    return result


# ============= Tool / ToolRegistry =============

class Tool:
    """单个工具定义"""
    def __init__(self, name: str, description: str, parameters: dict,
                 handler: Optional[Callable] = None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表"""
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self, names: list = None) -> list[str]:
        if names is None:
            tools = self._tools.values()
        else:
            tools = [self._tools[n] for n in names if n in self._tools]
        return [t.name for t in tools]

    def to_openai_tools(self, names: list = None) -> list[dict]:
        if names is None:
            tools = self._tools.values()
        else:
            tools = [self._tools[n] for n in names if n in self._tools]
        return [t.to_openai_tool() for t in tools]

    async def execute(self, name: str, arguments: dict) -> dict:
        """执行工具,返回 JSON-friendly 的结果"""
        tool = self.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name}"}
        if tool.handler is None:
            return {"error": f"tool {name} has no handler"}
        try:
            logger.info(f"executing tool: {name}({arguments})")
            result = tool.handler(**arguments)
            # 限制结果大小,避免 LLM context 被撑爆
            result_str = json.dumps(result, ensure_ascii=False, default=str)
            if len(result_str) > 50000:
                if isinstance(result, list) and len(result) > 20:
                    result = result[:20] + [{"_truncated": f"original {len(result_str)} chars, kept first 20 items"}]
            return {"success": True, "data": result}
        except Exception as e:
            logger.error(f"tool {name} failed: {e}")
            return {"success": False, "error": str(e)}


# ============= Singleton =============
_registry_instance: Optional[ToolRegistry] = None

def get_tool_registry() -> ToolRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = build_combined_registry()
    return _registry_instance


def build_combined_registry() -> ToolRegistry:
    """构建完整工具注册表(金融工具 + 子 Agent 工具 + DUMMY 工具)"""
    reg = ToolRegistry()
    # 1. 14 个金融数据工具(2026-08 加回 get_financial_report)
    for t in build_finance_tools():
        reg.register(t)
    # 2. 3 个子 Agent 工具
    for t in build_subagent_tools_as_tool_list():
        reg.register(t)
    # 3. 3 个 DUMMY 工具(2026-08 新增,标 dummy_ 前缀)
    try:
        from .dummy_tools import build_dummy_tools
        for t in build_dummy_tools():
            reg.register(t)
    except ImportError as e:
        logger.warning(f"DUMMY 工具未加载: {e}")
    return reg


# ============= 1. 金融数据工具(13 个) =============

def build_finance_tools() -> list[Tool]:
    """构建金融领域工具 - 14 个 akshare 接口(2026-08 加回 get_financial_report)"""
    registry = []
    ak = get_tools()
    disabled = set(getattr(config, "DISABLED_TOOLS", []))

    def reg_if(t: Tool):
        if t.name not in disabled:
            registry.append(t)

    # ---- 1. 涨停股池 ----
    reg_if(Tool(
        name="get_limit_up_pool",
        description="获取某日(YYYYMMDD)涨停股池,返回每只涨停股的代码、名称、所属行业、涨跌幅等。",
        parameters={
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "日期 YYYYMMDD,默认当天"},
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=lambda date=None: _truncate_list(ak.get_limit_up_pool(date), 30),
    ))

    # ---- 2. 跌停股池 ----
    reg_if(Tool(
        name="get_limit_down_pool",
        description="获取某日跌停股池。",
        parameters={
            "type": "object",
            "properties": {"date": {"type": "string", "description": "日期 YYYYMMDD"}},
            "required": [],
            "additionalProperties": False,
        },
        handler=lambda date=None: _truncate_list(ak.get_limit_down_pool(date), 30),
    ))

    # ---- 3. 板块异动 ----
    reg_if(Tool(
        name="get_board_change",
        description="获取当日板块异动(板块涨跌幅、异动次数、主力资金)。",
        parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        handler=lambda: _truncate_list(ak.get_board_change(), 20),
    ))

    # ---- 4. 板块资金流 ----
    reg_if(Tool(
        name="get_sector_fund_flow",
        description="获取板块资金流(按时间段累计:今日/3日/5日/10日)。",
        parameters={
            "type": "object",
            "properties": {
                "indicator": {"type": "string", "enum": ["今日", "3日", "5日", "10日"], "description": "时间段"},
                "sector_type": {"type": "string", "enum": ["行业资金流", "概念资金流", "地域资金流"]},
            },
            "required": ["indicator"],
            "additionalProperties": False,
        },
        handler=lambda indicator="今日", sector_type="行业资金流":
            _truncate_list(ak.get_sector_fund_flow(indicator, sector_type)),
    ))

    # ---- 5. 个股资金流(disable 仍保留注册以便回滚)----
    reg_if(Tool(
        name="get_individual_fund_flow",
        description="获取个股最近 100 个交易日资金流(净额、主力净额等)。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码"},
                "market": {"type": "string", "enum": ["sh", "sz"], "description": "市场"},
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
        handler=lambda symbol="", market="sh":
            _truncate_list(ak.get_individual_fund_flow(symbol, market), 30),
    ))

    # ---- 6. 全 A 实时行情 ----
    reg_if(Tool(
        name="get_zh_a_spot",
        description="获取全 A 实时行情(5000+ 只股票的最新价、涨跌幅、量比)。",
        parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        handler=lambda: _truncate_list(ak.get_zh_a_spot(), 50),
    ))

    # ---- 7. 个股基本信息 ----
    reg_if(Tool(
        name="get_individual_info",
        description="获取个股基本信息(股票简称、行业、总股本、流通股本等)。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "pattern": r"^\d{6}$", "description": "股票代码"},
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
        handler=lambda symbol="": ak.get_individual_info(symbol),
    ))

    # ---- 8. 业绩预告 ----
    reg_if(Tool(
        name="get_yjyg",
        description="按报告期(YYYYMMDD)获取业绩预告。",
        parameters={
            "type": "object",
            "properties": {
                "date": {"type": "string", "pattern": r"^\d{8}$", "description": "报告期 YYYYMMDD"},
            },
            "required": ["date"],
            "additionalProperties": False,
        },
        handler=lambda date="": _truncate_list(ak.get_yjyg(date), 30),
    ))

    # ---- 9. 三大报表(2026-08 解禁)----
    reg_if(Tool(
        name="get_financial_report",
        description="获取上市公司三大报表数据(资产负债表/利润表/现金流量表)。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "pattern": r"^\d{6}$", "description": "股票代码"},
                "report_type": {
                    "type": "string",
                    "enum": ["资产负债表", "利润表", "现金流量表"],
                },
            },
            "required": ["symbol", "report_type"],
            "additionalProperties": False,
        },
        handler=lambda symbol="", report_type="利润表":
            _truncate_list(ak.get_financial_report(symbol, report_type), 20),
    ))

    # ---- 10. 个股研报(akshare bug,disabled)----
    reg_if(Tool(
        name="get_research_report",
        description="获取个股研报列表(报告名称、机构、评级)。注意:akshare 1.18 有 KeyError bug,可能调用失败。",
        parameters={
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "股票代码或中文名"}},
            "required": ["symbol"],
            "additionalProperties": False,
        },
        handler=lambda symbol="": _truncate_list(ak.get_research_report(symbol), 5),
    ))

    # ---- 11. 个股新闻 ----
    reg_if(Tool(
        name="get_news",
        description="获取个股相关新闻(标题、内容、发布时间、来源)。",
        parameters={
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "股票代码或中文名"}},
            "required": ["symbol"],
            "additionalProperties": False,
        },
        handler=lambda symbol="": _truncate_list(ak.get_news(symbol), 10),
    ))

    # ---- 12. 全球快讯 ----
    reg_if(Tool(
        name="get_global_news",
        description="获取全球财经快讯(标题、摘要、链接)。",
        parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        handler=lambda: _truncate_list(ak.get_global_news(), 20),
    ))

    # ---- 13. 历史 K 线 ----
    reg_if(Tool(
        name="get_zh_a_hist",
        description="获取个股历史 K 线(OHLCV)。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "pattern": r"^\d{6}$"},
                "start_date": {"type": "string", "pattern": r"^\d{8}$"},
                "end_date": {"type": "string", "pattern": r"^\d{8}$"},
                "adjust": {"type": "string", "enum": ["qfq", "hfq", ""], "description": "复权方式"},
            },
            "required": ["symbol", "start_date", "end_date"],
            "additionalProperties": False,
        },
        handler=lambda symbol="", start_date="", end_date="", adjust="qfq":
            _truncate_list(ak.get_zh_a_hist(symbol, start_date, end_date, adjust), 30),
    ))

    return registry


# ============= 2. 子 Agent 工具(3 个,2026-08 新增) =============

def build_subagent_tools_as_tool_list() -> list[Tool]:
    """把 sub-agent tools 转成 Tool 列表,加到注册表"""
    from . import agent_subagent_tools as sub
    sub_dict = sub.build_subagent_tools()
    out = []
    for t in sub_dict.values():
        out.append(Tool(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"],
            handler=t["handler"],
        ))
    return out
