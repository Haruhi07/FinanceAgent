"""
工具层:对 akshare 接口的封装
============================

所有外部数据访问走这里,提供:
1. 统一的异常处理 + 重试
2. 缓存(基于文件)
3. 数据标准化(列名、类型)

不要在 Agent 里直接 import akshare,统一走这个模块。
"""

from .akshare_tools import (
    AkShareTools,
    get_tools,
    Signal,
    # 数据类型常量
    SIGNAL_PRICE_ANOMALY,
    SIGNAL_VOLUME_ANOMALY,
    SIGNAL_LIMIT_UP,
    SIGNAL_LIMIT_DOWN,
    SIGNAL_BOARD_CHANGE,
    SIGNAL_LHB,
    SIGNAL_FUND_FLOW,
    SIGNAL_RESEARCH,
    SIGNAL_NEWS,
    SIGNAL_ANNOUNCEMENT,
)
from .llm_client import (
    LLMClient,
    get_llm,
    chat,
    chat_json,
    is_llm_available,
)
from .agent_tools import (
    Tool,
    ToolRegistry,
    get_tool_registry,
    build_finance_tools,
)

__all__ = [
    "AkShareTools",
    "get_tools",
    "Signal",
    "SIGNAL_PRICE_ANOMALY",
    "SIGNAL_VOLUME_ANOMALY",
    "SIGNAL_LIMIT_UP",
    "SIGNAL_LIMIT_DOWN",
    "SIGNAL_BOARD_CHANGE",
    "SIGNAL_LHB",
    "SIGNAL_FUND_FLOW",
    "SIGNAL_RESEARCH",
    "SIGNAL_NEWS",
    "SIGNAL_ANNOUNCEMENT",
    # LLM
    "LLMClient",
    "get_llm",
    "chat",
    "chat_json",
    "is_llm_available",
    # Agent Tools
    "Tool",
    "ToolRegistry",
    "get_tool_registry",
    "build_finance_tools",
]
