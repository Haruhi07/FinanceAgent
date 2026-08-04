"""
Agent 基类
==========

所有 Agent 的统一接口:
- run(input_data) -> AgentResult
- log_event(...)
"""

from __future__ import annotations
import abc
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from memory import get_working_memory

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """所有 Agent 的统一输出格式"""
    agent_name: str
    success: bool
    data: Any = None                          # 输出数据(每个 Agent 类型不同)
    items: list = field(default_factory=list)  # 通用条目列表
    metrics: dict = field(default_factory=dict)  # 量化指标
    errors: list = field(default_factory=list)  # 错误信息
    duration_ms: int = 0
    
    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "data": self.data,
            "items_count": len(self.items),
            "metrics": self.metrics,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
        }


class BaseAgent(abc.ABC):
    name: str = "base"
    
    def __init__(self):
        self.wm = get_working_memory()
        self.logger = logging.getLogger(f"agent.{self.name}")
    
    @abc.abstractmethod
    def _run(self, input_data: Any) -> AgentResult:
        """子类必须实现"""
        pass
    
    def run(self, input_data: Any = None) -> AgentResult:
        """统一入口,带计时/异常处理/事件记录"""
        start = time.time()
        self.wm.log_event(self.name, "start", {"input": str(type(input_data))})
        try:
            result = self._run(input_data)
            result.agent_name = self.name
            result.duration_ms = int((time.time() - start) * 1000)
            self.wm.log_event(self.name, "done", result.to_dict())
            self.logger.info(
                f"done: {len(result.items)} items, {result.duration_ms}ms, "
                f"success={result.success}"
            )
            return result
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.logger.error(f"failed: {err}\n{traceback.format_exc()}")
            self.wm.log_event(self.name, "error", err)
            return AgentResult(
                agent_name=self.name,
                success=False,
                errors=[err],
                duration_ms=int((time.time() - start) * 1000),
            )
