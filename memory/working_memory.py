"""
Working Memory - 当前事件上下文
================================

会话级,内存里。Agent 协作时共享当前任务的全部状态。
"""

from __future__ import annotations
import threading
from typing import Any, Optional
from collections import deque
from datetime import datetime


class WorkingMemory:
    """线程安全的 Working Memory"""
    
    def __init__(self, max_history: int = 1000):
        self._lock = threading.Lock()
        self._state: dict = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "current_task": None,
            "signals": [],          # 当前发现的所有信号
            "anomalies": [],        # 检测到的异动
            "hotspot_candidates": [],
            "topics": [],
            "researches": [],
            "drafts": [],
            "reviews": [],
            "published": [],
        }
        self._events: deque = deque(maxlen=max_history)
    
    # ---- 状态读写 ----
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)
    
    def append(self, key: str, value: Any) -> None:
        with self._lock:
            if key not in self._state:
                self._state[key] = []
            self._state[key].append(value)
    
    # ---- 事件流 ----
    def log_event(self, agent_name: str, event_type: str, payload: Any = None) -> None:
        with self._lock:
            self._events.append({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "agent": agent_name,
                "event": event_type,
                "payload": payload,
            })
    
    def get_recent_events(self, n: int = 20) -> list:
        with self._lock:
            return list(self._events)[-n:]
    
    # ---- 快照 ----
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": dict(self._state),
                "events": list(self._events),
            }
    
    def summary(self) -> str:
        """生成可读摘要,给 Writer / Reviewer 看"""
        s = self._state
        return (
            f"会话: {s['session_id']}\n"
            f"信号数: {len(s.get('signals', []))}\n"
            f"异动数: {len(s.get('anomalies', []))}\n"
            f"候选热点: {len(s.get('hotspot_candidates', []))}\n"
            f"话题: {len(s.get('topics', []))}\n"
            f"研究: {len(s.get('researches', []))}\n"
            f"草稿: {len(s.get('drafts', []))}\n"
            f"已发布: {len(s.get('published', []))}\n"
        )


_wm_instance: Optional[WorkingMemory] = None
def get_working_memory() -> WorkingMemory:
    global _wm_instance
    if _wm_instance is None:
        _wm_instance = WorkingMemory()
    return _wm_instance
