"""
记忆层:Agent 系统的灵魂
========================

三层记忆结构(对应方案设计):
1. Working Memory  - 当前事件上下文(内存)
2. Episodic Memory - 历史热点案例(JSON 文件,可检索)
3. Semantic Memory - 行业知识 / 读者偏好(内存 dict)
"""

from .working_memory import WorkingMemory, get_working_memory
from .episodic_memory import EpisodicMemory, get_episodic_memory
from .semantic_memory import SemanticMemory, get_semantic_memory

__all__ = [
    "WorkingMemory", "get_working_memory",
    "EpisodicMemory", "get_episodic_memory",
    "SemanticMemory", "get_semantic_memory",
]
