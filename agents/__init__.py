"""
Agent 基类 + 6 个核心 Agent
===========================

按"能力边界"划分,而不是按"流程阶段"划分。
"""

from .base import BaseAgent, AgentResult
from .scanner import ScannerAgent
from .anomaly_detector import AnomalyDetector
from .hotspot_detector import HotspotDetector
from .topic_modeler import TopicModeler
from .researcher import ResearchAgent
from .writer import WriterAgent
from .reviewer import ReviewerAgent
from .orchestrator import Orchestrator

__all__ = [
    "BaseAgent", "AgentResult",
    "ScannerAgent",
    "AnomalyDetector",
    "HotspotDetector",
    "TopicModeler",
    "ResearchAgent",
    "WriterAgent",
    "ReviewerAgent",
    "Orchestrator",
]
