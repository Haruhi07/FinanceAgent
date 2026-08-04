"""
Hotspot Detector - 热点识别 + 7 维评分
=======================================

核心:把 anomalies 转化为带 7 维特征向量的候选热点

H = (I, P, V, N, R, D, L)
- I: Intensity 强度
- P: Persistence 持续性
- V: Virality 传播性
- N: Novelty 新颖度
- R: Relevance 主体关联性
- D: Value Density 价值密度
- L: Lead Time 提前量

Score(H) = w^T * H + b, w 在 config.WEIGHTS
"""

from __future__ import annotations
import logging
from datetime import datetime
from collections import Counter

from .base import BaseAgent, AgentResult
from memory import get_episodic_memory, get_semantic_memory
from tools.persist import save_candidates
import config

logger = logging.getLogger(__name__)


class HotspotDetector(BaseAgent):
    name = "hotspot_detector"
    
    def __init__(self):
        super().__init__()
        self.em = get_episodic_memory()
        self.sm = get_semantic_memory()
    
    def _run(self, input_data: dict | None) -> AgentResult:
        anomalies = self.wm.get("anomalies", [])
        if not anomalies:
            return AgentResult(
                agent_name=self.name, success=False,
                errors=["no anomalies in working memory"],
            )
        
        # ---- 1. 把 anomalies 按"主体"聚类 ----
        grouped = self._group_by_subject(anomalies)
        
        # ---- 2. 对每个组,提取 7 维特征并打分 ----
        candidates = []
        for subject, group in grouped.items():
            candidate = self._build_candidate(subject, group)
            candidates.append(candidate)
        
        # ---- 3. 按分数排序,过滤低分 ----
        candidates.sort(key=lambda x: x["score"], reverse=True)
        filtered = [c for c in candidates if c["score"] >= config.HOTSPOT_LOW_CONF]
        
        # ---- 4. 写入 WM + Episodic Memory ----
        for c in filtered:
            self.wm.append("hotspot_candidates", c)
            # 记入案例库(无反馈,等后续人工补充)
            self.em.add_case(c)

        # 2026-08 新增:落盘 JSON 供 Orchestrator 读 + 留 trace
        path = save_candidates(filtered, extra={
            "total_candidates": len(candidates),
            "passed_threshold": len(filtered),
            "source": "hotspot_detector",
        })

        return AgentResult(
            agent_name=self.name,
            success=len(filtered) > 0,
            items=filtered,
            metrics={
                "total_candidates": len(candidates),
                "passed_threshold": len(filtered),
                "high_conf": sum(1 for c in filtered if c["score"] >= config.HOTSPOT_HIGH_CONF),
                "mid_conf": sum(1 for c in filtered if config.HOTSPOT_MID_CONF <= c["score"] < config.HOTSPOT_HIGH_CONF),
                "low_conf": sum(1 for c in filtered if config.HOTSPOT_LOW_CONF <= c["score"] < config.HOTSPOT_MID_CONF),
                "top_score": filtered[0]["score"] if filtered else 0,
                "json_path": str(path),
            },
        )
    
    def _group_by_subject(self, anomalies: list[dict]) -> dict[str, list[dict]]:
        """按板块/主体聚类"""
        grouped: dict[str, list[dict]] = {}
        for a in anomalies:
            subj = a.get("board") or a.get("name", "unknown")
            if subj not in grouped:
                grouped[subj] = []
            grouped[subj].append(a)
        return grouped
    
    def _build_candidate(self, subject: str, group: list[dict]) -> dict:
        """构建单个候选热点"""
        # ---- 7 维特征 ----
        
        # I: Intensity - 强度(基于异常类型 + count)
        anomaly_types = [a.get("anomaly_type", "") for a in group]
        type_scores = {
            "change_with_limitup": 1.0,
            "board_change_with_fundflow": 0.9,
            "board_resonance": 0.7,
            "risk_concentration": 0.5,
        }
        intensity = max(
            (type_scores.get(t, 0.4) for t in anomaly_types),
            default=0.3,
        )
        # 加上信号量的加成
        signal_count = sum(a.get("limit_up_count", a.get("change_count", 1)) for a in group)
        intensity = min(intensity + min(signal_count, 10) * 0.02, 1.0)
        
        # P: Persistence - 持续性(原型里用 anomaly 数量近似,生产应该看时间序列)
        persistence = min(len(group) * 0.3, 1.0)
        
        # V: Virality - 传播性(是否跨多个异常类型)
        virality = min(len(set(anomaly_types)) * 0.4, 1.0)
        
        # N: Novelty - 新颖度(对比历史案例库)
        similar = self.em.find_similar({"topic": subject, "board": subject}, top_k=3)
        novelty = max(0.0, 1.0 - len(similar) * 0.2)
        
        # R: Relevance - 主体关联性(是否在我们的行业知识图谱里)
        industry = self.sm.find_industry_by_keyword(subject)
        relevance = 0.9 if industry else 0.5
        # 政策敏感行业加分
        if industry:
            ind_info = self.sm.get_industry(industry) or {}
            if ind_info.get("policy_sensitive"):
                relevance = min(relevance + 0.1, 1.0)
        
        # D: Value Density - 价值密度(基于异常类型,这里简化)
        value_density = {
            "change_with_limitup": 0.9,
            "board_change_with_fundflow": 0.85,
            "board_resonance": 0.7,
            "risk_concentration": 0.6,
        }.get(anomaly_types[0] if anomaly_types else "", 0.5)
        
        # L: Lead Time - 提前量(原型里给一个基础分,生产应该从历史数据学)
        lead_time = 0.6  # 默认中等
        
        # ---- 综合评分 ----
        weights = config.WEIGHTS
        score = (
            weights["intensity"] * intensity +
            weights["persistence"] * persistence +
            weights["virality"] * virality +
            weights["novelty"] * novelty +
            weights["relevance"] * relevance +
            weights["value_density"] * value_density +
            weights["lead_time"] * lead_time
        )
        
        # 风险信号降权
        if "risk_concentration" in anomaly_types:
            score *= 0.7
        
        # 描述拼接
        description = " | ".join(a.get("description", "") for a in group[:3])
        
        # 确信度
        if score >= config.HOTSPOT_HIGH_CONF:
            confidence = "high"
        elif score >= config.HOTSPOT_MID_CONF:
            confidence = "mid"
        else:
            confidence = "low"
        
        return {
            "candidate_id": f"cand_{int(datetime.now().timestamp())}_{hash(subject) % 10000}",
            "subject": subject,
            "industry": industry,
            "anomaly_types": anomaly_types,
            "anomaly_count": len(group),
            "features": {
                "intensity": round(intensity, 3),
                "persistence": round(persistence, 3),
                "virality": round(virality, 3),
                "novelty": round(novelty, 3),
                "relevance": round(relevance, 3),
                "value_density": round(value_density, 3),
                "lead_time": round(lead_time, 3),
            },
            "score": round(score, 3),
            "confidence": confidence,
            "description": description,
            "related_symbols": self._collect_symbols(group),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
    
    def _collect_symbols(self, group: list[dict]) -> list[str]:
        syms = []
        for a in group:
            syms.extend(a.get("symbols", []))
        return list(set(syms))[:20]
