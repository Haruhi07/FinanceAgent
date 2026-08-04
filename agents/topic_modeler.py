"""
Topic Modeler - 话题建模
==========================

职责:把散乱信号聚合成结构化话题卡片

输出:
- 话题标题
- 关键事件/数据
- 涉及个股
- 话题摘要
- 关联板块/行业
"""

from __future__ import annotations
import logging
from datetime import datetime
from collections import Counter

from .base import BaseAgent, AgentResult
from memory import get_semantic_memory

logger = logging.getLogger(__name__)


class TopicModeler(BaseAgent):
    name = "topic_modeler"
    
    def __init__(self):
        super().__init__()
        self.sm = get_semantic_memory()
    
    def _run(self, input_data: dict | None) -> AgentResult:
        candidates = self.wm.get("hotspot_candidates", [])
        if not candidates:
            return AgentResult(
                agent_name=self.name, success=False,
                errors=["no hotspot candidates"],
            )
        
        topics = []
        for cand in candidates:
            topic = self._build_topic(cand)
            topics.append(topic)
            self.wm.append("topics", topic)
        
        return AgentResult(
            agent_name=self.name,
            success=len(topics) > 0,
            items=topics,
            metrics={"topic_count": len(topics)},
        )
    
    def _build_topic(self, candidate: dict) -> dict:
        """从候选热点构建话题卡片"""
        subject = candidate["subject"]
        industry = candidate.get("industry")
        
        # 涉及个股
        symbols = candidate.get("related_symbols", [])
        
        # 话题模板
        if "risk_concentration" in candidate.get("anomaly_types", []):
            template = "risk"
        elif "change_with_limitup" in candidate.get("anomaly_types", []):
            template = "sector_breakout"
        elif "board_change_with_fundflow" in candidate.get("anomaly_types", []):
            template = "fund_driven"
        else:
            template = "general"
        
        # 标题生成(模板化)
        title = self._generate_title(subject, candidate, template)
        
        # 摘要
        summary = self._generate_summary(subject, candidate, symbols, industry)
        
        # 关键要点
        key_points = self._extract_key_points(candidate, symbols)
        
        # 时间线
        timeline = self._build_timeline(candidate)
        
        return {
            "topic_id": f"topic_{int(datetime.now().timestamp())}_{hash(subject) % 10000}",
            "candidate_id": candidate["candidate_id"],
            "subject": subject,
            "industry": industry,
            "title": title,
            "summary": summary,
            "key_points": key_points,
            "timeline": timeline,
            "symbols": symbols,
            "anomaly_types": candidate.get("anomaly_types", []),
            "score": candidate.get("score", 0),
            "confidence": candidate.get("confidence", "low"),
            "template": template,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
    
    def _generate_title(self, subject: str, candidate: dict, template: str) -> str:
        """根据模板生成标题"""
        anom = candidate.get("anomaly_types", [])
        score = candidate.get("score", 0)
        
        if template == "sector_breakout":
            lu = sum(a.get("limit_up_count", 0) for a in [candidate] if isinstance(a, dict))
            return f"{subject} 板块强势爆发,{len(candidate.get('related_symbols', []))} 只个股涨停,资金加速涌入"
        elif template == "fund_driven":
            return f"主力资金抢筹 {subject},板块异动频繁,后市怎么看?"
        elif template == "risk":
            return f"{subject} 板块多股跌停,警惕风险信号"
        else:
            return f"{subject} 板块异动观察"
    
    def _generate_summary(self, subject: str, candidate: dict, symbols: list, industry: str | None) -> str:
        feats = candidate.get("features", {})
        anom = candidate.get("anomaly_types", [])
        
        parts = [
            f"{subject} 板块今日出现明显异动。",
        ]
        if "change_with_limitup" in anom:
            parts.append(f"板块异动频繁,同时伴随多只个股涨停,显示资金做多意愿强烈。")
        if "board_change_with_fundflow" in anom:
            parts.append(f"主力资金同步净流入,资金驱动特征明显。")
        if "board_resonance" in anom:
            parts.append(f"板块内多只个股共振,形成板块效应。")
        if industry:
            ind_info = self.sm.get_industry(industry) or {}
            if ind_info.get("policy_sensitive"):
                parts.append(f"{industry} 行业受政策影响较大,需关注后续政策走向。")
        parts.append(f"综合评分 {candidate.get('score', 0):.2f},确信度:{candidate.get('confidence', 'low')}。")
        
        return "".join(parts)
    
    def _extract_key_points(self, candidate: dict, symbols: list) -> list[str]:
        """提取关键要点"""
        points = []
        anom = candidate.get("anomaly_types", [])
        
        if "change_with_limitup" in anom:
            points.append("板块异动频繁 + 多只涨停,共振信号强")
        if "board_change_with_fundflow" in anom:
            points.append("主力资金净流入,资金面支持")
        if "board_resonance" in anom:
            points.append("板块内多股共振,形成板块效应")
        if symbols:
            points.append(f"涉及个股:{', '.join(symbols[:5])}")
        
        feats = candidate.get("features", {})
        if feats.get("novelty", 0) > 0.7:
            points.append("历史上较少出现的板块异动,新颖度高")
        if feats.get("relevance", 0) > 0.8:
            points.append("属于重点关注行业,关联性高")
        
        return points
    
    def _build_timeline(self, candidate: dict) -> list[dict]:
        """构建简单时间线(基于当前数据)"""
        # 原型里只有当下时间点,生产应该有滚动时间窗口
        return [
            {
                "ts": candidate.get("ts"),
                "event": "系统检测到板块异动",
                "type": "detection",
            }
        ]

