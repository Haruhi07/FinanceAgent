"""
Episodic Memory - 历史热点案例库
=================================

存成 JSONL,每个 case 一行,带反馈标签。

为什么要带反馈标签?
- 系统初期评分权重 w 不准,需要从历史"是不是真的成为热点"来学习
- 类似推荐系统里的"隐式反馈"
"""

from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

import config


class EpisodicMemory:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or (config.DATA_DIR / "episodic_memory.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 如果文件不存在,初始化
        if not self.path.exists():
            self.path.touch()
    
    def add_case(self, hotspot: dict, feedback: Optional[dict] = None) -> str:
        """记录一个案例
        
        Args:
            hotspot: 候选热点 dict(包含 7 维特征、关联的 signals、话题等)
            feedback: 人工反馈或自动结果 {"is_real_hotspot": True/False, "score": 0.8}
        
        Returns:
            case_id
        """
        case = {
            "case_id": uuid.uuid4().hex[:12],
            "ts": datetime.now().isoformat(timespec="seconds"),
            "hotspot": hotspot,
            "feedback": feedback or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(case, ensure_ascii=False, default=str) + "\n")
        return case["case_id"]
    
    def add_feedback(self, case_id: str, feedback: dict) -> bool:
        """给已存在的 case 追加反馈(用于后续反馈闭环)"""
        if not self.path.exists():
            return False
        cases = self._read_all()
        found = False
        for c in cases:
            if c["case_id"] == case_id:
                c["feedback"].update(feedback)
                c["feedback"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
                found = True
                break
        if found:
            self._write_all(cases)
        return found
    
    def find_similar(self, hotspot: dict, top_k: int = 5) -> list[dict]:
        """找历史相似热点(简单文本相似度)
        
        生产环境应该用向量检索,这里先用关键词重叠度做原型
        """
        cases = self._read_all()
        if not cases:
            return []
        
        # 提取当前热点的关键词
        cur_keywords = self._extract_keywords(hotspot)
        
        scored = []
        for c in cases:
            old_keywords = self._extract_keywords(c.get("hotspot", {}))
            # Jaccard 相似度
            inter = cur_keywords & old_keywords
            union = cur_keywords | old_keywords
            sim = len(inter) / len(union) if union else 0
            scored.append((sim, c))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for sim, c in scored[:top_k] if sim > 0]
    
    def get_recent(self, n: int = 50) -> list[dict]:
        cases = self._read_all()
        return cases[-n:]
    
    def get_with_feedback(self) -> list[dict]:
        """只返回已经有反馈的 case(用于训练)"""
        return [c for c in self._read_all() if c.get("feedback")]
    
    def stats(self) -> dict:
        cases = self._read_all()
        with_feedback = [c for c in cases if c.get("feedback")]
        return {
            "total": len(cases),
            "with_feedback": len(with_feedback),
            "confirmed_hotspot": sum(1 for c in with_feedback if c["feedback"].get("is_real_hotspot")),
            "rejected": sum(1 for c in with_feedback if c["feedback"].get("is_real_hotspot") is False),
        }
    
    def _extract_keywords(self, hotspot: dict) -> set:
        """从 hotspot dict 里抽关键词"""
        kws = set()
        for field in ["name", "topic", "board", "summary", "title"]:
            v = hotspot.get(field, "")
            if isinstance(v, str):
                # 中文按 2 字切,英文按词
                for w in v.split():
                    if len(w) >= 2:
                        kws.add(w)
        return kws
    
    def _read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        cases = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return cases
    
    def _write_all(self, cases: list[dict]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")


_em_instance: Optional[EpisodicMemory] = None
def get_episodic_memory() -> EpisodicMemory:
    global _em_instance
    if _em_instance is None:
        _em_instance = EpisodicMemory()
    return _em_instance
