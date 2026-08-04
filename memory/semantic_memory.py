"""
Semantic Memory - 行业知识 / 读者偏好
======================================

内存结构,启动时从 JSON 加载,运行中持续更新。
"""

from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Optional

import config


# 默认行业知识(简化版 - 生产环境应该用知识图谱)
DEFAULT_INDUSTRY_KG = {
    "新能源": {
        "上游": ["锂矿", "钴矿", "镍矿", "正极材料", "负极材料", "电解液", "隔膜"],
        "中游": ["电池", "电机", "电控", "整车"],
        "下游": ["充电桩", "运营", "回收"],
        "key_players": ["宁德时代", "比亚迪", "赣锋锂业", "华友钴业"],
        "policy_sensitive": True,
    },
    "半导体": {
        "上游": ["硅片", "光刻胶", "电子气体", "靶材"],
        "中游": ["设计", "制造", "封测", "设备"],
        "下游": ["消费电子", "汽车电子", "服务器"],
        "key_players": ["中芯国际", "北方华创", "韦尔股份", "长电科技"],
        "policy_sensitive": True,
    },
    "医药": {
        "上游": ["原料药", "CXO", "试剂"],
        "中游": ["化药", "中药", "生物制品", "医疗器械"],
        "下游": ["医院", "药店", "电商"],
        "key_players": ["恒瑞医药", "迈瑞医疗", "药明康德"],
        "policy_sensitive": True,
    },
    "金融": {
        "上游": [],
        "中游": ["银行", "保险", "证券", "信托"],
        "下游": ["实体经济"],
        "key_players": ["工商银行", "招商银行", "中国平安", "中信证券"],
        "policy_sensitive": True,
    },
    "消费": {
        "上游": ["农产品", "原材料"],
        "中游": ["食品饮料", "纺织服装", "家电", "化妆品"],
        "下游": ["商超", "电商", "专卖店"],
        "key_players": ["贵州茅台", "五粮液", "美的集团", "海天味业"],
        "policy_sensitive": False,
    },
    "AI": {
        "上游": ["算力(GPU/服务器)", "数据", "算法"],
        "中游": ["大模型", "应用开发", "Agent"],
        "下游": ["内容生成", "办公", "金融科技", "自动驾驶", "教育"],
        "key_players": ["科大讯飞", "商汤", "寒武纪", "海光信息"],
        "policy_sensitive": True,
    },
}

# 默认读者画像
DEFAULT_READER_PROFILE = {
    "audience_type": "professional_investor",   # / "retail" / "researcher"
    "risk_preference": "balanced",              # / "conservative" / "aggressive"
    "preferred_topics": ["宏观", "行业", "公司", "政策"],
    "preferred_length": 1500,                   # 目标字数
    "tone": "professional",                     # / "casual" / "academic"
    "compliance_level": "high",                # 合规要求
    "forbidden_words": ["暴涨", "稳赚", "必涨", "翻倍"],
    "must_disclose": ["投资有风险", "过往业绩不代表未来"],
}


class SemanticMemory:
    def __init__(self, path: Optional[Path] = None):
        self._lock = threading.Lock()
        self.path = Path(path or (config.DATA_DIR / "semantic_memory.json"))
        self._industry_kg: dict = {}
        self._reader_profile: dict = {}
        self._load_or_init()
    
    def _load_or_init(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._industry_kg = data.get("industry_kg", DEFAULT_INDUSTRY_KG)
                self._reader_profile = data.get("reader_profile", DEFAULT_READER_PROFILE)
                return
            except Exception:
                pass
        self._industry_kg = DEFAULT_INDUSTRY_KG
        self._reader_profile = DEFAULT_READER_PROFILE
        self._save()
    
    def _save(self) -> None:
        with self._lock:
            data = {
                "industry_kg": self._industry_kg,
                "reader_profile": self._reader_profile,
            }
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    
    # ---- 行业知识 ----
    def get_industry(self, name: str) -> Optional[dict]:
        with self._lock:
            return self._industry_kg.get(name)
    
    def get_all_industries(self) -> list[str]:
        with self._lock:
            return list(self._industry_kg.keys())
    
    def add_industry(self, name: str, info: dict) -> None:
        with self._lock:
            self._industry_kg[name] = info
        self._save()
    
    def find_industry_by_keyword(self, kw: str) -> Optional[str]:
        """根据关键词找行业"""
        kw = kw.lower()
        with self._lock:
            for name, info in self._industry_kg.items():
                if kw in name.lower():
                    return name
                for player in info.get("key_players", []):
                    if kw in player.lower():
                        return name
                for segment in info.get("上游", []) + info.get("中游", []) + info.get("下游", []):
                    if kw in segment.lower():
                        return name
        return None
    
    # ---- 读者画像 ----
    def get_reader_profile(self) -> dict:
        with self._lock:
            return dict(self._reader_profile)
    
    def update_reader_profile(self, updates: dict) -> None:
        with self._lock:
            self._reader_profile.update(updates)
        self._save()
    
    def get_preferred_length(self) -> int:
        with self._lock:
            return self._reader_profile.get("preferred_length", 1500)
    
    def get_tone(self) -> str:
        with self._lock:
            return self._reader_profile.get("tone", "professional")
    
    def is_compliance_high(self) -> bool:
        with self._lock:
            return self._reader_profile.get("compliance_level", "high") == "high"
    
    def get_forbidden_words(self) -> list[str]:
        with self._lock:
            return list(self._reader_profile.get("forbidden_words", []))


_sm_instance: Optional[SemanticMemory] = None
def get_semantic_memory() -> SemanticMemory:
    global _sm_instance
    if _sm_instance is None:
        _sm_instance = SemanticMemory()
    return _sm_instance
