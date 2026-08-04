"""
Dummy Tools - 占位工具集
=========================

2026-08 新增:为了满足"可以新增一些 tool 或者 dummy tool"的需求,
加几个标记为 DUMMY 的占位工具。这些工具的 handler 返回固定 mock 数据,
**不调真实 API**,仅用于:
1. 演示 Orchestrator/Researcher 可调用的工具集如何扩展
2. 测试在没有真实数据源时端到端能否跑通
3. 给 LLM 演示"如果接入了网络搜索/社媒舆情,会返回什么格式的数据"

使用方式:
    from tools.dummy_tools import build_dummy_tools
    reg = ToolRegistry()
    for t in build_dummy_tools():
        reg.register(t)

每个 DUMMY 工具:
- name 前缀 dummy_ 让人一眼看出
- description 里明确标 [DUMMY]
- handler 返回结构化的 mock 数据(可被 LLM 正常解析)
"""
from __future__ import annotations
import logging
import random
from datetime import datetime, timedelta

from .agent_tools import Tool

logger = logging.getLogger(__name__)


def build_dummy_tools() -> list[Tool]:
    """构建 4 个 DUMMY 工具"""
    return [
        Tool(
            name="dummy_web_search",
            description=(
                "[DUMMY] 模拟网络搜索引擎。输入关键词,返回固定的 mock 搜索结果。"
                "不调真实 API,仅用于演示扩展点。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词,例如 'AI 教育 政策'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量,默认 5",
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=_dummy_web_search,
        ),
        Tool(
            name="dummy_social_sentiment",
            description=(
                "[DUMMY] 模拟社交媒体(微博/雪球/股吧)上某话题的舆情数据。"
                "返回发帖量、情绪分布、热门评论等 mock 数据。"
                "实际生产应该接雪球/微博 API 或爬虫。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "话题关键词,例如 '教育'",
                    },
                    "platform": {
                        "type": "string",
                        "enum": ["weibo", "xueqiu", "guba", "all"],
                        "description": "平台,默认 all",
                    },
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
            handler=_dummy_social_sentiment,
        ),
        Tool(
            name="dummy_macro_indicator",
            description=(
                "[DUMMY] 模拟宏观经济指标(CPI/PPI/PMI/社融)。"
                "返回最近 3 个月的 mock 趋势数据。"
                "实际生产应该接国家统计局/央行 API。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "indicator": {
                        "type": "string",
                        "enum": ["CPI", "PPI", "PMI", "社融", "M2"],
                        "description": "指标名",
                    },
                },
                "required": ["indicator"],
                "additionalProperties": False,
            },
            handler=_dummy_macro_indicator,
        ),
        # 2026-08 新增:字数检查作为可调 tool,让 Reviewer 能自主决定调不调
        Tool(
            name="check_article_length",
            description=(
                "检查文章字数是否在合理范围内。"
                "返回 {passed, actual, target, min_ok, max_ok, message}。"
                "Reviewer 调这个 tool,如果不通过把 message 字段当 issue 反馈给 Orchestrator。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "article": {
                        "type": "object",
                        "description": "文章 dict,需要 word_count / content 字段",
                    },
                    "target_length": {
                        "type": "integer",
                        "description": "目标字数(可选,默认从 SM 读)",
                    },
                    "tolerance": {
                        "type": "number",
                        "description": "容差(默认 0.3,即 ±30%)",
                        "default": 0.3,
                    },
                },
                "required": ["article"],
                "additionalProperties": False,
            },
            handler=_check_article_length,
        ),
    ]


# ============= Handlers - 都返回 mock 数据或简单计算 =============

def _check_article_length(
    article: dict,
    target_length: int | None = None,
    tolerance: float = 0.3,
) -> dict:
    """字数检查:返回 passed/actual/target/min_ok/max_ok/message

    Reviewer 调这个 tool,如果 message 不为空,把它当 issue 加到 review 结果。
    """
    actual = article.get("word_count") or len(article.get("content", ""))

    if target_length is None:
        # 从 SM 读默认 target
        try:
            from memory import get_semantic_memory
            sm = get_semantic_memory()
            target_length = sm.get_preferred_length()
        except Exception:
            target_length = 1500

    min_ok = int(target_length * (1 - tolerance))
    max_ok = int(target_length * (1 + tolerance))
    passed = min_ok <= actual <= max_ok

    if actual < min_ok:
        message = f"字数不足:实际 {actual},目标 {target_length},建议范围 {min_ok}-{max_ok}"
    elif actual > max_ok:
        message = f"字数过多:实际 {actual},目标 {target_length},建议范围 {min_ok}-{max_ok}"
    else:
        message = ""  # 通过

    return {
        "passed": passed,
        "actual": actual,
        "target": target_length,
        "min_ok": min_ok,
        "max_ok": max_ok,
        "tolerance": tolerance,
        "message": message,
    }

def _dummy_web_search(query: str, top_k: int = 5) -> list[dict]:
    """[DUMMY] 模拟搜索结果"""
    templates = [
        f"【{query}】相关政策解读,2024 年最新动态",
        f"深度分析:{query} 板块前景与风险点",
        f"专家观点:{query} 行业三季度展望",
        f"市场观察:{query} 资金流向与个股表现",
        f"研报推荐:{query} 龙头公司估值解析",
        f"投资者讨论:{query} 长期价值几何",
        f"政策利好:{query} 迎重大政策催化",
    ]
    random.seed(hash(query) % (2**31))
    selected = random.sample(templates, min(top_k, len(templates)))
    return [
        {
            "title": t,
            "snippet": f"这是 {query} 的 mock 搜索结果片段,DUMMY 数据,实际生产应该接真实搜索引擎 API。",
            "url": f"https://dummy.example.com/search?q={query}&n={i}",
            "source": "dummy.search",
            "ts": (datetime.now() - timedelta(hours=i)).isoformat(timespec="seconds"),
        }
        for i, t in enumerate(selected, 1)
    ]


def _dummy_social_sentiment(topic: str, platform: str = "all") -> dict:
    """[DUMMY] 模拟社媒舆情"""
    random.seed(hash(topic + platform) % (2**31))
    total_posts = random.randint(500, 5000)
    sentiment = {
        "positive": random.randint(20, 50),
        "neutral": random.randint(30, 50),
        "negative": random.randint(10, 30),
    }
    hot_comments = [
        f"看好{topic},这个位置可以加仓了",
        f"{topic} 短期炒作,注意风险",
        f"{topic} 业绩落地,后面还有空间",
        f"机构在出货,大家小心",
    ]
    return {
        "topic": topic,
        "platform": platform,
        "total_posts_24h": total_posts,
        "sentiment_distribution": sentiment,
        "hot_comments": hot_comments,
        "top_influencers": [
            {"name": f"@大V{i}", "followers": random.randint(10000, 1000000), "stance": random.choice(["看多", "看空", "中性"])}
            for i in range(1, 4)
        ],
        "trend_24h": "上升" if sentiment["positive"] > sentiment["negative"] else "下降",
        "note": "[DUMMY] 这是 mock 数据,实际生产应该接雪球/微博 API",
    }


def _dummy_macro_indicator(indicator: str) -> list[dict]:
    """[DUMMY] 模拟宏观指标"""
    base_values = {
        "CPI": 2.5, "PPI": -1.2, "PMI": 50.5, "社融": 35000, "M2": 8.5,
    }
    base = base_values.get(indicator, 5.0)
    random.seed(hash(indicator) % (2**31))
    # 返回最近 3 个月
    today = datetime.now()
    result = []
    for i in range(2, -1, -1):
        date = (today - timedelta(days=30 * i)).strftime("%Y%m")
        value = base + random.uniform(-0.5, 0.5)
        result.append({
            "indicator": indicator,
            "period": date,
            "value": round(value, 2),
            "unit": "%" if indicator in ["CPI", "PPI", "PMI", "M2"] else "亿元",
            "yoy": round(random.uniform(-2, 5), 2),
            "note": "[DUMMY] mock 数据",
        })
    return result
