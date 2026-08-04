"""
Sub-Agent Tools - 把 Researcher/Writer/Reviewer 暴露为工具
============================================================

2026-08 改造:Orchestrator 改成 ReAct 后,需要通过工具调子 Agent。

每个 tool handler 实际是调用子 Agent 的对应方法,返回结构化结果。
Orchestrator 看到的就是普通 function-call 工具,可以 ReAct 调多次。

设计:
- call_researcher: 传 topic + 可选 focus_areas,返回 research_brief
- call_writer: 传 research_brief + 可选 style_hint,返回 article dict
- call_reviewer: 传 article + 可选 focus,返回 review dict

注意:这 3 个工具互相**不知道对方存在**,完全由 Orchestrator 决定调用顺序。
"""
from __future__ import annotations
import json
import logging
from typing import Any

import config

logger = logging.getLogger(__name__)


def build_subagent_tools() -> dict:
    """构建 3 个子 Agent 工具 - 返回 dict 而不是注册表,方便 Orchestrator 直接用

    Returns:
        dict[tool_name -> {
            "name": str,
            "description": str,
            "parameters": OpenAI 格式 schema,
            "handler": callable
        }]
    """
    return {
        "call_researcher": {
            "name": "call_researcher",
            "description": (
                "调 Researcher 子 Agent 对一个话题做深度研究。"
                "返回 research_brief(含行业知识、关键事实、工具调用记录、研究摘要)。\n\n"
                "适用:已确定要写某话题但缺数据/财务/新闻深度时调用。"
                "Reviewer 反馈'缺数据'时也可再次调用补充。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_subject": {
                        "type": "string",
                        "description": "话题主题,例如 '通用设备' '电网设备'",
                    },
                    "topic_industry": {
                        "type": "string",
                        "description": "所属行业(可选),例如 '机械' '电力'",
                    },
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "可选,你想重点研究的方向。"
                            "例如 ['财务三表', '业绩预告', '板块资金流']"
                        ),
                    },
                    "extra_symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "额外关注的个股代码列表(可选)",
                    },
                },
                "required": ["topic_subject"],
                "additionalProperties": False,
            },
            "handler": _handle_call_researcher,
        },
        "call_writer": {
            "name": "call_writer",
            "description": (
                "调 Writer 子 Agent 根据研究简报写文章。"
                "输入是 call_researcher 返回的 research_brief(JSON 字符串),"
                "输出是 article dict(content / title / word_count)。\n\n"
                "适用:已有 research_brief,准备产出文章。"
                "Orchestrator 可通过参数控制:\n"
                "- **style_hint**: 风格/结构/可读性调整。值:"
                "'专业严谨' / '通俗易懂' / '学术化' / '短篇快讯' / '深度长文' / '精简到 X 字' 等\n"
                "- **length_target**: 目标字数(整数,默认 1500)。"
                "Reviewer 说「字数过多/不足」时必须传这个(从 issues 里的「目标 N」读出)\n"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "research_brief_json": {
                        "type": "string",
                        "description": (
                            "call_researcher 返回的研究简报 JSON 字符串。"
                            "必须是合法 JSON,完整传回。"
                        ),
                    },
                    "style_hint": {
                        "type": "string",
                        "description": (
                            "风格/结构/可读性调整提示。常用值:\n"
                            "- '专业严谨' / '通俗易懂' / '学术化'\n"
                            "- '短篇快讯' / '深度长文'\n"
                            "- '精简到 N 字' / '扩展到 N 字'\n"
                            "- '加强合规声明' / '加更多数据支撑'"
                        ),
                    },
                    "length_target": {
                        "type": "integer",
                        "description": (
                            "目标字数(整数)。默认读 SM 配置(1500)。"
                            "Reviewer 反馈「字数过多/不足」时必须传此参数,"
                            "值从 issues 里的「目标 N」读出。"
                        ),
                    },
                },
                "required": ["research_brief_json"],
                "additionalProperties": False,
            },
            "handler": _handle_call_writer,
        },
        "call_reviewer": {
            "name": "call_reviewer",
            "description": (
                "调 Reviewer 子 Agent 审阅文章。"
                "输入是 call_writer 返回的 article(JSON 字符串),"
                "输出是 review dict(passed / score / issues / review_summary)。\n\n"
                "适用:文章写完后审查。\n"
                "如果 passed=true 表示不需要修改,否则 Orchestrator 根据 issues "
                "决定补充数据(再调 call_researcher)或修改 prompt(再调 call_writer)。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_json": {
                        "type": "string",
                        "description": "call_writer 返回的文章 JSON 字符串",
                    },
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "审核重点(可选)",
                    },
                },
                "required": ["article_json"],
                "additionalProperties": False,
            },
            "handler": _handle_call_reviewer,
        },
    }


# ============= Handlers - 实际调子 Agent =============

def _handle_call_researcher(topic_subject: str, topic_industry: str = "",
                            focus_areas: list | None = None,
                            extra_symbols: list | None = None) -> dict:
    """call_researcher handler: 调 ResearcherAgent._research_topic"""
    try:
        from agents.researcher import ResearchAgent
        from memory import get_working_memory
        wm = get_working_memory()
        # 从 WM 找匹配的 topic
        topics = wm.get("topics", [])
        topic = None
        for t in topics:
            if t.get("subject") == topic_subject:
                topic = t
                break
        if topic is None:
            from datetime import datetime
            topic = {
                "topic_id": f"topic_synth_{int(datetime.now().timestamp())}",
                "subject": topic_subject,
                "industry": topic_industry or None,
                "anomaly_types": [],
                "score": 0.6,
                "confidence": "mid",
                "description": f"Orchestrator 合成话题:{topic_subject}",
                "symbols": extra_symbols or [],
            }

        agent = ResearchAgent()
        brief = agent._research_topic(topic)
        logger.info(
            f"call_researcher: {topic_subject} → "
            f"brief {len(brief.get('key_facts', []))} facts"
        )

        # 2026-08-04 新增:落盘 brief 到 output/briefs/,让文章能链回
        try:
            from tools.persist import save_brief
            brief_path = save_brief(brief, extra={
                "topic_id": topic.get("topic_id"),
                "source": "call_researcher",
            })
            brief["_brief_path"] = str(brief_path)  # 加到 brief 自身
        except Exception as e:
            logger.warning(f"保存 brief 失败(不影响主流程): {e}")

        return {"success": True, "data": brief}
    except Exception as e:
        logger.error(f"call_researcher failed: {e}")
        return {"success": False, "error": str(e)}


def _handle_call_writer(research_brief_json: str, style_hint: str = "",
                        length_target: int | None = None) -> dict:
    """call_writer handler: 调 WriterAgent._write_article"""
    try:
        if isinstance(research_brief_json, str):
            try:
                brief = json.loads(research_brief_json)
            except json.JSONDecodeError:
                return {"success": False, "error": "research_brief_json 不是合法 JSON"}
        else:
            brief = research_brief_json

        from agents.writer import WriterAgent
        agent = WriterAgent()
        # 临时覆盖语义记忆
        if style_hint:
            brief["_style_override"] = style_hint
        if length_target:
            brief["_length_override"] = length_target
        article = agent._write_article(brief)
        logger.info(
            f"call_writer: {brief.get('subject', '?')} → "
            f"article {len(article.get('content', ''))} chars"
        )
        return {"success": True, "data": article}
    except Exception as e:
        logger.error(f"call_writer failed: {e}")
        return {"success": False, "error": str(e)}


def _handle_call_reviewer(article_json: str, focus_areas: list | None = None) -> dict:
    """call_reviewer handler: 调 ReviewerAgent._review_draft"""
    try:
        if isinstance(article_json, str):
            try:
                article = json.loads(article_json)
            except json.JSONDecodeError:
                return {"success": False, "error": "article_json 不是合法 JSON"}
        else:
            article = article_json

        from agents.reviewer import ReviewerAgent
        agent = ReviewerAgent()
        review = agent._review_draft(article)
        logger.info(
            f"call_reviewer: {article.get('subject', '?')} → "
            f"passed={review.get('passed')}, score={review.get('score')}, "
            f"issues={len(review.get('issues', []))}"
        )
        return {"success": True, "data": review}
    except Exception as e:
        logger.error(f"call_reviewer failed: {e}")
        return {"success": False, "error": str(e)}
