"""
Orchestrator - 编排器(ReAct 模式,2026-08 改造)
================================================

原版:显式状态机调度,6 个 agent 串行 + 写稿阶段 refine 闭环
新版:**ReAct 模式** + Plan-and-Execute 子 Agent
  1. Scanner 仍走显式调用(数据采集前置)
  2. AnomalyDetector / HotspotDetector 仍走显式调用(纯规则,没必要 ReAct)
  3. Topic → Research → Write → Review 改为 OrchestratorAgent 自主 ReAct 决策:
     - Orchestrator 通过 chat_with_tools 调 3 个 sub-agent 工具
       (call_researcher / call_writer / call_reviewer)
     - Orchestrator 根据 Reviewer 反馈决定:补充数据 or 改 prompt
     - 直到 Reviewer passed=true 或 max_rounds
  4. 子 Agent 之间不直接通信(完全由 Orchestrator 调度)

偏好 memory:从 SM.em 读取读者画像/风格/必含披露等
"""
from __future__ import annotations
import logging
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import BaseAgent, AgentResult
from .scanner import ScannerAgent
from .anomaly_detector import AnomalyDetector
from .hotspot_detector import HotspotDetector
from .topic_modeler import TopicModeler
from .researcher import ResearchAgent
from .writer import WriterAgent
from .reviewer import ReviewerAgent
from memory import get_working_memory, get_episodic_memory, get_semantic_memory
from tools import get_tool_registry, get_llm, is_llm_available
from tools.persist import load_latest_candidates
import config

logger = logging.getLogger(__name__)


class Orchestrator(BaseAgent):
    name = "orchestrator"
    
    STATE_INIT = "init"
    STATE_SCANNED = "scanned"
    STATE_ANOMALIES = "anomalies_detected"
    STATE_HOTSPOTS = "hotspots_identified"
    STATE_TOPICS = "topics_modeled"
    STATE_RESEARCHED = "researched"
    STATE_WRITTEN = "written"
    STATE_REVIEWED = "reviewed"
    STATE_DONE = "done"
    STATE_FAILED = "failed"
    
    def __init__(self):
        super().__init__()
        # 实例化所有子 Agent
        self.scanner = ScannerAgent()
        self.anomaly = AnomalyDetector()
        self.hotspot = HotspotDetector()
        self.topic = TopicModeler()
        self.researcher = ResearchAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()
        # 2026-08 新增:Orchestrator 自己也用 LLM 做 ReAct 决策
        self.llm = get_llm() if is_llm_available() else None

        self.state = self.STATE_INIT
    
    def _run(self, input_data: dict | None) -> AgentResult:
        """主流程(2026-08 改造):
        1. Scanner - 多源信号扫描(3 个工作日)
        2. AnomalyDetector - 异动检测(落 JSON)
        3. HotspotDetector - 7 维评分(落 JSON)
        4. TopicModeler - 1:1 包装
        5. **OrchestratorAgent(ReAct)** - 对 top-1 candidate 自主规划:
              call_researcher → call_writer → call_reviewer
              反馈循环:passed=true 退出;否则修改 prompt / 补数据
        """
        if input_data is None:
            input_data = {}

        import time
        t0 = time.time()

        # 1. 扫描
        logger.info("[1/5] Scanner 开始...")
        r = self.scanner.run(input_data)
        logger.info(f"[1/5] Scanner done: {r.duration_ms}ms, metrics={r.metrics}")
        if not r.success:
            return self._fail("scanner failed", r)
        self.state = self.STATE_SCANNED

        # 2. 异动检测(落 JSON)
        logger.info("[2/5] Anomaly Detector 开始...")
        r = self.anomaly.run({})
        logger.info(f"[2/5] Anomaly done: {r.duration_ms}ms, metrics={r.metrics}")
        if not r.success:
            return self._fail("anomaly detector failed", r)
        self.state = self.STATE_ANOMALIES

        # 3. 热点识别(落 JSON)
        logger.info("[3/5] Hotspot Detector 开始...")
        r = self.hotspot.run({})
        logger.info(f"[3/5] Hotspot done: {r.duration_ms}ms, metrics={r.metrics}")
        if not r.success:
            return self._fail("hotspot detector failed", r)
        self.state = self.STATE_HOTSPOTS

        # 4. 话题建模
        candidates = self.wm.get("hotspot_candidates", [])
        actionable = [
            c for c in candidates
            if c["score"] >= config.HOTSPOT_MID_CONF
        ]
        logger.info(f"[4/5] actionable 候选: {len(actionable)}/{len(candidates)}")
        if not actionable:
            return AgentResult(
                agent_name=self.name, success=True,
                data={"reason": "no actionable hotspots", "total_candidates": len(candidates)},
                metrics={"candidates": len(candidates), "actionable": 0},
            )
        r = self.topic.run({"candidates": actionable})
        logger.info(f"[4/5] Topic done: {r.duration_ms}ms, {r.metrics}")
        if not r.success:
            return self._fail("topic modeler failed", r)
        self.state = self.STATE_TOPICS

        # 5. **ReAct Orchestrator** - 对 top-N candidates 并行跑 R/W/R 闭环
        logger.info("[5/5] OrchestratorAgent(ReAct Top-N 并行) 开始...")
        t5 = time.time()
        r = self._react_orchestrate_topn()
        logger.info(
            f"[5/5] ReAct Orch done: {time.time()-t5:.1f}s, metrics={r.metrics}"
        )
        if not r.success:
            return self._fail("react orchestration failed", r)
        self.state = self.STATE_DONE

        # 落盘终稿
        self._persist_output()

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={"state": self.state, "mode": "react"},
            metrics={
                "candidates": len(candidates),
                "actionable": len(actionable),
                "topics": len(self.wm.get("topics", [])),
                "published": len(self.wm.get("published", [])),
                "react_rounds": r.metrics.get("rounds", 0),
                "react_tool_calls": r.metrics.get("tool_calls", 0),
            },
        )
    
    def _fail(self, reason: str, last_result: AgentResult) -> AgentResult:
        self.state = self.STATE_FAILED
        return AgentResult(
            agent_name=self.name,
            success=False,
            errors=[reason] + last_result.errors,
            data={"state": self.state, "last_agent": last_result.agent_name},
        )
    
    def _parallel_research(self, topics: list[dict], max_workers: int = 3) -> AgentResult:
        """并行研究多个 topic(线程隔离,直接传 topic 不走 WM)"""
        if not topics:
            return AgentResult(agent_name="parallel_research", success=False,
                             errors=["no topics"])
        
        briefs = []
        errors = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_topic = {
                executor.submit(self._research_one_safe, t): t
                for t in topics
            }
            for future in as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    brief = future.result()
                    if brief:
                        briefs.append(brief)
                except Exception as e:
                    errors.append(f"{topic.get('subject', '?')}: {e}")
                    logger.error(f"research failed for {topic.get('subject', '?')}: {e}")
        
        # 串行 append 到 WM(避免竞争)
        for b in briefs:
            self.wm.append("researches", b)
        
        return AgentResult(
            agent_name="parallel_research",
            success=len(briefs) > 0,
            items=briefs,
            metrics={"researched": len(briefs), "failed": len(errors)},
            errors=errors,
        )
    
    def _research_one_safe(self, topic: dict) -> Optional[dict]:
        """线程隔离地研究单个 topic - 直接传 topic 不走 WM"""
        try:
            # 直接调内部方法,避免 _run 从 WM 读 topics
            agent = ResearchAgent()
            return agent._research_topic(topic)
        except Exception as e:
            logger.error(f"_research_one_safe failed: {e}")
            return None
    
    def _parallel_write(self, max_workers: int = 3) -> AgentResult:
        """并行写多篇(线程隔离)"""
        researches = self.wm.get("researches", [])
        if not researches:
            return AgentResult(agent_name="parallel_write", success=False,
                             errors=["no researches"])
        
        drafts = []
        errors = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_brief = {
                executor.submit(self._write_one_safe, b): b
                for b in researches
            }
            for future in as_completed(future_to_brief):
                brief = future_to_brief[future]
                try:
                    draft = future.result()
                    if draft:
                        drafts.append(draft)
                except Exception as e:
                    errors.append(f"{brief.get('subject', '?')}: {e}")
                    logger.error(f"write failed: {e}")
        
        for d in drafts:
            self.wm.append("drafts", d)
        
        return AgentResult(
            agent_name="parallel_write",
            success=len(drafts) > 0,
            items=drafts,
            metrics={"drafted": len(drafts), "failed": len(errors)},
            errors=errors,
        )
    
    def _write_one_safe(self, brief: dict) -> Optional[dict]:
        """线程隔离地写单个 brief"""
        try:
            agent = WriterAgent()
            return agent._write_article(brief)
        except Exception as e:
            logger.error(f"_write_one_safe failed: {e}")
            return None
    
    def _parallel_write_with_refinement(self, max_workers: int = 3) -> AgentResult:
        """并行写多篇,带 Writer-Reviewer 迭代修订
        
        流程:
        1. 并行写初稿
        2. 并行跑 Reviewer
        3. 没通过的文章,Writer 根据 review issues 修改
        4. 再 review,直到 pass 或达到 max_rounds
        """
        researches = self.wm.get("researches", [])
        if not researches:
            return AgentResult(agent_name="parallel_write_refine", success=False,
                             errors=["no researches"])
        
        # 1. 并行写初稿
        drafts = []
        draft_to_brief = {}  # article_id -> brief
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_brief = {
                executor.submit(self._write_one_safe, b): b
                for b in researches
            }
            for future in as_completed(future_to_brief):
                brief = future_to_brief[future]
                try:
                    draft = future.result()
                    if draft:
                        drafts.append(draft)
                        draft_to_brief[draft["article_id"]] = brief
                except Exception as e:
                    logger.error(f"initial write failed: {e}")
        
        for d in drafts:
            self.wm.append("drafts", d)
        
        if not drafts:
            return AgentResult(agent_name="parallel_write_refine", success=False,
                             errors=["no drafts written"])
        
        # 2. 迭代 refine
        max_rounds = config.LLM_REFINEMENT_MAX_ROUNDS
        refined_count = 0
        total_rounds = 0
        refinement_history = []  # 记录所有 refine 轨迹
        
        for round_idx in range(max_rounds + 1):  # round 0 = 初次
            # 只 review 需要 review 的 draft(已 review 过的没改的不再 review)
            to_review = []
            for d in drafts:
                if round_idx == 0:
                    # 第一轮:全部 review
                    to_review.append(d)
                else:
                    # 后续轮:只 review 被 refine 过的
                    if d.get("revised"):
                        to_review.append(d)
            
            review_results_map = {}  # article_id -> review
            review_results_list = self._parallel_review_drafts(to_review, max_workers)
            for d, r in zip(to_review, review_results_list):
                if r is not None:
                    review_results_map[d.get("article_id")] = r
            
            # 把 review 结果存到 WM
            for article_id, review in review_results_map.items():
                review["_round"] = round_idx
                self.wm.append("reviews", review)
            
            total_rounds += 1
            
            # 合并所有 drafts 的 review 状态
            all_reviews = []
            for d in drafts:
                if d.get("article_id") in review_results_map:
                    all_reviews.append(review_results_map[d.get("article_id")])
                else:
                    # 没 review 的(已经通过的)用上次的 review
                    # 从 WM reviews 里找这个 article 的最新 review
                    latest = None
                    for r in reversed(self.wm.get("reviews", [])):
                        if r.get("article_id") == d.get("article_id"):
                            latest = r
                            break
                    all_reviews.append(latest)
            
            # 检查哪些需要改
            to_revise = []
            for draft, review in zip(drafts, all_reviews):
                if review is None:
                    continue
                if review["passed"]:
                    # 通过的加入 published(去重)
                    article_id = draft.get("article_id")
                    if not any(p.get("article_id") == article_id
                              for p in self.wm.get("published", [])):
                        # 标记这是 refine 后的最终通过版本
                        draft["final_review_score"] = review.get("score", 0)
                        self.wm.append("published", draft)
                    continue
                if round_idx >= max_rounds:
                    # 已达最大轮数,看是否接受
                    if not config.LLM_REFINEMENT_ACCEPT_ALL:
                        # 标记为丢弃(可以从 published 移除)
                        logger.warning(
                            f"Article '{draft['subject']}' 达到 max_rounds={max_rounds} "
                            f"仍未通过,标记为丢弃"
                        )
                        draft["dropped"] = True
                    continue
                to_revise.append((draft, review))

            if not to_revise:
                # 全部通过(或全部丢弃,都到终态)
                break
            
            # 3. 并行 refine
            new_drafts = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_draft = {
                    executor.submit(
                        self._revise_one,
                        draft,
                        review,
                        draft_to_brief.get(draft["article_id"]),
                    ): (draft, review)
                    for draft, review in to_revise
                }
                for future in as_completed(future_to_draft):
                    draft, review = future_to_draft[future]
                    try:
                        revised = future.result()
                        if revised:
                            new_drafts.append(revised)
                            # 更新 drafts 列表中对应位置
                            for i, d in enumerate(drafts):
                                if d["article_id"] == draft["article_id"]:
                                    drafts[i] = revised
                                    break
                            # 记录
                            refinement_history.append({
                                "article_id": draft["article_id"],
                                "subject": draft["subject"],
                                "round": round_idx + 1,
                                "issues": review.get("issues", []),
                                "old_score": review.get("score", 0),
                                "new_word_count": len(revised["content"]),
                            })
                            refined_count += 1
                    except Exception as e:
                        logger.error(f"refine failed: {e}")
            
            # 4. 更新 WM 里的 drafts
            self.wm.set("drafts", drafts)
            # 5. 更新 published 列表(去掉被 refine 的)
            new_published = [
                d for d in self.wm.get("published", [])
                if d.get("article_id") not in {r["article_id"] for r in new_drafts}
            ]
            self.wm.set("published", new_published)
        
        # 把 refinement_history 存到 WM
        if refinement_history:
            self.wm.set("refinement_history", refinement_history)
        
        return AgentResult(
            agent_name="parallel_write_refine",
            success=len(drafts) > 0,
            items=drafts,
            metrics={
                "drafted": len(drafts),
                "refined": refined_count,
                "total_rounds": total_rounds,
                "refinement_history": refinement_history,
            },
        )
    
    def _parallel_review_drafts(self, drafts: list, max_workers: int = 3) -> list:
        """并行 review 多个 drafts,返回与 drafts 对应的 review 列表"""
        results = [None] * len(drafts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._review_one, d): i
                for i, d in enumerate(drafts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    review = future.result()
                    results[idx] = review
                except Exception as e:
                    logger.error(f"review failed for draft {idx}: {e}")
                    results[idx] = None
        return results
    
    def _review_one(self, draft: dict) -> dict:
        """线程隔离地 review 单个 draft"""
        try:
            agent = ReviewerAgent()
            return agent._review_draft(draft)
        except Exception as e:
            logger.error(f"_review_one failed: {e}")
            return None
    
    def _revise_one(self, draft: dict, review: dict, brief: dict | None) -> Optional[dict]:
        """线程隔离地 revise 单个 draft"""
        try:
            agent = WriterAgent()
            return agent.revise_article(draft, review, brief)
        except Exception as e:
            logger.error(f"_revise_one failed: {e}")
            return None
    
    def _persist_output(self) -> None:
        """把通过的 articles 落盘到 output/articles/(按 article_id 去重)"""
        articles_dir = config.ARTICLES_DIR
        reviews = self.wm.get("reviews", [])
        drafts = self.wm.get("drafts", [])
        
        # 按 article_id 去重,只保留每篇文章的"最新通过 review"
        latest_passed_by_aid = {}  # article_id -> review
        for review in reviews:
            if not review.get("passed"):
                continue
            aid = review.get("article_id")
            latest_passed_by_aid[aid] = review  # 后出现的覆盖前出现的(WM 顺序)
        
        for aid, review in latest_passed_by_aid.items():
            # 找匹配的 draft(取最新版本)
            for draft in drafts:
                if draft.get("article_id") == aid:
                    self._save_article(draft, review)
                    break
    
    def _save_article(self, draft: dict, review: dict) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        subject_safe = draft.get("subject", "unknown").replace("/", "_").replace(" ", "_")
        path = config.ARTICLES_DIR / f"{ts}_{subject_safe}.md"

        # 2026-08-04:从 brief_id 找对应 brief 路径,塞到 article 头
        brief_id = draft.get("brief_id")
        brief_path_str = "未生成"
        if brief_id:
            try:
                from tools.persist import load_brief
                brief = load_brief(brief_id)
                if brief:
                    saved_at = brief.get("saved_at", "")
                    filename = f"{saved_at}_{subject_safe}_{brief_id}.json"
                    brief_path_str = f"output/briefs/{filename}"
            except Exception as e:
                logger.warning(f"load_brief({brief_id}) failed: {e}")

        # 2026-08-04 简化:正文只保留 article 本体,审阅明细/简报摘要/关键事实不再下放到正文
        # 这些元数据保留在头部的 > 行,brief 通过路径引用
        content = f"""# {draft.get('title')}

> 文章ID:{draft.get('article_id')}  
> 主题:{draft.get('subject')}  
> 字数:{draft.get('word_count')}  
> 生成时间:{draft.get('ts')}  
> 审核:{review.get('review_summary')}  
> 审核分数:{review.get('score')}/100  
> 研究简报:{brief_path_str}  

---

{draft.get('content')}
"""
        path.write_text(content, encoding="utf-8")
        logger.info(f"已保存文章: {path.name} (brief: {brief_path_str})")

        # 给案例库加初始反馈
        # 这里简化:审核通过当作"可能是真的热点"的弱信号
        em = get_episodic_memory()
        for cand in self.wm.get("hotspot_candidates", []):
            if cand.get("subject") == draft.get("subject"):
                em.add_feedback(
                    cand.get("candidate_id", ""),
                    {"auto_passed_review": True, "review_score": review.get("score")},
                )
                break
    
    def run_with_human_review(self) -> AgentResult:
        """带人工审核的流程:中确信度的 candidate 需要人工确认后再继续"""
        # 先跑扫描到热点识别
        r1 = self.scanner.run({})
        r2 = self.anomaly.run({})
        r3 = self.hotspot.run({})
        if not (r1.success and r2.success and r3.success):
            return self._fail("initial scan failed", r3)
        
        candidates = self.wm.get("hotspot_candidates", [])
        # 分类
        high = [c for c in candidates if c["score"] >= config.HOTSPOT_HIGH_CONF]
        mid = [c for c in candidates if config.HOTSPOT_MID_CONF <= c["score"] < config.HOTSPOT_HIGH_CONF]
        low = [c for c in candidates if c["score"] < config.HOTSPOT_MID_CONF]
        
        logger.info(f"分类结果:high={len(high)}, mid={len(mid)}, low={len(low)}")
        
        # mid 走人工确认(这里模拟,直接接受)
        logger.info(f"中确信度候选 {len(mid)} 个,默认接受(实际生产应展示给人工)")
        actionable = high + mid
        
        if not actionable:
            return AgentResult(
                agent_name=self.name, success=True,
                data={"reason": "no actionable"},
            )
        
        self.topic.run({})
        self.researcher.run({})
        self.writer.run({})
        self.reviewer.run({})
        self._persist_output()

        return AgentResult(
            agent_name=self.name, success=True,
            metrics={"actionable": len(actionable)},
        )

    # ============= 2026-08 新增:ReAct Orchestrator =============

    def _react_orchestrate_top1(self) -> AgentResult:
        """OrchestratorAgent(ReAct 模式)对 top-1 candidate 跑 R/W/R 闭环

        流程:
        1. 选 top-1 candidate(评分最高,conf=mid+)
        2. 读偏好 memory(SM)
        3. 启动 chat_with_tools,工具:3 个 sub-agent
        4. Orchestrator 自主决定调用顺序
        5. 直到 call_reviewer 返回 passed=true,或 max_rounds 到达
        6. 终稿存 WM.published

        与串行版的区别:
        - 子 Agent 不直接通信,Orchestrator 转发数据
        - Orchestrator 根据 Reviewer 反馈动态决定下一步
        - 可在 ReAct 循环中再次调 call_researcher 补充数据
        """
        # 选 top-1 candidate
        candidates = [
            c for c in self.wm.get("hotspot_candidates", [])
            if c["score"] >= config.HOTSPOT_MID_CONF
        ]
        if not candidates:
            return AgentResult(
                agent_name="react_orchestrator", success=False,
                errors=["无可用 candidate"],
            )
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top1 = candidates[0]
        # 委托给 _react_orchestrate_one
        return self._react_orchestrate_one(top1, trace_suffix="top1")

    def _react_orchestrate_topn(self, top_n: int | None = None, max_workers: int | None = None) -> AgentResult:
        """OrchestratorAgent(ReAct 模式)对 top-N candidates 并行跑 R/W/R 闭环

        2026-08-23 新增:把单 candidate 的 ReAct 扩展到 top-N 并行

        流程:
        1. 取 top-N candidates(按 score 降序)
        2. 用 ThreadPoolExecutor 并行跑每个 candidate 的 ReAct
        3. 每个 worker 独立:自己的 LLM chat_with_tools,自己的 messages
        4. 收集所有 published articles
        """
        top_n = top_n if top_n is not None else getattr(config, "ORCHESTRATOR_TOP_N", 5)
        max_workers = max_workers if max_workers is not None else getattr(config, "ORCHESTRATOR_MAX_WORKERS", 3)

        candidates = [
            c for c in self.wm.get("hotspot_candidates", [])
            if c["score"] >= config.HOTSPOT_MID_CONF
        ]
        if not candidates:
            return AgentResult(
                agent_name="react_orchestrator_topn", success=False,
                errors=["无可用 candidate"],
            )
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:top_n]
        logger.info(
            f"并行 ReAct 启动: {len(top_candidates)} 个 candidate, "
            f"max_workers={max_workers}"
        )

        # 并行跑
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(top_candidates))) as executor:
            future_to_topic = {
                executor.submit(
                    self._react_orchestrate_one,
                    topic,
                    trace_suffix=f"topn_{i}",
                ): (i, topic)
                for i, topic in enumerate(top_candidates, 1)
            }
            for future in as_completed(future_to_topic):
                i, topic = future_to_topic[future]
                try:
                    result = future.result()
                    results.append((i, topic.get("subject", "?"), result))
                    logger.info(
                        f"  Top-{i} {topic.get('subject')} 完成: "
                        f"rounds={result.metrics.get('rounds', 0)}, "
                        f"tool_calls={result.metrics.get('tool_calls', 0)}"
                    )
                except Exception as e:
                    logger.error(f"  Top-{i} {topic.get('subject')} 失败: {e}")
                    results.append((i, topic.get("subject", "?"), None))

        # 汇总
        success_count = sum(1 for _, _, r in results if r and r.success)
        published = self.wm.get("published", [])
        logger.info(
            f"并行 ReAct 完成: {success_count}/{len(top_candidates)} 成功, "
            f"共发布 {len(published)} 篇"
        )

        return AgentResult(
            agent_name="react_orchestrator_topn",
            success=len(published) > 0,
            data={"candidates_processed": [s for _, s, _ in results]},
            metrics={
                "top_n": top_n,
                "max_workers": max_workers,
                "candidates_attempted": len(top_candidates),
                "candidates_succeeded": success_count,
                "total_published": len(published),
            },
        )

    def _react_orchestrate_one(self, target_topic: dict, trace_suffix: str = "one") -> AgentResult:
        """对单个 candidate 跑 ReAct 闭环(给 top1 / topN 共用)

        流程:
        1. 读偏好 memory(SM)
        2. 启动 chat_with_tools,工具:3 个 sub-agent
        3. Orchestrator 自主决定调用顺序
        4. 外层循环:直到 call_reviewer 返回 passed=true,或 max_outer 到达
        5. 终稿存 WM.published
        """
        self.wm.set("react_target_topic", target_topic)
        logger.info(
            f"ReAct target: {target_topic.get('subject')} "
            f"(score={target_topic.get('score')}, conf={target_topic.get('confidence')})"
        )

        if not is_llm_available():
            return AgentResult(
                agent_name="react_orchestrator", success=False,
                errors=["LLM 不可用,ReAct Orchestrator 需要 LLM"],
            )

        # 选 top-1 candidate(给 _react_orchestrate_one 内部 fallback 用,实际由 caller 传 target_topic)
        candidates = [
            c for c in self.wm.get("hotspot_candidates", [])
            if c["score"] >= config.HOTSPOT_MID_CONF
        ]
        if not candidates:
            return AgentResult(
                agent_name="react_orchestrator", success=False,
                errors=["无可用 candidate"],
            )

        # 读偏好 memory
        sm = get_semantic_memory()
        reader_profile = sm._reader_profile
        pref_str = (
            f"读者画像: {reader_profile.get('audience_type', '?')}, "
            f"风险偏好: {reader_profile.get('risk_preference', '?')}, "
            f"风格: {sm.get_tone()}, "
            f"必含披露: {reader_profile.get('must_disclose', [])}, "
            f"目标字数: {sm.get_preferred_length()}"
        )

        # 准备 Orchestrator 的 prompt
        system_prompt = self._build_orchestrator_system_prompt()
        user_prompt = self._build_orchestrator_user_prompt(target_topic, pref_str)

        # 工具:3 个 sub-agent
        registry = get_tool_registry()
        subagent_tool_names = ["call_researcher", "call_writer", "call_reviewer"]
        tools_schema = registry.to_openai_tools(names=subagent_tool_names)

        def tool_executor(name: str, args: dict) -> dict:
            return asyncio.run(registry.execute(name, args))

        # ---- 2026-08: 显式外层循环,review 没通过就追加 review 反馈再跑一轮 ----
        # 内层 max_rounds=6 留给单次 ReAct;外层 max_outer=3 留给"改 prompt 重试"
        max_outer = getattr(config, "ORCHESTRATOR_MAX_OUTER_ROUNDS", 3)
        all_articles = []
        all_briefs = []
        all_reviews = []
        final_article = None
        final_review = None
        total_tool_calls = 0
        outer_round = 0
        last_review_feedback = None
        last_article_json = None
        last_brief_json = None
        # 2026-08-05 新增:自我反思状态(让 LLM 自己观察 delta,决定是否调整策略)
        prev_article_snapshot = None   # 上轮 article 的关键指标
        prev_outer_round = 0            # 用于判断 round 1 vs 后续

        while outer_round < max_outer:
            outer_round += 1
            logger.info(f"--- Orchestrator outer round {outer_round}/{max_outer} ---")

            # 构造本轮 user message(把上一轮 review 反馈塞进去)
            round_user_msg = user_prompt
            if last_review_feedback:
                # 2026-08-05:先做 self-reflection(把上轮 article 快照给 LLM,
                # 让它自己对比 delta、判断策略是否有效)
                reflection_block = ""
                if prev_article_snapshot:
                    reflection_block = self._build_reflection_block(
                        prev_article_snapshot, final_review, last_review_feedback
                    )
                round_user_msg += (
                    f"\n\n## 上一轮 Reviewer 反馈(必须解决这些 issues,不能跳过)\n"
                    f"{last_review_feedback}\n"
                    f"{reflection_block}"
                    f"\n\n你的下一步决策(**严格按 issues 类型调工具**):\n"
                    f"\n"
                    f"### 🔍 需要补数据类(调 call_researcher,不要调 call_writer)\n"
                    f"- **[CONTEXT] 行业背景信息缺失** → 调 call_researcher,focus_areas=['行业背景','产业链','政策环境'],让 Researcher 重点查 industry 知识\n"
                    f"- **[DATA] 研究数据不完整:缺少 XXX** → 调 call_researcher,focus_areas 指向缺失的类别(如 ['涨停股池','个股新闻'])\n"
                    f"- **缺数据/缺财务/缺新闻(LLM 笼统说的)** → 调 call_researcher 补数据\n"
                    f"\n"
                    f"### ✏️ 需要改文章类(调 call_writer)\n"
                    f"- **字数问题([LENGTH] 字数过多/不足)** → 调 call_writer,**必须传 length_target=数字**(从 issues 解析出 Y,如 '字数过多:实际 3289,目标 1500' → length_target=1500)\n"
                    f"- **关键事实未体现([FACTS] 关键事实未体现: ...)** → 调 call_writer,只传 style_hint 提醒它把缺失事实写进文章(如 style_hint='务必把 XXX 关键事实写进文章')\n"
                    f"- **风格/结构/可读性/合规** → 调 call_writer,只传 style_hint 改 prompt(不带 length_target,除非同时有字数问题)\n"
                    f"\n"
                    f"### 🔀 多 issue 混合\n"
                    f"- **同时有 [CONTEXT]/[DATA] 和 [LENGTH]/[FACTS]** → 先调 call_researcher 补数据,再调 call_writer 重写,最后 call_reviewer 验证\n"
                    f"- **只有 [LENGTH] + [FACTS]** → 一次性调 call_writer,同时传 style_hint 和 length_target\n"
                    f"\n"
                    f"### ⚠️ 验证\n"
                    f"调完修复工具后**必须再调 call_reviewer** 验证修改后的文章\n"
                    f"\n## 🚨 硬性规则(2026-08 强制)\n"
                    f"1. **字数问题必须用 length_target,严禁用 style_hint 改字数**\n"
                    f"   - 反例:issues=[字数过多 3289/1500] → style_hint='精简到 1500 字' ❌ 错\n"
                    f"   - 正例:issues=[字数过多 3289/1500] → length_target=1500 ✅ 对\n"
                    f"2. 解析 issues 格式:「字数过多:实际 X,目标 Y」→ length_target=Y,不要从别的字段拿\n"
                    f"3. 只有非字数类的 style_hint(专业/通俗/学术/短篇/深度/精简风格)才算合规修改\n"
                    f"\n## ⛔ 绝对禁止(2026-08-04 新增)\n"
                    f"4. **Reviewer 返回 passed=false 时,严禁直接输出 TASK_COMPLETE**\n"
                    f"   - 必须在下一轮调 call_writer 或 call_researcher 修复 issues\n"
                    f"   - 然后再调 call_reviewer 验证\n"
                    f"   - 循环直到 passed=true 或达到 max_outer 轮数\n"
                    f"5. **外层最多 {max_outer} 轮**,如果到上限还是 passed=false,这篇文章会被拒收(不会发布)\n"
                )

            # 跑一轮 ReAct
            try:
                result = self.llm.chat_with_tools(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": round_user_msg},
                    ],
                    tools=tools_schema,
                    tool_executor=tool_executor,
                    max_rounds=getattr(config, "ORCHESTRATOR_MAX_ROUNDS", 6),
                    max_tokens=config.LLM_LONG_COT_MAX_TOKENS,
                    enable_thinking_for_tools=True,
                )
            except Exception as e:
                logger.error(f"ReAct Orchestrator LLM 调用失败(round {outer_round}): {e}")
                break

            # 解析本轮产出
            def _unwrap(content_str: str) -> dict | None:
                try:
                    obj = json.loads(content_str)
                except (json.JSONDecodeError, TypeError):
                    return None
                while isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
                    obj = obj["data"]
                return obj

            round_briefs = []
            round_articles = []
            round_reviews = []
            for msg in result.get("messages", []):
                if msg.get("role") != "tool":
                    continue
                data = _unwrap(msg.get("content", "{}"))
                if not isinstance(data, dict):
                    continue
                if "article_id" in data and "content" in data:
                    round_articles.append(data)
                elif "key_facts" in data and "research_summary" in data:
                    round_briefs.append(data)
                elif "passed" in data and "score" in data:
                    round_reviews.append(data)

            total_tool_calls += len(result.get("tool_calls", []))
            all_briefs.extend(round_briefs)
            all_articles.extend(round_articles)
            all_reviews.extend(round_reviews)

            if round_articles:
                final_article = round_articles[-1]
                last_article_json = json.dumps(final_article, ensure_ascii=False, default=str)
            if round_briefs:
                last_brief_json = json.dumps(round_briefs[-1], ensure_ascii=False, default=str)
            if round_reviews:
                final_review = round_reviews[-1]

            # 2026-08-05:捕获当前 article 的"指纹"快照,供下轮 self-reflection
            if final_article:
                prev_article_snapshot = self._snapshot_article(final_article, final_review)

            # 检查是否要继续
            if final_review and final_review.get("passed"):
                logger.info(
                    f"Orchestrator 闭环完成:passed=True, "
                    f"score={final_review.get('score')}, "
                    f"outer_rounds={outer_round}"
                )
                break
            elif final_review:
                # Reviewer 没通过 → 准备下轮反馈
                issues = final_review.get("issues", [])
                last_review_feedback = (
                    f"Reviewer 评分: {final_review.get('score', 0)}/100\n"
                    f"Issues:\n" + "\n".join(f"  - {i}" for i in issues)
                )
                logger.warning(
                    f"Round {outer_round} Reviewer 未通过,score={final_review.get('score')},"
                    f"{len(issues)} issues,继续下一轮..."
                )
            else:
                # 没有 review 结果(异常),退出
                logger.warning(f"Round {outer_round} 没拿到 review 结果,停止")
                break

        # 把产物存 WM(供落盘)
        for b in all_briefs:
            self.wm.append("researches", b)
        for a in all_articles:
            self.wm.append("drafts", a)
        for r in all_reviews:
            self.wm.append("reviews", r)

        # 终稿加入 published(2026-08-04 收紧:Reviewer 不过不发布)
        if (
            final_article
            and final_review is not None
            and final_review.get("passed") is True
        ):
            self.wm.append("published", final_article)
            logger.info(
                f"ReAct 终稿产出: {final_article.get('subject')} "
                f"({len(final_article.get('content', ''))} chars, "
                f"score={final_review.get('score')})"
            )
            success = True
        elif final_article and (final_review is None or not final_review.get("passed")):
            # Reviewer 没通过 → 不发布,记到 failed_articles 供 Orchestrator 复盘
            logger.error(
                f"❌ Reviewer 未通过,拒绝发布: {final_article.get('subject')} "
                f"score={final_review.get('score') if final_review else 'N/A'}, "
                f"issues={final_review.get('issues', []) if final_review else []}"
            )
            self.wm.append("failed_articles", {
                "article": final_article,
                "review": final_review,
                "outer_rounds": outer_round,
                "ts": datetime.now().isoformat(timespec="seconds"),
            })
            success = False
        else:
            success = False

        # 存 trace
        trace = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "target_topic": target_topic.get("subject"),
            "rounds": result.get("rounds", 0),
            "tool_calls": [
                {"name": tc.get("name"), "args": tc.get("arguments_raw", "")[:100]}
                for tc in result.get("tool_calls", [])
            ],
            "final_content": result.get("final_content", "")[:500],
            "preferences": pref_str,
            "n_briefs": len(all_briefs),
            "n_articles": len(all_articles),
            "n_reviews": len(all_reviews),
        }
        self.wm.set("react_trace", trace)
        # 也存到磁盘
        try:
            trace_path = config.OUTPUT_DIR / "react_traces" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning(f"trace 落盘失败: {e}")

        return AgentResult(
            agent_name="react_orchestrator",
            success=success,
            items=[final_article] if final_article else [],
            metrics={
                "rounds": result.get("rounds", 0),
                "tool_calls": len(result.get("tool_calls", [])),
                "n_briefs": len(all_briefs),
                "n_articles": len(all_articles),
                "n_reviews": len(all_reviews),
                "final_passed": final_review.get("passed") if final_review else None,
                "final_score": final_review.get("score") if final_review else None,
            },
        )

    def _build_orchestrator_system_prompt(self) -> str:
        """Orchestrator ReAct 的 system prompt"""
        return (
            "你是 **OrchestratorAgent**,负责协调 3 个子 Agent 产出一篇高质量财经热点文章。\n\n"
            "## 你能调的工具(子 Agent)\n"
            "1. **call_researcher**(topic_subject, ...) - 让 Researcher 去收集数据/财务/新闻\n"
            "2. **call_writer**(research_brief_json, style_hint, ...) - 让 Writer 根据研究简报写文章\n"
            "3. **call_reviewer**(article_json, ...) - 让 Reviewer 审阅文章,返回 issues 和 score\n\n"
            "## ReAct 工作循环\n"
            "每一步都要思考:\n"
            "  **Thought**: 我下一步要做什么?为什么?\n"
            "  **Action**: 调用哪个工具?参数是什么?\n"
            "  **Observation**: 工具返回了什么?\n"
            "  ...继续到 Reviewer 返回 passed=true 为止。\n\n"
            "## 决策规则\n"
            "- **第一次**:必须先调 call_researcher(没数据没法写)\n"
            "- 拿到 research_brief 后,调 call_writer\n"
            "- 拿到 article 后,必须调 call_reviewer\n"
            "- **如果 Reviewer 没通过(passed=false)**,严格按 issue 前缀决定调哪个工具:\n"
            "  - **[CONTEXT] 行业背景缺失** 或 **[DATA] 研究数据不完整** → 调 call_researcher 补数据(focus_areas 指向缺失项)\n"
            "  - **笼统「缺数据/缺财务/缺新闻」** → 调 call_researcher 补数据\n"
            "  - **[LENGTH] 字数过多/不足** → 调 call_writer,**必须传 length_target=Y**(从 issues 解析)\n"
            "  - **[FACTS] 关键事实未体现** → 调 call_writer,style_hint 提醒它引用 brief 里的事实\n"
            "  - **风格/结构/可读性/合规** → 调 call_writer,style_hint 改 prompt\n"
            "  - **同时有 [CONTEXT]/[DATA] 和 [LENGTH]/[FACTS]** → 先 researcher 补,再 writer 改\n"
            "  - 修复后**必须再调 call_reviewer** 验证 passed\n"
            "- **如果 Reviewer 通过(passed=true)** → 停止,输出「TASK_COMPLETE」\n\n"
            "## ⛔ 绝对禁止(2026-08-04 新增)\n"
            "- **Reviewer 返回 passed=false 时,严禁直接输出 TASK_COMPLETE**\n"
            "- 必须先调 call_writer / call_researcher 修复,再调 call_reviewer 验证\n"
            "- 如果硬要输出 TASK_COMPLETE 而 passed=false,这篇文章会被系统拒收(不会落盘)\n\n"
            "## 偏好 memory\n"
            "Orchestrator 有自己的偏好 memory(从 SM 读),文章要符合:专业风格、必含合规披露、字数 ~1800。\n\n"
            "## 子 Agent 互不通信\n"
            "call_researcher / call_writer / call_reviewer 之间不直接传数据。\n"
            "你必须把上一次工具返回的 JSON 完整传回给下一个工具。\n\n"
            "## 工具调用格式提醒\n"
            "- call_researcher 返回 {\"success\": true, \"data\": {brief}} - 取 data 字段\n"
            "- call_writer 返回 {\"success\": true, \"data\": {article}} - 取 data 字段\n"
            "- call_reviewer 返回 {\"success\": true, \"data\": {review}} - 取 data 字段\n"
            "**传下一个工具时,把 data 字段 JSON 序列化后传**(如 research_brief_json 字段)。\n\n"
            "## 结束条件\n"
            "最后一条 assistant 消息里写「TASK_COMPLETE」表示本次 Orchestrator 任务完成。"
        )

    def _build_orchestrator_user_prompt(self, top1: dict, pref_str: str) -> str:
        """Orchestrator ReAct 的 user prompt: 任务 + 偏好"""
        return (
            f"## 你的任务\n"
            f"为下面这个候选热点产出一篇高质量财经分析文章,经 Reviewer 审核通过后即完成。\n\n"
            f"## 候选热点(top-1,评分最高)\n"
            f"- subject: **{top1.get('subject')}**\n"
            f"- score: {top1.get('score'):.3f}\n"
            f"- confidence: {top1.get('confidence')}\n"
            f"- industry: {top1.get('industry', '?')}\n"
            f"- anomaly_types: {top1.get('anomaly_types', [])}\n"
            f"- description: {top1.get('description', '')}\n"
            f"- related_symbols: {top1.get('related_symbols', [])[:5]}\n\n"
            f"## Orchestrator 偏好\n"
            f"{pref_str}\n\n"
            f"## 行动指引\n"
            f"1. 先调 call_researcher,subject = \"{top1.get('subject')}\", "
            f"industry = \"{top1.get('industry', '')}\"\n"
            f"2. 拿到 research_brief 后,调 call_writer\n"
            f"3. 拿到 article 后,调 call_reviewer\n"
            f"4. 根据 review 决定下一步,直到 passed=true\n"
            f"5. 最终输出「TASK_COMPLETE」\n"
        )

    # ============= 2026-08-05 新增:自我反思机制 =============

    def _snapshot_article(self, article: dict, review: dict | None) -> dict:
        """提取 article 的"指纹",供下轮 self-reflection 对比

        维度:
        - word_count: 字数
        - score: 评分
        - passed: 是否通过
        - issue_kinds: issues 的种类([LENGTH] / [FACTS] / [CONTEXT] / [DATA])
        - length_target_used: 当时传的 length_target
        - style_hint_used: 当时传的 style_hint 摘要
        """
        snap = {
            "word_count": article.get("word_count", 0) or len(article.get("content", "")),
            "score": (review or {}).get("score", 0),
            "passed": (review or {}).get("passed", False),
            "issue_kinds": [],
            # 2026-08-05:writer 把覆盖参数存到 article,这里读出来供反思
            "length_target_used": article.get("_length_target") or article.get("_length_override"),
            "style_hint_used": (
                article.get("_style_hint")
                or article.get("_style_override")
                or ""
            )[:80],
        }
        for issue in (review or {}).get("issues", []):
            # 抽取 [XXX] 前缀
            import re as _re
            m = _re.match(r"\[(\w+)\]", issue)
            if m:
                snap["issue_kinds"].append(m.group(1))
        return snap

    def _build_reflection_block(
        self, prev_snap: dict, prev_review: dict | None, prev_feedback: str
    ) -> str:
        """构建 self-reflection 块,让 LLM 观察 delta 后决定下一步策略

        关键设计:不是给 LLM 一个固定答案,而是把它"上轮的策略"和"上轮的结果"摆出来,
        让它自己推理:
          1. 上轮我做了什么?
          2. 上轮结果是什么?
          3. delta 显示策略是否有效?
          4. 下一步该调整什么?
        """
        cur_word = prev_snap["word_count"]
        cur_score = prev_snap["score"]
        issue_kinds = prev_snap["issue_kinds"]
        used_lt = prev_snap.get("length_target_used")
        used_sh = prev_snap.get("style_hint_used", "")
        prev_passed = prev_snap["passed"]

        # 1) 列出上轮动作
        action_lines = []
        if used_lt is not None:
            action_lines.append(f"  - 传了 `length_target={used_lt}`")
        if used_sh:
            action_lines.append(f"  - 传了 `style_hint` 摘要: {used_sh!r}")
        if not action_lines:
            action_lines.append("  - (无参数覆盖,完全由 LLM 默认决策)")
        action_block = "\n".join(action_lines) if action_lines else "  - (无)"

        # 2) 自动推断反弹模式(给 LLM 一个 hint,但不强制)
        warning = ""
        if "LENGTH" in issue_kinds:
            if used_lt is None:
                warning = (
                    f"⚠️ **本轮字数 = {cur_word},且你上轮没传 length_target**。\n"
                    f"这就是字数失控的根因。**这轮必须传 length_target**。\n"
                )
            elif used_lt and cur_word > used_lt * 1.2:
                used_sh_safe = used_sh.replace('"', "'")
                warning = (
                    f"⚠️ **本轮字数 = {cur_word},你上轮传了 length_target={used_lt} 但 article 仍然超长**。\n"
                    f"说明你的 style_hint({used_sh_safe!r}) 给了「加内容」的指令,与 length_target 冲突。\n"
                    f"**这轮修法**:\n"
                    f"  - 只传 length_target={used_lt},**不传 style_hint**\n"
                    f"  - 或 style_hint 改成「严格精简到 {used_lt} 字以内,不要扩充内容」\n"
                )
            elif used_lt and cur_word < used_lt * 0.7:
                warning = (
                    f"⚠️ **本轮字数 = {cur_word},远低于 length_target={used_lt}**。\n"
                    f"可能精简过度了。\n"
                )
        if "FACTS" in issue_kinds and not used_sh:
            warning += (
                f"⚠️ 有关键事实未体现([FACTS]),但你上轮没传 style_hint 提醒 writer 补事实。\n"
                f"**这轮传 style_hint 列出缺失的事实**。\n"
            )

        # 3) 输出反思块
        return (
            f"\n\n## 🪞 上轮修复自检(2026-08-05 新增,你必须先观察再行动)\n"
            f"\n"
            f"**上轮 article 指标**:\n"
            f"  - 字数: {cur_word}\n"
            f"  - 评分: {cur_score}/100\n"
            f"  - 通过: {prev_passed}\n"
            f"  - issue 类型: {issue_kinds if issue_kinds else '(无)'}\n"
            f"\n"
            f"**上轮你调 writer 时传了什么**:\n"
            f"{action_block}\n"
            f"\n"
            f"**自动推断的根因**(仅供参考,你也可以自己推理):\n"
            f"{warning if warning else '(无明显反弹模式,可按 issue 类型正常选工具)'}\n"
            f"\n"
            f"**请先思考再决策**(不要无脑套规则):\n"
            f"  1. 上轮我的策略 vs 上轮结果,差距在哪里?\n"
            f"  2. 这个差距的原因是什么?(参数没传对? style_hint 与 length_target 冲突? writer 没遵守?)\n"
            f"  3. 这轮怎么调整?(只调一个参数 / 换工具 / 同时调?)\n"
        )

