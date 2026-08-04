"""
Evaluator - 系统评估
=====================

三层评估:
1. 热点发现质量(系统有效性)
2. 内容质量(生产有效性)
3. 业务指标(商业价值)

支持:
- 离线评估:基于本次运行的 metrics + 历史数据
- 报告输出:Markdown 报告
"""

from __future__ import annotations
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import Counter

from memory import get_working_memory, get_episodic_memory
import config

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self):
        self.wm = get_working_memory()
        self.em = get_episodic_memory()
    
    def evaluate_run(self) -> dict:
        """对当前 run 做全面评估"""
        signals = self.wm.get("signals", [])
        anomalies = self.wm.get("anomalies", [])
        candidates = self.wm.get("hotspot_candidates", [])
        topics = self.wm.get("topics", [])
        researches = self.wm.get("researches", [])
        drafts = self.wm.get("drafts", [])
        reviews = self.wm.get("reviews", [])
        published = self.wm.get("published", [])
        refinement_history = self.wm.get("refinement_history", [])

        # ---- 1. 热点发现质量 ----
        discovery = self._eval_discovery(signals, anomalies, candidates)

        # ---- 2. 内容质量 ----
        content_q = self._eval_content(drafts, reviews, refinement_history)

        # ---- 3. 业务指标 ----
        business = self._eval_business(topics, drafts, reviews, published)

        # ---- 4. 系统健康 ----
        system = self._eval_system_health()

        # ---- 5. 迭代修订轨迹 ----
        refinement = self._eval_refinement(refinement_history)

        result = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "discovery": discovery,
            "content": content_q,
            "business": business,
            "system": system,
            "refinement": refinement,
            "summary": self._summary(discovery, content_q, business),
        }
        return result
    
    def _eval_discovery(self, signals, anomalies, candidates) -> dict:
        # 信号到异动的转化率
        sig_to_anom = len(anomalies) / len(signals) if signals else 0
        # 异动到候选的转化率
        anom_to_cand = len(candidates) / len(anomalies) if anomalies else 0
        # 候选的高/中/低分布
        high = sum(1 for c in candidates if c.get("confidence") == "high")
        mid = sum(1 for c in candidates if c.get("confidence") == "mid")
        low = sum(1 for c in candidates if c.get("confidence") == "low")
        # 评分分布
        scores = [c.get("score", 0) for c in candidates]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        return {
            "signal_count": len(signals),
            "anomaly_count": len(anomalies),
            "candidate_count": len(candidates),
            "signal_to_anomaly_rate": round(sig_to_anom, 3),
            "anomaly_to_candidate_rate": round(anom_to_cand, 3),
            "high_conf": high,
            "mid_conf": mid,
            "low_conf": low,
            "avg_score": round(avg_score, 3),
            "max_score": round(max_score, 3),
        }
    
    def _eval_content(self, drafts, reviews, refinement_history=None) -> dict:
        if not drafts:
            return {"draft_count": 0}
        word_counts = [d.get("word_count", 0) for d in drafts]
        review_scores = [r.get("score", 0) for r in reviews]
        passed = sum(1 for r in reviews if r.get("passed"))
        # 修订相关
        revised_drafts = [d for d in drafts if d.get("revised")]
        refinement_history = refinement_history or []
        return {
            "draft_count": len(drafts),
            "avg_word_count": int(sum(word_counts) / len(word_counts)) if word_counts else 0,
            "min_word_count": min(word_counts) if word_counts else 0,
            "max_word_count": max(word_counts) if word_counts else 0,
            "review_passed": passed,
            "review_pass_rate": round(passed / len(reviews), 3) if reviews else 0,
            "avg_review_score": round(sum(review_scores) / len(review_scores), 1) if review_scores else 0,
            "revised_count": len(refinement_history),
            "revised_draft_count": len(revised_drafts),
        }

    def _eval_refinement(self, refinement_history) -> dict:
        """评估 Writer-Reviewer 迭代修订"""
        if not refinement_history:
            return {"enabled": False, "history": []}
        # 按 article_id 分组
        by_article = {}
        for r in refinement_history:
            aid = r.get("article_id", "")
            by_article.setdefault(aid, []).append(r)
        # 统计每篇文章修订轮数
        rounds_per_article = {aid: len(rs) for aid, rs in by_article.items()}
        return {
            "enabled": True,
            "total_revisions": len(refinement_history),
            "articles_refined": len(by_article),
            "max_rounds_per_article": max(rounds_per_article.values()) if rounds_per_article else 0,
            "avg_rounds": round(sum(rounds_per_article.values()) / len(rounds_per_article), 2) if rounds_per_article else 0,
            "history": refinement_history,
        }
    
    def _eval_business(self, topics, drafts, reviews, published) -> dict:
        # 简化版业务指标
        return {
            "topic_count": len(topics),
            "draft_count": len(drafts),
            "published_count": len(published),
            "publish_rate": round(len(published) / len(drafts), 3) if drafts else 0,
        }
    
    def _eval_system_health(self) -> dict:
        # 案例库统计
        em_stats = self.em.stats()
        return {
            "episodic_memory": em_stats,
        }
    
    def _summary(self, discovery, content, business) -> str:
        # 整体评价
        health = "🟢 健康" if discovery.get("candidate_count", 0) > 0 and content.get("review_pass_rate", 0) > 0.5 else "🟡 一般"
        if discovery.get("candidate_count", 0) == 0:
            health = "🔴 无候选"
        return health
    
    def generate_report(self, eval_result: dict, path: Optional[Path] = None) -> Path:
        """生成评估报告(Markdown)"""
        path = path or (config.REPORTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        md = self._render_report(eval_result)
        path.write_text(md, encoding="utf-8")
        logger.info(f"报告已保存: {path}")
        return path
    
    def _render_report(self, r: dict) -> str:
        d = r["discovery"]
        c = r["content"]
        b = r["business"]
        s = r["system"]
        rf = r.get("refinement", {})
        sm = r["summary"]

        md = f"""# 系统运行评估报告

> 生成时间:{r['ts']}
> 整体健康度:**{sm}**

---

## 1. 热点发现质量(系统有效性)

| 指标 | 数值 |
|------|------|
| 信号总数 | {d['signal_count']} |
| 异动总数 | {d['anomaly_count']} |
| 候选热点数 | {d['candidate_count']} |
| 信号→异动 转化率 | {d['signal_to_anomaly_rate']:.1%} |
| 异动→候选 转化率 | {d['anomaly_to_candidate_rate']:.1%} |
| 高确信度候选 | {d['high_conf']} |
| 中确信度候选 | {d['mid_conf']} |
| 低确信度候选 | {d['low_conf']} |
| 平均评分 | {d['avg_score']:.2f} |
| 最高评分 | {d['max_score']:.2f} |

## 2. 内容质量(生产有效性)

| 指标 | 数值 |
|------|------|
| 草稿数 | {c.get('draft_count', 0)} |
| 平均字数 | {c.get('avg_word_count', 0)} |
| 审核通过数 | {c.get('review_passed', 0)} |
| 审核通过率 | {c.get('review_pass_rate', 0):.1%} |
| 平均审核分 | {c.get('avg_review_score', 0):.1f}/100 |
| 修订文章数 | {c.get('revised_count', 0)} |
| 最终修订版草稿数 | {c.get('revised_draft_count', 0)} |

## 3. 业务指标

| 指标 | 数值 |
|------|------|
| 话题数 | {b['topic_count']} |
| 草稿数 | {b['draft_count']} |
| 发布数 | {b['published_count']} |
| 发布率 | {b['publish_rate']:.1%} |

## 4. 系统健康

| 指标 | 数值 |
|------|------|
| 案例库总数 | {s['episodic_memory']['total']} |
| 有反馈案例 | {s['episodic_memory']['with_feedback']} |
| 确认为真热点 | {s['episodic_memory']['confirmed_hotspot']} |
| 已拒绝案例 | {s['episodic_memory']['rejected']} |

## 5. 迭代修订轨迹(Writer-Reviewer 闭环)

"""
        if rf.get("enabled"):
            md += f"""| 指标 | 数值 |
|------|------|
| 触发修订的文章数 | {rf['articles_refined']} |
| 总修订次数 | {rf['total_revisions']} |
| 单篇最多修订轮数 | {rf['max_rounds_per_article']} |
| 平均修订轮数 | {rf['avg_rounds']} |

### 修订详情

"""
            for h in rf["history"]:
                issues = h.get("issues", [])
                issues_str = "; ".join(issues[:3]) if issues else "无"
                if len(issues) > 3:
                    issues_str += f" ...(共{len(issues)}条)"
                md += (
                    f"- **{h.get('subject', '?')}** "
                    f"(第 {h.get('round', 1)} 轮):"
                    f"旧分 {h.get('old_score', 0)} → 新版 {h.get('new_word_count', 0)} 字\n"
                    f"  - 修复:{issues_str}\n"
                )
        else:
            md += "_本次运行未触发迭代修订(所有文章一次性通过审核)。_\n"

        md += f"""
---

## 6. Working Memory 摘要

```
{self.wm.summary()}
```

---

## 7. Top 候选热点

"""
        # 列出 top 候选
        for c in self.wm.get("hotspot_candidates", [])[:10]:
            md += f"- **{c['subject']}** (score={c['score']}, conf={c['confidence']}):{c['description'][:80]}\n"

        return md


_ev_instance = None
def get_evaluator() -> Evaluator:
    global _ev_instance
    if _ev_instance is None:
        _ev_instance = Evaluator()
    return _ev_instance
