"""
Reviewer Agent - 审核(LLM-as-a-Judge 增强)
=============================================

职责:对生成的文章做合规、事实、风格审核

两层审核:
1. 规则层(快速、确定):禁用词、字数、合规声明等
2. LLM 层(深度、智能):内容质量、深度、风格

输出: review (dict) - 通过 / 退回修改 + 审核意见
"""

from __future__ import annotations
import logging
import json
import re
from datetime import datetime

from .base import BaseAgent, AgentResult
from memory import get_semantic_memory
from tools import get_llm, is_llm_available
import config

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    name = "reviewer"
    
    def __init__(self):
        super().__init__()
        self.sm = get_semantic_memory()
        self.llm = get_llm()
        self.use_llm = is_llm_available()
    
    def _run(self, input_data: dict | None) -> AgentResult:
        drafts = (input_data or {}).get("drafts")
        if not drafts:
            drafts = self.wm.get("drafts", [])
        
        if not drafts:
            return AgentResult(
                agent_name=self.name, success=False,
                errors=["no drafts to review"],
            )
        
        # 检查哪些 draft 已经被 review 过(Refinement 阶段会触发)
        reviewed_ids = {
            r.get("article_id") 
            for r in self.wm.get("reviews", [])
            if r.get("article_id")
        }
        
        reviews = []
        for draft in drafts:
            # 如果这个 draft 已经被 review 过(且是 refine 循环里生成的),跳过
            if draft.get("article_id") in reviewed_ids:
                # 但如果是 refine 后的最新版,需要再 review 一次
                if not draft.get("revised"):
                    continue
                # revised=True 的是 refine 后的版本,需要重新 review
                # (这种情况:refine 循环里的 review 已经被存,这里再 review 一次)
            
            review = self._review_draft(draft)
            reviews.append(review)
            self.wm.append("reviews", review)
            if review["passed"]:
                self.wm.append("published", draft)
        
        passed = sum(1 for r in reviews if r["passed"])
        return AgentResult(
            agent_name=self.name,
            success=passed > 0,
            items=reviews,
            metrics={
                "total": len(reviews),
                "passed": passed,
                "rejected": len(reviews) - passed,
                "pass_rate": passed / len(reviews) if reviews else 0,
                "judge_mode": "llm+rule" if self.use_llm else "rule_only",
            },
        )
    
    def _review_draft(self, draft: dict) -> dict:
        """审核单篇 draft(2026-08-04 改造:Plan-and-Execute + 字数 + 关键事实覆盖)"""
        issues = []
        score = 100

        content = draft.get("content", "")
        title = draft.get("title", "")

        # ---- 1. 规则层(快速、确定) — 字数检查已挪到 step 3 tool ----
        rule_result = self._rule_check(content, title)
        issues.extend(rule_result["issues"])
        score -= rule_result["deduction"]

        # ---- 2. LLM 层:Plan-and-Execute ----
        llm_result = None
        checklist = None
        if self.use_llm:
            try:
                checklist = self._make_review_checklist(draft)
                logger.info(
                    f"审核 checklist: {len(checklist)} 项" if checklist else "checklist 生成失败"
                )
                llm_result = self._llm_judge_with_checklist(draft, checklist)
                if llm_result and not llm_result.get("passed", True):
                    for issue in llm_result.get("issues", []):
                        if issue not in issues:
                            issues.append(f"[LLM] {issue}")
                    score -= 15
            except Exception as e:
                self.logger.warning(f"LLM judge failed: {e}")

        # ---- 3. 字数检查(2026-08 改造:作为 tool 调,失败信息回传给 Orchestrator) ----
        length_check = self._check_length_as_tool(draft)
        if not length_check.get("passed"):
            msg = length_check.get("message", "")
            if msg and msg not in issues:
                issues.append(f"[LENGTH] {msg}")
                score -= 5  # 字数违规扣 5 分

        # ---- 4. 关键事实覆盖检查(2026-08-04 新增) ----
        # 思路:从 article.brief_id 读 brief,拿到 key_facts,
        # 逐条检查 article 里有没有体现(规则层抽关键词 + LLM 兜底)
        facts_check = self._check_key_facts_coverage(draft, content)
        if facts_check.get("missing_facts"):
            for f in facts_check["missing_facts"]:
                issue = f"[FACTS] 关键事实未体现: {f}"
                if issue not in issues:
                    issues.append(issue)
                    score -= 8  # 缺失关键事实扣 8 分(中等)

        # ---- 5. brief 数据完整性检查(2026-08-05 新增) ----
        # 区分「brief 里就有,writer 没用」vs「brief 里就没,writer 没法用」
        # 触发条件:
        #   [CONTEXT] industry_context 是空 / 默认「暂无」→ 信息提示(不 block),
        #             Orchestrator 可选调 call_researcher 补
        #   [DATA]    tool_data 缺关键类别(limit_up/board_change/financial_report/news)
        #             → 严重,block pass
        context_data_check = self._check_brief_data_completeness(draft)
        if context_data_check.get("missing_context"):
            issue = f"[CONTEXT] 行业背景信息缺失,需调 researcher 补 research"
            if issue not in issues:
                issues.append(issue)
                # 2026-08-05:不扣分(因为 industry_context 经常缺,扣分会永远 block)
                # 只在 issues 里出现,Orchestrator 可以选择处理
        if context_data_check.get("missing_data"):
            for kind in context_data_check["missing_data"]:
                issue = f"[DATA] 研究数据不完整:缺少 {kind},需调 researcher 补 research"
                if issue not in issues:
                    issues.append(issue)
                    score -= 5  # 缺失数据扣 5 分(中等,3 分以上还能 pass)

        # 通过条件(2026-08-05 收紧):必须同时满足
        # 1. 分数 >= min_score
        # 2. 无禁用词(致命)
        # 3. 字数检查通过(确定性指标,不再容忍 [LENGTH] issue)
        # 4. LLM judge 没判未通过(LLM 是软指标,但仍要尊重)
        # 5. 关键事实全部覆盖(2026-08-04 新增)
        # 6. tool_data 完整(2026-08-05 新增:只有 [DATA] 才 block,[CONTEXT] 只提示)
        # 7. industry_context 缺失(2026-08-05 新增:不 block,只是提示)
        llm_passed = (llm_result or {}).get("passed", True)
        length_passed = length_check.get("passed", True)
        facts_covered = facts_check.get("all_covered", True)
        data_ok = not context_data_check.get("missing_data")
        passed = (
            score >= config.LLM_REFINEMENT_MIN_SCORE
            and not any("禁用词" in i for i in issues)
            and length_passed         # 2026-08-04 新增:字数必须过
            and llm_passed            # 2026-08-04 新增:LLM judge 也得过
            and facts_covered         # 2026-08-04 新增:关键事实必须覆盖
            and data_ok               # 2026-08-05 新增:关键数据缺失会 block
            # 注意:[CONTEXT] 不 block,Orchestrator 可选处理
        )

        return {
            "review_id": f"rev_{int(datetime.now().timestamp())}_{hash(content[:50]) % 10000}",
            "article_id": draft.get("article_id"),
            "passed": passed,
            "score": max(0, score),
            "issues": issues,
            "rule_check": rule_result,
            "facts_check": facts_check,  # 2026-08-04 新增
            "context_data_check": context_data_check,  # 2026-08-05 新增
            "llm_check": llm_result,
            "length_check": length_check,  # 2026-08 新增:tool 调用结果
            "checklist": checklist,
            "review_summary": self._summarize(passed, score, issues),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }

    def _check_length_as_tool(self, draft: dict) -> dict:
        """调 check_article_length tool(2026-08 新增)

        把字数检查从规则层移到独立 tool 调用,
        这样 Reviewer → Orchestrator 闭环里,Orchestrator 能直接看到 [LENGTH] 字数过多 issue。
        """
        try:
            from tools.dummy_tools import _check_article_length
            return _check_article_length(
                article=draft,
                target_length=self.sm.get_preferred_length(),
                tolerance=0.3,
            )
        except Exception as e:
            self.logger.warning(f"字数检查 tool 失败: {e}")
            return {"passed": True, "message": "", "error": str(e)}

    # ============= 关键事实覆盖检查(2026-08-04 新增) =============

    def _check_key_facts_coverage(self, draft: dict, content: str) -> dict:
        """检查文章是否覆盖 brief 里所有关键事实

        流程:
        1. 从 article.brief_id 读 brief → 拿 key_facts
        2. 每条 key_fact 抽关键词(中文 2-4 字 / 数字+单位 / 股票名)
        3. 规则层:任一关键词命中 → 视为覆盖
        4. LLM 兜底(对没命中的):让 LLM 判断是否语义覆盖
        5. 返回 {all_covered, missing_facts, covered_facts, details}

        失败处理:读不到 brief → 直接视为全部覆盖(避免 block)
        """
        result = {
            "all_covered": True,
            "total_facts": 0,
            "covered_count": 0,
            "missing_facts": [],
            "covered_facts": [],
            "details": [],
        }

        # 1. 拿 key_facts
        key_facts = self._get_key_facts(draft)
        if not key_facts:
            return result  # 没 brief / 没 facts,不 block
        result["total_facts"] = len(key_facts)

        # 2. 规则层检查
        for fact in key_facts:
            if not isinstance(fact, str) or not fact.strip():
                continue
            keywords = self._extract_fact_keywords(fact)
            hits = [k for k in keywords if k in content]
            if hits:
                result["covered_count"] += 1
                result["covered_facts"].append(fact)
                result["details"].append({
                    "fact": fact,
                    "covered": True,
                    "method": "rule",
                    "hits": hits[:3],
                })
            else:
                result["missing_facts"].append(fact)
                result["details"].append({
                    "fact": fact,
                    "covered": False,
                    "method": "rule",
                    "hits": [],
                })

        # 3. LLM 兜底:对规则层判定为 missing 的,再问一次 LLM
        # 2026-08-04 修复:只在有部分覆盖 + 文章不短时启用,避免 LLM 过度宽松
        if (
            result["missing_facts"]
            and result["covered_count"] > 0   # 至少覆盖了 1 条,LLM 才有参照
            and len(content) >= 500            # 文章太短,LLM 容易幻觉
        ):
            still_missing = self._llm_facts_check(content, result["missing_facts"])
            if still_missing != result["missing_facts"]:
                # LLM 改判了一些 → 更新
                recovered = set(result["missing_facts"]) - set(still_missing)
                result["missing_facts"] = still_missing
                # 重新统计
                result["covered_count"] = result["total_facts"] - len(still_missing)
                for d in result["details"]:
                    if d["fact"] in recovered:
                        d["covered"] = True
                        d["method"] = "llm"
                        d["hits"] = ["(LLM 语义判定)"]

        result["all_covered"] = len(result["missing_facts"]) == 0
        return result

    def _get_key_facts(self, draft: dict) -> list[str]:
        """从 article.brief_id 读 brief,拿 key_facts

        备选:article 自身可能带 key_facts 字段
        """
        # 备选 1:article 自身带
        if draft.get("key_facts"):
            return draft["key_facts"]

        # 备选 2:从 brief 读
        brief_id = draft.get("brief_id")
        if not brief_id:
            return []
        try:
            from tools.persist import load_brief
            brief = load_brief(brief_id)
            if brief:
                return brief.get("key_facts", []) or []
        except Exception as e:
            self.logger.warning(f"读 brief({brief_id}) 失败: {e}")
        return []

    def _extract_fact_keywords(self, fact: str) -> list[str]:
        """从 key_fact 抽关键搜索词(2026-08-04 v2:收紧,避免结构性短语误判)

        策略:
        1. 冒号后内容(核心信息)优先保留
        2. 数字 + 单位(最直接,优先保留)
        3. 4 字中文片段(避开"板块内多股共振"这种结构性短语)
        4. 通用词过滤
        """
        import re
        keywords = []
        stop_words = {
            "板块", "个股", "涨停", "跌停", "共振", "数据", "情况", "以及", "等",
            "公司", "今日", "最新", "未知", "板块内", "的", "了", "在", "是",
            "新闻", "提示", "已", "已覆盖", "未覆盖",
        }

        # 1) 冒号后内容(整段保留,如「风范股份, 汇金通」)
        # 冒号前是结构性标签(板块内/新闻/未知),冒号后才是真事实
        fact_core = fact
        for sep in ["：", ":"]:
            if sep in fact:
                tail = fact.split(sep, 1)[1].strip()
                # 按逗号拆,每段作为独立关键词
                if 2 <= len(tail) <= 60:
                    fact_core = tail
                    # 单个元素(股票名/公司名)按 , 拆
                    for piece in re.split(r"[,，、;；]", tail):
                        piece = piece.strip()
                        if 2 <= len(piece) <= 20:
                            keywords.append(piece)
                break

        # 2) 数字 + 单位
        for m in re.finditer(r"(\d+\.?\d*\s*[亿万元%]+|\d+\.\d+|\d{4,})", fact):
            kw = m.group(0).strip()
            if len(kw) >= 2 and kw not in keywords:
                keywords.append(kw)

        # 3) 4 字中文片段(避开"板块内多股"这种 2-3 字结构性短语)
        for m in re.finditer(r"[\u4e00-\u9fa5]{4,6}", fact_core):
            kw = m.group(0)
            if kw in stop_words:
                continue
            if kw not in keywords:
                keywords.append(kw)

        # 4) 兜底:3 字中文(只在没有其他关键词时)
        if not keywords:
            for m in re.finditer(r"[\u4e00-\u9fa5]{3}", fact_core):
                kw = m.group(0)
                if kw in stop_words:
                    continue
                if kw not in keywords:
                    keywords.append(kw)
                    if len(keywords) >= 3:
                        break

        # 5) 整句兜底(防极端情况)
        if not keywords and 4 <= len(fact_core) <= 50:
            keywords.append(fact_core)

        return keywords

    def _llm_facts_check(self, content: str, missing_facts: list[str]) -> list[str]:
        """LLM 兜底:对规则层判定为 missing 的事实,语义层面再判定

        2026-08-04 v2:收紧 prompt,避免 LLM 过度宽松

        Returns:
            仍然 missing 的事实列表(LLM 改判覆盖的就剔除)
        """
        if not missing_facts or not self.use_llm:
            return missing_facts

        try:
            # 截断 article 防 token 爆
            content_trunc = content[:2500] if len(content) > 2500 else content
            facts_str = "\n".join(f"{i+1}. {f}" for i, f in enumerate(missing_facts))

            prompt = (
                f"你是严格的事实覆盖审核员。\n\n"
                f"## 关键事实清单(这些是 brief 里的关键事实,文章必须体现)\n"
                f"{facts_str}\n\n"
                f"## 文章内容(节选)\n{content_trunc}\n\n"
                f"## 任务\n"
                f"对每条关键事实,判断文章是否**明确提到或用同义表达**了该事实的核心信息。\n"
                f"判断标准(严格):\n"
                f"- 文章必须出现事实中的**具体数据/具体名称/具体数字**\n"
                f"- 仅出现泛泛的同主题描述不算覆盖(例如事实要求「300615 复牌」,文章只说「某股票可能复牌」不算)\n"
                f"- 文章中没有任何与该事实相关的内容 → 未覆盖\n\n"
                f"## 输出格式(严格 JSON 数组,只列未被覆盖的)\n"
                f"只输出**未被覆盖**的事实原文(用与清单完全一致的字符串)。\n"
                f"全部覆盖 → []\n"
                f"示例:[\"事实2原文\", \"事实4原文\"]\n"
                f"**只输出 JSON 数组,不要其他文字、不要 markdown 包装、不要解释**"
            )

            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600,
                enable_thinking=False,
            )
            response = response.strip()
            # 解析
            import re as _re
            # 优先找 ```json ... ``` 块
            m = _re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, _re.DOTALL)
            if not m:
                m = _re.search(r"\[(.*?)\]", response, _re.DOTALL)
            if m:
                try:
                    arr_str = m.group(1) if m.lastindex else m.group(0)
                    if not arr_str.startswith("["):
                        arr_str = "[" + arr_str + "]"
                    arr = json.loads(arr_str)
                    if isinstance(arr, list):
                        # 只保留原 missing_facts 里实际存在的(防止 LLM 幻觉)
                        still_missing = [f for f in arr if isinstance(f, str) and f in missing_facts]
                        return still_missing
                except (json.JSONDecodeError, IndexError):
                    pass
        except Exception as e:
            self.logger.warning(f"LLM facts check 失败: {e}")

        return missing_facts  # 兜底:维持原判

    # ============= brief 数据完整性检查(2026-08-05 新增) =============

    def _check_brief_data_completeness(self, draft: dict) -> dict:
        """检查 brief 本身的数据完整性(区分 [FACTS] 和 [CONTEXT]/[DATA])

        目的:让 Orchestrator 知道「数据缺失」和「数据没用上」是两件事:
        - [FACTS] brief 里有,但 article 没体现 → 调 call_writer 改 style_hint
        - [CONTEXT] / [DATA] brief 里就没有 → 调 call_researcher 补数据

        Returns:
            {
                "missing_context": bool,  # industry_context 空 / 默认值
                "missing_data": list[str],  # 缺失的 tool_data 类别
                "context_value": str,  # 实际 industry_context
                "data_keys": list[str],  # 实际 tool_data 的 keys
            }
        """
        result = {
            "missing_context": False,
            "missing_data": [],
            "context_value": "",
            "data_keys": [],
        }

        # 1) 读 brief
        brief_id = draft.get("brief_id")
        if not brief_id:
            # 没有 brief → 没法判断,放行(避免误报)
            return result
        try:
            from tools.persist import load_brief
            brief = load_brief(brief_id)
        except Exception as e:
            self.logger.warning(f"_check_brief_data_completeness 读 brief 失败: {e}")
            return result
        if not brief:
            return result

        # 2) 检查 industry_context
        ctx = brief.get("industry_context", "") or ""
        result["context_value"] = ctx.strip()
        # 触发条件:空 / 默认提示语 / None
        if not result["context_value"] or result["context_value"] in {
            "暂无行业背景信息。",
            "暂无行业背景信息",
            "(无)",
            "",
        }:
            result["missing_context"] = True

        # 3) 检查 tool_data 关键类别
        # 期望 4 类:limit_up / board_change / financial_report / news
        # (对应 Orchestrator 决策用的盘面/异动/财务/新闻)
        EXPECTED_DATA_KINDS = {
            "limit_up": "涨停股池(盘面)",
            "board_change": "板块异动数据",
            "financial_report": "财务三表",
            "news": "相关新闻",
        }
        tool_data = brief.get("tool_data", {}) or {}
        result["data_keys"] = list(tool_data.keys())
        for kind, label in EXPECTED_DATA_KINDS.items():
            if kind not in tool_data or not tool_data[kind]:
                # 检查是否完全空(0 条)
                val = tool_data.get(kind)
                if not val:  # None / [] / {} / ""
                    result["missing_data"].append(f"{label}({kind})")

        return result

    # ============= Plan-and-Execute 审核(2026-08 新增) =============

    def _make_review_checklist(self, draft: dict) -> list[dict]:
        """调 LLM 生成审核 checklist

        Returns:
            list of {"category": str, "check": str, "weight": int}
        """
        title = draft.get("title", "")
        content = draft.get("content", "")
        if len(content) > 2000:
            content = content[:2000] + "\n...(截断)"

        checklist_prompt = (
            f"你是财经文章审核专家。请为以下文章生成审核 checklist(8-10 项)。\n\n"
            f"## 文章标题\n{title}\n\n"
            f"## 文章摘要(前 2000 字)\n{content}\n\n"
            f"## 任务\n"
            f"为这篇财经文章生成 8-10 项审核 checklist,每项:\n"
            f"- category: 大类(合规性/事实性/深度/客观性/可读性/逻辑性/数据支撑/语言质量)\n"
            f"- check: 审核点(一句话描述)\n"
            f"- weight: 权重 1-10(越高越重要,合规相关应是 8-10)\n\n"
            f"## 输出格式(严格 JSON 数组)\n"
            f"[{{\"category\": \"合规性\", \"check\": \"是否包含风险提示段落\", \"weight\": 10}}]\n"
            f"**只输出 JSON 数组,不要其他文字、不要 markdown 包装**"
        )

        try:
            content = self.llm.chat(
                messages=[{"role": "user", "content": checklist_prompt}],
                temperature=0.2,
                max_tokens=1500,
                enable_thinking=False,
            )
            return self._parse_checklist_json(content)
        except Exception as e:
            self.logger.warning(f"生成 checklist 失败: {e}")
        return []

    def _parse_checklist_json(self, content: str) -> list[dict]:
        """解析 checklist JSON(同 plan/outline)"""
        import re
        content = content.strip()
        try:
            obj = json.loads(content)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r"(\[.*?\])", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        return []

    def _llm_judge_with_checklist(self, draft: dict, checklist: list | None) -> dict:
        """按 checklist 审核(Plan-and-Execute 第二步)"""
        if not checklist:
            # 没 checklist → 退回原 _llm_judge
            return self._llm_judge(draft)

        # 把 checklist 加到 prompt 里,让 LLM 按清单逐项审
        title = draft.get("title", "")
        content = draft.get("content", "")
        if len(content) > 3000:
            content = content[:3000] + "\n...(截断)"

        # 构造 checklist 字符串
        checklist_str = "\n".join(
            f"- [{c.get('weight', 5)}] {c.get('category', '?')}: {c.get('check', '?')}"
            for c in checklist
        )

        system = (
            "你是一位资深的财经内容审核专家。\n"
            "请按下方 checklist 逐项审核文章,每项给 0-10 分。\n"
            "最后输出 JSON:\n"
            "{\n"
            '  "scores": {"合规性": 8, "事实性": 7, ...},\n'
            '  "average_score": 7.6,\n'
            '  "passed": true/false,\n'
            '  "issues": ["问题1", "问题2"],\n'
            '  "disclosure_check": {"投资有风险": "已覆盖/未覆盖", "过往业绩不代表未来": "已覆盖/未覆盖"},\n'
            '  "highlights": ["亮点"],\n'
            '  "suggestion": "改进建议"\n'
            "}\n\n"
            "判定:average_score >= 6.0 且 无严重合规问题 = passed\n"
            "必含披露(如「投资有风险」、「过往业绩不代表未来」)允许近义改写,只要核心精神覆盖即可。\n"
            f"## 审核 checklist(本文章)\n{checklist_str}"
        )
        user = (
            f"标题:{title}\n"
            f"主题:{draft.get('subject', '')}\n"
            f"字数:{draft.get('word_count', 0)}\n\n"
            f"---\n{content}\n---"
        )

        return self.llm.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=config.LLM_REVIEWER_MAX_TOKENS,
            enable_thinking=False,
        )
    
    # ============= 规则层 =============

    def _normalize_compliance_text(self, text: str) -> str:
        """合规关键词归一化

        去掉语气词 / 副词 / 助词,使「过往业绩不代表未来」和「过往业绩亦不代表未来表现」等价。
        避免因为加了「亦」「的」「的」等修饰词而被误判为「缺少合规声明」。
        """
        # 去除常见语气/副词/助词
        text = re.sub(r'[亦也的之乎者]', '', text)
        # 去除标点(中英文常见标点)
        punct = "，。、！？：；（）()【】《》\u201c\u201d\u2018\u2019\"' ,.!?;:()<>[]{}"
        text = re.sub("[" + re.escape(punct) + "]", "", text)
        # 去除多余空白
        text = re.sub(r'\s+', '', text)
        return text

    def _has_compliance_phrase(self, phrase: str, content: str) -> bool:
        """宽松判断:content 是否包含 phrase(归一化后,允许最多 15 字间隔)

        解决:
        - 语气词差异:「过往业绩亦不代表未来」vs「过往业绩不代表未来」
        - 间隔词:「过往业绩板块表现不代表未来」vs「过往业绩不代表未来」
        - 同时确保顺序敏感(「未来不代表过往」不命中)
        """
        n_phrase = self._normalize_compliance_text(phrase)
        n_content = self._normalize_compliance_text(content)
        # 短语太短就用直接子串
        if len(n_phrase) < 4:
            return n_phrase in n_content
        # 转成正则:每个字符之间允许 0-15 字间隔(吸收语气词 / 修饰词)
        pattern = ".{0,15}".join(re.escape(c) for c in n_phrase)
        return bool(re.search(pattern, n_content))

    def _rule_check(self, content: str, title: str) -> dict:
        """规则层审核"""
        issues = []
        deduction = 0
        
        # 必填项
        if not content or len(content) < 200:
            issues.append("内容过短")
            deduction += 30
        if not title:
            issues.append("标题缺失")
            deduction += 10
        
        # 禁用词
        forbidden = self.sm.get_forbidden_words()
        for word in forbidden:
            if word in content:
                issues.append(f"包含禁用词:{word}")
                deduction += 20
        
        # 必含合规声明:放到 LLM 层做语义判断(硬规则不擅长近义改写)
        # 必须披露的关键词存到 draft 上,让 LLM judge 读得到
        # 注意:这里不下扣分,LLM judge 会在合规维度独立扣分

        # 风险提示章节(基础检查:有"风险"或"风险提示"字样即可)
        if "风险" not in content:
            issues.append("缺少风险提示")
            deduction += 15

        # 字数检查已挪到 _check_length_as_tool(2026-08 改造)

        # 标题质量
        if title:
            if len(title) > 60:
                issues.append("标题过长")
                deduction += 5
            if title.count("?") + title.count("!") > 3:
                issues.append("标题标点过多")
                deduction += 5
            if not re.search(r"[\u4e00-\u9fa5]", title):
                issues.append("标题应包含中文")
                deduction += 5
        
        return {
            "issues": issues,
            "deduction": deduction,
        }
    
    # ============= LLM-as-a-Judge =============
    
    def _llm_judge(self, draft: dict) -> dict:
        """用 LLM 审核内容质量(深度评估)"""
        messages = self._build_judge_prompt(draft)
        result = self.llm.chat_json(
            messages,
            temperature=0.3,
            max_tokens=config.LLM_REVIEWER_MAX_TOKENS,
        )
        return result
    
    def _build_judge_prompt(self, draft: dict) -> list[dict]:
        """构建 LLM judge 的 prompt - 用 JSON 模式"""
        # 把 must_disclose 列表塞到 prompt,让 LLM 做语义判断
        must_disclose = self.sm._reader_profile.get("must_disclose", [])
        disclosure_str = "\n".join(f"- {p}" for p in must_disclose) if must_disclose else "(无)"

        system = (
            "你是一位资深的财经内容审核专家,负责审核财经文章的合规性、深度和客观性。\n\n"
            "请从以下维度评估文章(每项 0-10 分):\n"
            "1. **合规性**:是否包含合规风险、煽动性词汇、明确投资建议\n"
            "2. **事实性**:数据是否可追溯、逻辑是否清晰\n"
            "3. **深度**:是否提供独到分析,而非简单罗列\n"
            "4. **客观性**:是否多角度呈现,避免单边观点\n"
            "5. **可读性**:结构是否清晰、语言是否通顺\n\n"
            "## 必含披露(语义判断,允许近义改写)\n"
            "下列披露是合规要求,文章需要在语义上覆盖这些要点。注意:\n"
            "- 「投资有风险」可以表达为「市场有风险,投资需谨慎」/「股市有风险」等近义说法\n"
            "- 「过往业绩不代表未来」可以表达为「过往表现不代表未来收益」/「历史业绩不代表未来」等\n"
            "- 只要核心精神被覆盖即可,不必逐字一致\n\n"
            f"{disclosure_str}\n\n"
            "请用 JSON 格式输出:\n"
            "{\n"
            '  "scores": {"合规性": 8, "事实性": 7, "深度": 6, "客观性": 8, "可读性": 9},\n'
            '  "average_score": 7.6,\n'
            '  "passed": true,\n'
            '  "issues": ["问题1", "问题2"],\n'
            '  "disclosure_check": {"投资有风险": "已覆盖|未覆盖", "过往业绩不代表未来": "已覆盖|未覆盖"},\n'
            '  "highlights": ["亮点1", "亮点2"],\n'
            '  "suggestion": "改进建议"\n'
            "}\n\n"
            "判定标准:\n"
            "- average_score >= 6.0 且 无严重合规问题 = passed\n"
            "- 如果「必含披露」中任何一项是「未覆盖」,必须在 issues 里写「缺少合规声明:<具体项>」\n"
        )

        # 截断内容防止超过 token 限制
        content = draft.get("content", "")
        if len(content) > 3000:
            content = content[:3000] + "\n...(内容过长已截断)"

        user = (
            f"请审核以下文章:\n\n"
            f"标题:{draft.get('title', '')}\n"
            f"主题:{draft.get('subject', '')}\n"
            f"字数:{draft.get('word_count', 0)}\n"
            f"生成模式:{draft.get('mode', 'unknown')}\n\n"
            f"---\n{content}\n---"
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    
    # ============= 汇总 =============
    
    def _summarize(self, passed: bool, score: int, issues: list) -> str:
        if passed:
            return f"审核通过(分数 {score}/100)"
        else:
            return f"审核未通过(分数 {score}/100):" + "; ".join(issues[:3])
