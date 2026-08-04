"""
Writer Agent - 内容生成(LLM 驱动)
====================================

职责:把研究简报转化为高质量财经文章

支持两种模式:
- LLM 模式(默认):用 DeepSeek 生成,有 prompt 模板 + 风格控制
- Mock 模式:无 API key 时降级到模板(保持系统可运行)

输出: article (dict)
"""

from __future__ import annotations
import json
import logging
import re
from datetime import datetime

from .base import BaseAgent, AgentResult
from memory import get_semantic_memory
from tools import get_llm, is_llm_available
import config

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    name = "writer"
    
    def __init__(self):
        super().__init__()
        self.sm = get_semantic_memory()
        self.llm = get_llm()
        self.target_length = self.sm.get_preferred_length()
        self.tone = self.sm.get_tone()
        self.forbidden = self.sm.get_forbidden_words()
        self.use_llm = is_llm_available()
        
        if self.use_llm:
            self.logger.info(f"Writer 启用 LLM 模式: {config.LLM_MODEL}")
        else:
            self.logger.warning("Writer 使用 mock 模式(未配置 LLM API key)")
    
    def _run(self, input_data: dict | None) -> AgentResult:
        briefs = (input_data or {}).get("briefs")
        if not briefs:
            briefs = self.wm.get("researches", [])
        
        if not briefs:
            return AgentResult(
                agent_name=self.name, success=False,
                errors=["no research briefs"],
            )
        
        drafts = []
        errors = []
        for brief in briefs:
            try:
                article = self._write_article(brief)
                drafts.append(article)
                self.wm.append("drafts", article)
            except Exception as e:
                errors.append(f"write {brief.get('brief_id', '?')}: {e}")
                self.logger.error(f"write failed: {e}")
        
        return AgentResult(
            agent_name=self.name,
            success=len(drafts) > 0,
            items=drafts,
            metrics={
                "drafted": len(drafts),
                "failed": len(errors),
                "mode": "llm" if self.use_llm else "mock",
            },
            errors=errors,
        )
    
    def _write_article(self, brief: dict) -> dict:
        """根据 brief 生成文章(2026-08 改造:Plan-and-Execute)"""
        # 覆盖 style/length(Orchestrator 可传)
        if brief.get("_style_override"):
            self.tone = brief["_style_override"]
        if brief.get("_length_override"):
            self.target_length = brief["_length_override"]
        # 兼容 agent_subagent_tools 传进来的 length_target 字段
        if brief.get("length_target"):
            self.target_length = brief["length_target"]

        if self.use_llm:
            # Plan-and-Execute:先 LLM 出 outline,再分章节写
            content = self._write_with_plan(brief)
            title = self._generate_title_with_llm(brief) if self.use_llm else self._generate_title(brief)
        else:
            content = self._write_template(brief)
            title = self._generate_title(brief)

        # 合规处理
        content = self._sanitize(content)
        title = self._sanitize(title)

        return {
            "article_id": f"art_{int(datetime.now().timestamp())}_{hash(brief['subject']) % 10000}",
            "brief_id": brief["brief_id"],
            "topic_id": brief.get("topic_id"),
            "subject": brief["subject"],
            "title": title,
            "content": content,
            "word_count": len(content),
            "tone": self.tone,
            "mode": "llm_plan" if self.use_llm else "mock",
            "model": config.LLM_MODEL if self.use_llm else None,
            # 2026-08-05 新增:把覆盖参数存到 article 里,供 Orchestrator self-reflection
            "_length_target": brief.get("_length_override") or self.target_length,
            "_style_hint": brief.get("_style_override") or self.tone,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }

    # ============= Plan-and-Execute 写作(2026-08:整体一次性,不分章节) =============

    def _write_with_plan(self, brief: dict) -> str:
        """Plan-and-Execute 写作(不分章节,整体一次性):

        1. 调 LLM 出"写作策略"(plan):
           - data_points: 必用的关键数据(从 brief.tool_data / key_facts 挑)
           - angle: 写作角度(怎么展现这是个财经热点)
           - must_include: 必含披露/必含结论
        2. 基于 plan,**一次性 chat** 出完整文章
        3. 不分章节,不分段调 LLM,确保字数和风格统一

        优点:
        - 一次写出全文,字数控制更准(LLM 看到完整 target_length)
        - 没有章节拼接断裂感
        - 不会因为多次 chat 累积超字数
        """
        subject = brief.get("subject", "未知")
        # 1. 调 LLM 出写作策略
        strategy = self._make_writing_strategy(brief)
        if not strategy or not isinstance(strategy, dict):
            logger.warning("写作策略生成失败,降级到单次 chat")
            return self._write_with_llm(brief)

        logger.info(
            f"写作策略: angle={strategy.get('angle', '?')[:50]}, "
            f"data_points={len(strategy.get('data_points', []))} 条"
        )

        # 2. 基于 strategy 一次性 chat 出全文
        return self._write_article_with_strategy(brief, strategy)

    def _make_writing_strategy(self, brief: dict) -> dict:
        """调 LLM 生成写作策略(不是章节大纲,而是"数据 + 角度")"""
        subject = brief["subject"]
        strategy_prompt = (
            f"你是财经文章编辑。请为以下话题生成「写作策略」(不要分章节,只要规划整篇怎么写)。\n\n"
            f"## 话题\n"
            f"- 主题:{subject}\n"
            f"- 行业:{brief.get('industry', '?')}\n"
            f"- 异动类型:{brief.get('anomaly_types', [])}\n\n"
            f"## 关键事实(来自研究,可挑选使用)\n"
            + "\n".join(f"- {f}" for f in brief.get("key_facts", [])[:10]) +
            f"\n\n## 任务\n"
            f"为这篇 **{self.target_length} 字左右** 的财经热点分析文章,生成写作策略:\n"
            f"1. **angle** (string): 整篇文章用什么角度切入能让这个热点有看点(一句话)\n"
            f"2. **data_points** (array of string): 必用哪些关键数据/事实(从上面挑 3-5 条最有力的)\n"
            f"3. **must_include** (array of string): 必含披露和必含结论(如「过往业绩不代表未来」/「板块共振信号」等)\n"
            f"4. **tone_hint** (string): 风格提示(数据驱动 / 政策解读 / 资金面分析 / 估值逻辑 等)\n\n"
            f"## 输出格式(严格 JSON 对象,不要 markdown 包装)\n"
            f'{{"angle": "...", "data_points": ["...", "..."], "must_include": ["...", "..."], "tone_hint": "..."}}\n'
            f"**只输出 JSON 对象,不要其他文字**"
        )

        try:
            content = self.llm.chat(
                messages=[{"role": "user", "content": strategy_prompt}],
                temperature=0.3,
                max_tokens=1500,
                enable_thinking=False,
            )
            return self._parse_strategy_json(content)
        except Exception as e:
            logger.error(f"生成写作策略失败: {e}")
        return {}

    def _parse_strategy_json(self, content: str) -> dict:
        """解析写作策略 JSON"""
        import re
        content = content.strip()
        # 尝试 1: 直接 parse
        try:
            obj = json.loads(content)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # 尝试 2: 找 ```json ... ``` 块
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # 尝试 3: 找第一个 { ... } 块
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        logger.warning(f"无法解析 strategy JSON: {content[:200]}")
        return {}

    def _write_article_with_strategy(self, brief: dict, strategy: dict) -> str:
        """基于写作策略,**一次性**调 LLM 生成完整文章

        把 strategy + brief 关键数据塞进 prompt,让 LLM 一次性写出
        {self.target_length} 字的文章(给 LLM 看完整 target,避免分章节累积超字数)
        """
        subject = brief["subject"]
        angle = strategy.get("angle", "")
        data_points = strategy.get("data_points", [])
        must_include = strategy.get("must_include", [])
        tone_hint = strategy.get("tone_hint", "")

        # 构造一次性 prompt
        parts = [
            f"请基于以下研究和写作策略,**一次性**写一篇 **{self.target_length} 字左右** 的财经热点分析文章。",
            f"",
            f"## 文章主题:{subject}",
            f"## 行业:{brief.get('industry', '?')}",
            f"## 异动类型:{brief.get('anomaly_types', [])}",
            f"## 目标字数:**{self.target_length} 字(必须严格控制在该数量级 ±20% 内)**",
            f"",
            f"## 写作策略",
            f"- **切入角度**:{angle}",
            f"- **风格提示**:{tone_hint}",
            f"",
            f"## 必用关键数据/事实",
        ]
        for dp in data_points:
            parts.append(f"- {dp}")

        parts.append("")
        parts.append("## 必含披露/必含结论")
        for mi in must_include:
            parts.append(f"- {mi}")

        # 关键事实补充(给 LLM 自由发挥空间)
        parts.append("")
        parts.append("## 其他可用事实(可选)")
        for f in brief.get("key_facts", [])[:8]:
            parts.append(f"- {f}")

        # 财务数据(若有)
        fin = brief.get("tool_data", {}).get("financial_report", [])
        if fin and isinstance(fin, list):
            parts.append("")
            parts.append("## 财务数据(如有需要)")
            for fr in fin[:2]:
                if isinstance(fr, dict):
                    date = fr.get("报告日", "?")
                    rev = fr.get("营业总收入") or fr.get("营业收入")
                    np_ = fr.get("归属于母公司所有者的净利润") or fr.get("净利润")
                    try:
                        if rev:
                            rev_yi = float(rev) / 1e8
                            line = f"- 报告期 {date}:营收 {rev_yi:.2f} 亿元"
                            if np_:
                                np_yi = float(np_) / 1e8
                                line += f",归母净利 {np_yi:.2f} 亿元"
                            parts.append(line)
                    except (ValueError, TypeError):
                        pass

        parts.append("")
        parts.append("## 输出要求(必须遵守)")
        parts.append(f"1. **字数必须严格控制**:{self.target_length} 字左右,允许 ±20%(即 {int(self.target_length*0.8)}-{int(self.target_length*1.2)} 字)")
        parts.append(f"2. **不要分章节,直接输出完整文章正文**(从导语到风险提示连贯写)")
        parts.append(f"3. **必含披露**完整(过往业绩不代表未来/投资有风险/数据来源等)")
        parts.append(f"4. **不要写标题**(标题会单独生成)")
        parts.append(f"5. **不要写 \"字数控制\" \"按上述要求\"** 等元话语,直接写文章")

        user_prompt = "\n".join(parts)
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]
        content = self.llm.chat(
            messages,
            max_tokens=config.LLM_WRITER_MAX_TOKENS,
            enable_thinking=False,
        )
        return self._post_process(content)

    # ============= LLM 模式 =============
    
    def _build_writer_prompt(self, brief: dict) -> list[dict]:
        """构建 Writer 的 prompt"""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(brief)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    
    def _build_system_prompt(self) -> str:
        """系统提示词 - 角色 + 风格 + 约束"""
        reader_profile = self.sm.get_reader_profile()
        
        tone_desc = {
            "professional": "专业、严谨、有数据支撑,语气克制,适合金融从业者阅读",
            "casual": "通俗易懂、生动有趣,适合普通投资者",
            "academic": "学术化、有理论深度,适合研究机构",
        }.get(self.tone, "专业")
        
        compliance = ""
        if reader_profile.get("compliance_level") == "high":
            compliance = (
                "\n\n【合规要求 - 必须遵守】\n"
                "1. 必须包含风险提示段落\n"
                "2. 不能使用煽动性词汇(如:暴涨/必涨/翻倍/稳赚等)\n"
                "3. 不能给出明确的投资建议(买/卖/加仓/减仓)\n"
                "4. 必须声明'过往业绩不代表未来表现'\n"
                "5. 涉及具体数据时,标注数据来源和时间\n"
            )
        
        return (
            f"你是一位资深的财经内容编辑,擅长把研究材料转化为高质量的财经分析文章。\n\n"
            f"风格:{tone_desc}\n"
            f"目标读者:{reader_profile.get('audience_type', 'professional_investor')}\n"
            f"目标字数:{self.target_length} 字左右\n"
            f"{compliance}\n"
            f"请直接输出文章正文,不要包含标题(标题会单独生成)。"
        )
    
    def _build_user_prompt(self, brief: dict) -> str:
        """用户提示词 - 把研究材料组织好喂给 LLM"""
        parts = [
            f"请基于以下研究材料,写一篇 {self.target_length} 字左右的财经分析文章。",
            f"",
            f"## 主题",
            f"{brief['subject']}",
        ]
        
        if brief.get("industry"):
            parts.append(f"\n## 所属行业\n{brief['industry']}")
        
        parts.append(f"\n## 行业背景\n{brief.get('industry_context', '暂无')}")
        
        if brief.get("key_facts"):
            parts.append(f"\n## 关键事实")
            for f in brief["key_facts"]:
                parts.append(f"- {f}")
        
        # 涉及个股
        symbol_research = brief.get("symbol_research", [])
        if symbol_research:
            parts.append(f"\n## 涉及个股")
            for sr in symbol_research[:3]:
                sym = sr.get("symbol", "")
                info = sr.get("info", {})
                if info:
                    name = info.get("名称", sym)
                    ind = info.get("行业", "")
                    parts.append(f"- {name}({sym}):{ind}")
                else:
                    parts.append(f"- {sym}")
        
        # 研报观点
        research_reports = brief.get("research_reports", [])
        if research_reports:
            parts.append(f"\n## 机构观点")
            for r in research_reports[:2]:
                title = r.get("标题", r.get("name", ""))
                if title:
                    parts.append(f"- {title}")
        
        # 信号描述
        parts.append(f"\n## 异动信号\n{brief.get('research_summary', '')}")
        
        parts.append(f"\n## 文章结构要求")
        parts.append("1. 导语:用 1-2 段点出核心事件和市场关注点")
        parts.append("2. 板块表现:用数据说话,列举异动指标")
        parts.append("3. 行业背景:用知识图谱里的产业链信息")
        parts.append("4. 相关个股:客观陈述,不做主观判断")
        parts.append("5. 机构观点:引用研报和分析师观点")
        parts.append("6. 后市展望:多角度分析,标注不确定性")
        parts.append("7. 风险提示:合规声明")
        
        return "\n".join(parts)
    
    def _write_with_llm(self, brief: dict) -> str:
        """用 LLM 写文章主体"""
        messages = self._build_writer_prompt(brief)
        try:
            content = self.llm.chat(
                messages,
                max_tokens=config.LLM_WRITER_MAX_TOKENS,
            )
            return self._post_process(content)
        except Exception as e:
            self.logger.error(f"LLM write failed, fallback to template: {e}")
            return self._write_template(brief)
    
    def revise_article(
        self,
        article: dict,
        review: dict,
        brief: dict | None = None,
    ) -> dict:
        """根据 review 的 issues 修改文章
        
        Args:
            article: 原文(包含 content/title 等)
            review: Reviewer 的审核结果(包含 issues 列表)
            brief: 可选的原始 brief(用于补充上下文)
        
        Returns:
            修改后的 article(保持原 article 的元数据,更新 content)
        """
        issues = review.get("issues", [])
        if not issues:
            self.logger.debug("no issues to revise, returning original")
            return article
        
        # ---- 构建 prompt ----
        system_prompt = (
            "你是一位资深的财经文章编辑。"
            "请根据审稿人指出的具体问题修改文章,要求:\n"
            "1. **精准修复**:只解决指出的问题,不要大幅改动其他部分\n"
            "2. **保持风格**:保留原文的写作风格、用词、结构\n"
            "3. **保持合规**:不引入新的违规内容、不使用煽动性词汇\n"
            "4. **输出完整文章**:直接输出修改后的完整文章正文,不要解释"
        )
        
        # 截断过长的内容防止 token 爆炸
        original_content = article.get("content", "")
        if len(original_content) > 5000:
            original_content = original_content[:5000] + "\n...(内容过长已截断)"
        
        user_prompt_parts = [
            "## 原文",
            original_content,
            "",
            "## 审稿意见",
            "\n".join(f"{i+1}. {issue}" for i, issue in enumerate(issues)),
            "",
            "## 修改要求",
            "请直接输出修改后的完整文章正文(从导语到风险提示)。",
        ]
        
        # 如果有 brief 上下文,加上
        if brief:
            user_prompt_parts.extend([
                "",
                "## 主题背景(参考)",
                f"主题:{brief.get('subject', '')}",
                f"行业:{brief.get('industry', '未知')}",
            ])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_prompt_parts)},
        ]
        
        # ---- 调 LLM 修改 ----
        try:
            new_content = self.llm.chat(
                messages,
                max_tokens=config.LLM_WRITER_MAX_TOKENS,
                enable_thinking=True,  # 修改时需要思考
                allow_mock_fallback=True,
            )
            new_content = self._post_process(new_content)
            new_content = self._sanitize(new_content)
            
            # 保留原 article 的元数据,只更新 content
            revised = {
                **article,
                "content": new_content,
                "word_count": len(new_content),
                "revised": True,
                "revised_at": datetime.now().isoformat(timespec="seconds"),
                "revision_round": article.get("revision_round", 0) + 1,
                "previous_issues": issues,
                "previous_review_score": review.get("score", 0),
            }
            self.logger.info(
                f"Article '{article.get('subject', '?')}' revised "
                f"(round {revised['revision_round']}, "
                f"fixed {len(issues)} issues)"
            )
            return revised
        except Exception as e:
            self.logger.error(f"revise failed: {e}, returning original")
            return article
    
    def _generate_title_with_llm(self, brief: dict) -> str:
        """用 LLM 生成标题(短输出,关闭 thinking)"""
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位财经文章标题专家。"
                    "请基于给定的文章主题,生成一个吸引人但合规的标题。"
                    "要求:20-40 字,具体明确,不能使用'暴涨''必涨''翻倍'等煽动性词汇。"
                    "只输出标题,不要其他内容。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"主题:{brief['subject']}\n"
                    f"行业:{brief.get('industry', '未知')}\n"
                    f"关键事实:{'; '.join(brief.get('key_facts', [])[:3])}\n"
                    f"评分:{brief.get('score', 0):.2f},确信度:{brief.get('confidence', 'low')}\n"
                ),
            },
        ]
        try:
            # 标题生成:关闭 thinking(标题太短,长思考会撑爆 max_tokens)
            # max_tokens 给 1000,留足 reasoning 空间(deepseek-v4-flash 即使 disable thinking 也会消耗部分 token)
            title = self.llm.chat(
                messages,
                temperature=0.8,
                max_tokens=1000,
                enable_thinking=False,  # 关键:关闭 thinking
                allow_mock_fallback=False,
            )
            # 清理可能的引号/换行
            title = title.strip().strip('"').strip("'").strip("「").strip("」").strip()
            # 截断到合理长度
            if len(title) > 60:
                title = title[:57] + "..."
            return title
        except Exception as e:
            self.logger.warning(f"LLM title failed, fallback to template: {e}")
            return self._generate_title(brief)
    
    def _post_process(self, content: str) -> str:
        """LLM 输出后处理"""
        # 去头尾可能的 markdown 包裹
        content = content.strip()
        if content.startswith("```"):
            first_newline = content.find("\n")
            if first_newline > 0:
                content = content[first_newline+1:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
    
    # ============= Mock 模式(降级) =============
    
    def _write_template(self, brief: dict) -> str:
        """模板化写作(没 LLM 时用)"""
        subject = brief["subject"]
        industry = brief.get("industry", "未知行业")
        industry_ctx = brief.get("industry_context", "")
        key_facts = brief.get("key_facts", [])
        
        lead = f"今日,{subject} 板块在 A 股市场出现明显异动,引发市场关注。\n\n"
        
        section1 = "**一、板块表现**\n\n从盘面来看,"
        section1 += f"{subject} 板块今日表现活跃。"
        for fact in key_facts[:3]:
            section1 += f"{fact}。"
        section1 += "\n\n"
        
        section2 = f"**二、行业背景**\n\n{industry_ctx}\n\n"
        
        symbol_research = brief.get("symbol_research", [])
        section3 = "**三、相关个股**\n\n"
        if symbol_research:
            for sr in symbol_research:
                info = sr.get("info", {})
                if info:
                    name = info.get("名称", sr["symbol"])
                    industry_str = info.get("行业", "")
                    section3 += f"- **{name}({sr['symbol']})**:{industry_str}\n"
                else:
                    section3 += f"- **{sr['symbol']}**\n"
        else:
            section3 += "暂无具体个股数据。\n"
        section3 += "\n"
        
        research_reports = brief.get("research_reports", [])
        section4 = "**四、机构观点**\n\n"
        if research_reports:
            for r in research_reports[:2]:
                title = r.get("标题", r.get("name", ""))
                rating = r.get("评级", "")
                if title:
                    section4 += f"- {title}"
                    if rating:
                        section4 += f"(评级:{rating})"
                    section4 += "\n"
        else:
            section4 += "暂未获取到相关研报。\n"
        section4 += "\n"
        
        section5 = (
            f"**五、后市展望**\n\n"
            f"综合来看,{subject} 板块的异动反映了市场对相关行业的高度关注。"
            f"投资者在关注机会的同时,也应注意以下几点:\n\n"
            f"1. 板块异动可能受到短期资金推动,持续性有待观察\n"
            f"2. 行业基本面是否支持估值修复需要进一步研究\n"
            f"3. 政策面变化可能对行业产生重大影响\n\n"
        )
        
        section6 = (
            f"**风险提示**\n\n"
            f"本文基于公开数据整理,不构成投资建议。"
            f"投资有风险,过往业绩不代表未来表现,入市需谨慎。\n"
        )
        
        return lead + section1 + section2 + section3 + section4 + section5 + section6
    
    def _generate_title(self, brief: dict) -> str:
        subject = brief["subject"]
        summary = brief.get("research_summary", "")
        if "强势" in summary or "爆发" in summary or "突破" in summary:
            return f"{subject}板块异动频频,主力资金抢筹,后市如何演绎?"
        elif "风险" in summary:
            return f"{subject}板块多股跌停,需要警惕什么?"
        else:
            return f"{subject}板块出现异动,关注哪些机会?"
    
    # ============= 合规处理 =============
    
    def _sanitize(self, text: str) -> str:
        """替换禁用词"""
        for word in self.forbidden:
            text = text.replace(word, "*" * len(word))
        return text
