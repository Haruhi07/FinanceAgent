"""
Research Agent - 深度研究(LLM + Tool Calls 驱动)
=====================================================

升级点:
- 之前:硬编码调用固定接口(可能因网络问题失败)
- 现在:让 LLM 自主决定调哪些工具查数据,失败自动尝试其他工具

工作流程:
1. 拿到 topic
2. 先做一轮"轻量预查"(基于行业知识图谱)
3. 把 topic + 行业知识 + 预查结果 喂给 LLM
4. LLM 自主决定调哪些工具深入查询
5. 工具结果 + topic 综合,LLM 生成研究简报
6. 输出 brief

输入: topic (dict)
输出: research_brief (dict)
"""

from __future__ import annotations
import json
import logging
import asyncio
from datetime import datetime
from typing import Optional

from .base import BaseAgent, AgentResult
from tools import get_tools, get_tool_registry, is_llm_available, get_llm
from memory import get_semantic_memory
import config

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    name = "researcher"
    
    def __init__(self):
        super().__init__()
        self.tools = get_tools()
        self.sm = get_semantic_memory()
        self.llm = get_llm()
        self.tool_registry = get_tool_registry()
        self.use_llm = is_llm_available()
        
        if self.use_llm:
            logger.info(f"Researcher 启用 LLM 模式: {config.LLM_MODEL}")
        else:
            logger.warning("Researcher 使用纯工具模式(无 LLM)")
    
    def _run(self, input_data: dict | None) -> AgentResult:
        topics = (input_data or {}).get("topics")
        if not topics:
            topics = self.wm.get("topics", [])
        
        if not topics:
            return AgentResult(
                agent_name=self.name, success=False,
                errors=["no topics to research"],
            )
        
        briefs = []
        errors = []
        for topic in topics:
            try:
                brief = self._research_topic(topic)
                briefs.append(brief)
                self.wm.append("researches", brief)
            except Exception as e:
                errors.append(f"research {topic.get('topic_id', '?')}: {e}")
                self.logger.error(f"research failed: {e}")
        
        return AgentResult(
            agent_name=self.name,
            success=len(briefs) > 0,
            items=briefs,
            metrics={
                "researched": len(briefs),
                "failed": len(errors),
                "llm_enabled": self.use_llm,
            },
            errors=errors,
        )
    
    def _research_topic(self, topic: dict) -> dict:
        """对一个话题做深度研究(2026-08 改造:Plan-and-Execute)"""
        subject = topic["subject"]
        symbols = topic.get("symbols", [])
        industry = topic.get("industry")

        # ---- 1. 行业知识(内部,不会失败) ----
        industry_knowledge = None
        if industry:
            industry_knowledge = self.sm.get_industry(industry)
        industry_context = self._build_industry_context(industry, industry_knowledge)

        # ---- 2. Plan-and-Execute 数据收集 ----
        if self.use_llm:
            # LLM 驱动:先 plan,再 execute
            tool_data, llm_tool_calls, plan = self._collect_data_plan_execute(topic, industry_context)
        else:
            # 纯工具模式(无 LLM,降级)
            tool_data, llm_tool_calls = self._collect_data_with_tools_only(topic, symbols), []
            plan = []

        # ---- 3. 提取关键事实 ----
        key_facts = self._extract_key_facts(topic, tool_data)

        # ---- 4. 生成研究简报 ----
        research_summary = self._build_research_summary(topic, industry_context, tool_data, key_facts)

        # ---- 5. 组装 brief ----
        return {
            "brief_id": f"brief_{int(datetime.now().timestamp())}_{hash(subject) % 10000}",
            "topic_id": topic["topic_id"],
            "subject": subject,
            "industry": industry,
            "industry_knowledge": industry_knowledge,
            "industry_context": industry_context,
            "plan": plan,  # 2026-08 新增:研究计划(可追溯)
            "tool_data": tool_data,
            "llm_tool_calls": llm_tool_calls,
            "key_facts": key_facts,
            "research_summary": research_summary,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }

    # ============= Plan-and-Execute 数据收集(2026-08 新增) =============

    def _collect_data_plan_execute(self, topic: dict, industry_context: str) -> tuple[dict, list, list]:
        """Plan-and-Execute 模式数据收集(2026-08:支持多开子 Agent 并行)

        流程:
        1. **预取真实股票代码**(2026-08-04 修复:避免 LLM 编股票代码)
           先跑一次 get_limit_up_pool,从结果里按行业/名称筛出真实代码
        2. 调 LLM 生成 research_plan(JSON 列表,每项是一个工具调用)
           prompt 里塞入真实代码列表,要求 LLM 只能从中选,严禁编造
        3. **把 plan 拆成独立 batch**: 同一工具的多次调用 → 并行 batch
           例如:plan = [get_financial_report(A), get_financial_report(B), get_financial_report(C)]
           会变成 3 个 sub-agent 并行跑
        4. 按 batch 顺序串行执行(同一 batch 内并行)
        5. 收集所有工具结果(预取的 limit_up 数据合并进去)
        6. 返回 (tool_data_dict, tool_call_history, plan)

        优点:
        - 比 LLM 自由 chat_with_tools 更可控(plan 决定要查什么)
        - 可被外部干预(Orchestrator 改 plan)
        - plan 本身是结构化的,可读、可解释
        - **多开 sub-agent 并行**(2026-08 新增)提高信息收集效率
        - **真实代码注入**(2026-08-04 修复)避免 LLM 瞎填 002931

        Returns:
            (tool_data, llm_tool_calls, plan)
        """
        subject = topic["subject"]
        symbols = topic.get("symbols", [])
        industry = topic.get("industry")
        anomaly_types = topic.get("anomaly_types", [])
        related_names = topic.get("related_symbols", []) or []

        # ---- 0. 预取真实股票代码(避免 LLM 编造) ----
        prefetched_codes = self._prefetch_real_symbols(
            subject=subject,
            industry=industry,
            related_names=related_names,
        )

        # 把预取的涨停股池也塞到 tool_data(供 Writer/Reviewer 用)
        tool_data: dict = {}
        if prefetched_codes:
            # 同步预取完整数据(给下游用,2026-08-04 避开 asyncio.run 嵌套)
            try:
                records = self._call_handler_sync("get_limit_up_pool", {}) or []
                if records:
                    tool_data["limit_up"] = records
                    logger.info(f"预取涨停股池 {len(records)} 条 → tool_data['limit_up']")
            except Exception as e:
                logger.warning(f"预取完整涨停股池数据失败: {e}")

        # ---- 1. Plan 阶段: 调 LLM 出 plan(注入真实代码) ----
        plan = self._make_research_plan(
            topic, industry_context, anomaly_types, prefetched_codes
        )
        if not plan:
            logger.warning(f"研究计划生成失败,降级到 LLM 自主调用")
            tool_data, tc = self._collect_data_with_llm(topic, industry_context)
            return tool_data, tc, []

        logger.info(f"研究计划: {len(plan)} 步 - {[s.get('tool', '?') for s in plan]}")

        # ---- 2. Plan 分 batch: 同一 tool 的连续调用 → 同一 batch(可并行) ----
        batches = self._split_plan_into_batches(plan)
        n_subagents = sum(1 for b in batches if len(b) > 1)
        logger.info(
            f"研究计划分 {len(batches)} 个 batch,"
            f"{n_subagents} 个 batch 用多 sub-agent 并行"
        )

        # ---- 3. Execute 阶段: 按 batch 顺序串行,batch 内并行 ----
        tool_data: dict = {}
        llm_tool_calls: list = []

        for batch_idx, batch in enumerate(batches, 1):
            if len(batch) == 1:
                # 单步 batch,串行执行
                step = batch[0]
                result_data, tc = self._execute_single_step(step, len(llm_tool_calls))
                if tc:
                    llm_tool_calls.append(tc)
                if result_data:
                    self._merge_tool_data(tool_data, step.get("tool", ""), result_data)
            else:
                # 多步 batch,并行执行(多开 sub-agent)
                logger.info(
                    f"  Batch {batch_idx}: 开 {len(batch)} 个 sub-agent 并行执行"
                    f" (tools: {[s.get('tool', '?') for s in batch]})"
                )
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=min(len(batch), 3)) as executor:
                    future_to_step = {
                        executor.submit(
                            self._execute_single_step,
                            step,
                            len(llm_tool_calls) + i,
                        ): (i, step)
                        for i, step in enumerate(batch)
                    }
                    for future in as_completed(future_to_step):
                        i, step = future_to_step[future]
                        try:
                            result_data, tc = future.result()
                            if tc:
                                llm_tool_calls.append(tc)
                            if result_data:
                                self._merge_tool_data(
                                    tool_data, step.get("tool", ""), result_data
                                )
                        except Exception as e:
                            logger.error(
                                f"sub-agent {step.get('tool', '?')} failed: {e}"
                            )

        logger.info(
            f"Researcher(Plan-and-Execute + 并行 sub-agent) 完成: "
            f"{len(llm_tool_calls)} tool calls, keys={list(tool_data.keys())}"
        )
        return tool_data, llm_tool_calls, plan

    def _split_plan_into_batches(self, plan: list[dict]) -> list[list[dict]]:
        """把 plan 拆成可并行的 batch

        规则:连续的同一 tool 调用 → 同一 batch
        例如:
          [A, B, C, D, E] (A=limit_up, B=board_change, C=fin(A), D=fin(B), E=fin(C))
          → [[A], [B], [C, D, E]]  (3 个 batch,最后一个 batch 3 个并行)

        不同工具的连续调用拆开,因为它们的依赖不同。
        """
        if not plan:
            return []
        batches = []
        cur_batch = [plan[0]]
        for step in plan[1:]:
            if step.get("tool") == cur_batch[-1].get("tool"):
                cur_batch.append(step)
            else:
                batches.append(cur_batch)
                cur_batch = [step]
        batches.append(cur_batch)
        return batches

    def _execute_single_step(self, step: dict, tc_idx: int) -> tuple[dict | None, dict | None]:
        """执行 plan 单步,返回 (data, tool_call_record)

        2026-08-04 修复:直接调同步 handler,不要 asyncio.run(嵌套冲突)
        之前用 asyncio.run(self.tool_registry.execute(...)) 在 orchestrator 的
        tool_executor 内部会触发「asyncio.run() cannot be called from a running
        event loop」。改用 _call_handler_sync 直接调 handler 即可。
        """
        tool_name = step.get("tool", "")
        args = step.get("args", {}) or {}
        rationale = step.get("rationale", "")

        try:
            data = self._call_handler_sync(tool_name, args)
            tc = {
                "id": f"plan_{tc_idx:02d}",
                "name": tool_name,
                "arguments_raw": json.dumps(args, ensure_ascii=False),
                "rationale": rationale,
            }
            if data is not None:
                logger.info(f"✅ step: {tool_name}({str(args)[:60]})")
            return data, tc
        except Exception as e:
            logger.error(f"plan step {tool_name} failed: {e}")
            return None, None

    def _call_handler_sync(self, tool_name: str, args: dict):
        """同步调 tool handler(避开 asyncio.run 嵌套问题)

        registry.execute 是 async 的,但 handler 本身是 sync 的(见 agent_tools.py)。
        直接拿 handler 调,跳过 async 包装。
        """
        from tools.agent_tools import get_tool_registry
        reg = get_tool_registry()
        tool = reg.get(tool_name)
        if tool is None:
            logger.warning(f"tool not found: {tool_name}")
            return None
        if tool.handler is None:
            logger.warning(f"tool {tool_name} has no handler")
            return None
        # 直接调 handler(handler 是普通函数,返回 dict/list/str)
        result = tool.handler(**args)
        return result

    def _merge_tool_data(self, tool_data: dict, tool_name: str, new_data) -> None:
        """把单步结果合并到 tool_data(支持同工具多次调用的列表追加)"""
        inv_map = {
            "get_limit_up_pool": "limit_up",
            "get_limit_down_pool": "limit_down",
            "get_board_change": "board_change",
            "get_sector_fund_flow": "sector_fund_flow",
            "get_individual_fund_flow": "individual_fund_flow",
            "get_zh_a_spot": "market_spot",
            "get_individual_info": "individual_info",
            "get_yjyg": "yjyg",
            "get_financial_report": "financial_report",
            "get_research_report": "research_report",
            "get_news": "news",
            "get_global_news": "global_news",
            "get_zh_a_hist": "price_history",
        }
        key = inv_map.get(tool_name, tool_name)
        if key in tool_data:
            existing = tool_data[key]
            if isinstance(existing, list) and isinstance(new_data, list):
                tool_data[key] = existing + new_data
            else:
                tool_data[key] = new_data
        else:
            tool_data[key] = new_data

    def _prefetch_real_symbols(
        self,
        subject: str,
        industry: str | None,
        related_names: list[str],
        max_codes: int = 8,
    ) -> list[dict]:
        """预取真实涨停股代码(避免 LLM 在 plan 里编股票代码)

        2026-08-04 修复:直接调 handler(避开 asyncio.run 嵌套)

        Returns:
            list of {"code": "000593", "name": "德龙汇能", "industry": "燃气Ⅱ"}
        """
        try:
            records = self._call_handler_sync("get_limit_up_pool", {}) or []
        except Exception as e:
            logger.warning(f"预取 limit_up_pool 失败: {e}")
            return []

        if not records:
            return []

        # 过滤:行业匹配 OR 名称相关
        picked: list[dict] = []
        seen_codes: set[str] = set()

        def _try_add(rec: dict) -> None:
            code = str(rec.get("代码") or "").strip()
            name = str(rec.get("名称") or "").strip()
            ind = str(rec.get("所属行业") or "").strip()
            if not code or not name or code in seen_codes:
                return
            # 简单股票代码格式校验(6 位数字)
            if not (len(code) == 6 and code.isdigit()):
                return
            seen_codes.add(code)
            picked.append({"code": code, "name": name, "industry": ind})

        # 1) 行业匹配优先(去掉 "Ⅱ"、"I" 等后缀做粗匹配)
        if industry:
            ind_key = industry.replace("Ⅱ", "").replace("I", "").replace("II", "").strip()
            for rec in records:
                rec_ind = (
                    rec.get("所属行业", "")
                    .replace("Ⅱ", "")
                    .replace("I", "")
                    .replace("II", "")
                    .strip()
                )
                if ind_key and (ind_key in rec_ind or rec_ind in ind_key):
                    _try_add(rec)
                    if len(picked) >= max_codes:
                        break

        # 2) 名称相关(related_symbols 模糊匹配)
        if len(picked) < max_codes and related_names:
            for rec in records:
                name = rec.get("名称", "")
                for rel in related_names:
                    if rel and (rel in name or name in rel):
                        _try_add(rec)
                        break
                if len(picked) >= max_codes:
                    break

        # 3) 主题名相关(行业都拿不到时,拿主题里出现的关键词)
        if len(picked) < max_codes and subject:
            for rec in records:
                name = rec.get("名称", "")
                # 2 字以上行业主题:取最后 2 字作为关键词
                if len(subject) >= 2:
                    kw = subject[-2:]
                    if kw in name:
                        _try_add(rec)
                if len(picked) >= max_codes:
                    break

        # 4) 兜底:取 top N(按涨跌幅排)
        if not picked:
            sorted_recs = sorted(
                records, key=lambda r: r.get("涨跌幅", 0) or 0, reverse=True
            )
            for rec in sorted_recs[:max_codes]:
                _try_add(rec)

        logger.info(
            f"预取真实代码:主题={subject} 行业={industry} 关联={related_names} "
            f"→ {len(picked)} 个:{[c['code']+c['name'] for c in picked[:5]]}"
        )
        return picked

    def _make_research_plan(
        self,
        topic: dict,
        industry_context: str,
        anomaly_types: list,
        real_symbols: list[dict] | None = None,
    ) -> list[dict]:
        """调 LLM 生成研究计划

        Args:
            real_symbols: 预取的涨停股代码(2026-08-04 新增,避免 LLM 编股票代码)

        Returns:
            list of {"tool": str, "args": dict, "rationale": str}
        """
        subject = topic["subject"]
        real_symbols = real_symbols or []

        # 构造"真实代码候选列表"段
        if real_symbols:
            symbols_block = "\n".join(
                f"  - {s['code']} {s['name']} (行业:{s['industry']})"
                for s in real_symbols
            )
        else:
            symbols_block = "  (预取失败,无可用真实代码)"

        # ---- Plan 阶段 prompt ----
        # 关键:严格列出可用工具 + 严格禁止编造,LLM 在零样本下倾向"自由发挥"
        plan_prompt = (
            f"你是一个研究规划师。请为以下话题生成研究计划。\n\n"
            f"## 话题\n"
            f"- 主题:{subject}\n"
            f"- 行业:{topic.get('industry', '?')}\n"
            f"- 异动类型:{anomaly_types}\n"
            f"- 评分:{topic.get('score', 0):.2f}\n\n"
            f"## 已有信息(行业知识图谱)\n{industry_context or '(无)'}\n\n"
            f"## 今日真实涨停股(2026-08-04 注入,只能从这里选 symbol)\n"
            f"{symbols_block}\n\n"
            f"## 严格可用工具列表(只能从以下选,不要编造)\n"
            f"1. get_limit_up_pool - 涨停股池,args: {{}}\n"
            f"2. get_board_change - 板块异动,args: {{}}\n"
            f"3. get_global_news - 全球快讯,args: {{}}\n"
            f"4. get_news - 个股新闻,args: {{\"symbol\": \"6位股票代码(必须从上面真实涨停股里选)\"}}\n"
            f"5. get_yjyg - 业绩预告,args: {{\"date\": \"20250331\"}}\n"
            f"6. get_financial_report - 三大报表,args: {{\"symbol\": \"6位股票代码(必须从上面真实涨停股里选)\", \"report_type\": \"利润表或资产负债表或现金流量表\"}}\n\n"
            f"## 任务\n"
            f"为这个话题生成 **3-5 步** 研究计划,每步指定调哪个工具 + 参数 + 理由。\n"
            f"**硬性要求**:\n"
            f"1. 第一步必须是 get_limit_up_pool 拿涨停股(已有预取数据,可仅作为信息记录,不重复深查)\n"
            f"2. 必须包括 get_board_change 拿板块异动\n"
            f"3. 如果是板块共振类异动,必须包括 1-2 步 get_financial_report,"
            f"**symbol 只能从「今日真实涨停股」列表里选,严禁编造 6 位数字**\n"
            f"4. 如需个股新闻(get_news),symbol 也必须从真实涨停股里选\n\n"
            f"## 输出格式(严格 JSON 数组)\n"
            f"输出形如:[{{\"tool\": \"get_xxx\", \"args\": {{}}, \"rationale\": \"...\"}}]\n"
            f"**只输出 JSON 数组,不要任何其他文字、不要 markdown 包装**"
        )

        try:
            content = self.llm.chat(
                messages=[{"role": "user", "content": plan_prompt}],
                temperature=0.2,  # 降低温度,减少编造
                max_tokens=1500,
                enable_thinking=False,  # 关 thinking 避免 reasoning 抢 token
            )
            # 解析 JSON
            plan = self._parse_plan_json(content)
            if plan and isinstance(plan, list):
                return plan
        except Exception as e:
            logger.error(f"生成研究计划失败: {e}")
        return []

    def _parse_plan_json(self, content: str) -> list[dict]:
        """从 LLM 输出里解析 plan JSON(增强版)"""
        content = content.strip()
        # 尝试 1: 直接 parse 整个 content
        try:
            obj = json.loads(content)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
        # 尝试 2: 找 ```json ... ``` 块(支持多行)
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, list):
                    return obj
            except json.JSONDecodeError:
                pass
        # 尝试 3: 找第一个 [ ... ] (支持多行)
        m = re.search(r"(\[.*?\])", content, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, list):
                    return obj
            except json.JSONDecodeError:
                pass
        # 尝试 4: 找第一个 { ... } 块(可能 LLM 输出对象包了数组)
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "plan" in obj:
                    plan = obj["plan"]
                    if isinstance(plan, list):
                        return plan
            except json.JSONDecodeError:
                pass
        logger.warning(f"无法解析 plan JSON: {content[:300]}")
        return []

    # ============= 数据收集:LLM + Tool Calls(旧版,降级用) =============

    def _collect_data_with_llm(self, topic: dict, industry_context: str) -> tuple[dict, list]:
        """让 LLM 自主决定调哪些工具收集数据(降级路径)

        Returns:
            (tool_data_dict, tool_call_history)
        """
        subject = topic["subject"]
        symbols = topic.get("symbols", [])
        industry = topic.get("industry")
        anomaly_types = topic.get("anomaly_types", [])

        # ---- 准备 prompt ----
        system_prompt = self._build_researcher_system_prompt()
        user_prompt = self._build_researcher_user_prompt(topic, industry_context, symbols)

        # ---- 工具执行器(2026-08-04 修复:同步直接调 handler,避开 asyncio 嵌套) ----
        def tool_executor(name: str, args: dict):
            return self._call_handler_sync(name, args)

        # ---- LLM 自主调工具 ----
        try:
            result = self.llm.chat_with_tools(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=self.tool_registry.to_openai_tools(),
                tool_executor=tool_executor,
                max_rounds=getattr(config, "LLM_MAX_TOOL_ROUNDS", 10),
            )
            tool_calls_history = [
                {
                    "id": tc.get("id", ""),
                    "name": tc["name"],
                    "args": tc["arguments_raw"],
                }
                for tc in result.get("tool_calls", [])
            ]

            # 把工具结果从 messages 里提取出来,组织成 dict
            tool_data = self._extract_tool_data_from_messages(
                result.get("messages", []),
                tool_calls_history,
            )

            logger.info(
                f"Researcher(fallback) 调了 {len(tool_calls_history)} 次工具: "
                f"{[tc['name'] for tc in tool_calls_history]}"
            )

            return tool_data, tool_calls_history
        except Exception as e:
            logger.error(f"LLM-driven data collection failed: {e}, fallback to tool-only")
            return self._collect_data_with_tools_only(topic, symbols), []
    
    def _extract_tool_data_from_messages(self, messages: list, tool_calls: list) -> dict:
        """从 messages 里把工具结果抽出来,组织成 {tool_name: result} 字典"""
        result_map = {
            "get_limit_up_pool": "limit_up",
            "get_limit_down_pool": "limit_down",
            "get_board_change": "board_change",
            "get_sector_fund_flow": "sector_fund_flow",
            "get_individual_fund_flow": "individual_fund_flow",
            "get_zh_a_spot": "market_spot",
            "get_individual_info": "individual_info",
            "get_yjyg": "yjyg",
            "get_financial_report": "financial_report",
            "get_research_report": "research_report",
            "get_news": "news",
            "get_global_news": "global_news",
            "get_zh_a_hist": "price_history",
        }
        
        out: dict = {}
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "{}")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            
            # 找对应的 tool call
            for tc in tool_calls:
                if tc.get("id") == tool_call_id:
                    key = result_map.get(tc["name"], tc["name"])
                    if isinstance(parsed, dict) and "data" in parsed:
                        out[key] = parsed["data"]
                    else:
                        out[key] = parsed
                    break
        
        return out
    
    def _build_researcher_system_prompt(self) -> str:
        """系统提示词"""
        return (
            "你是一位资深的财经研究员,负责对热点话题做深度研究。\n\n"
            "你的工作方式:\n"
            "1. 用户会给你一个热点话题(包括主题、相关个股、异动类型等)\n"
            "2. 你有 13 个金融数据工具可以调用(涨停股池、板块异动、资金流、个股信息、研报、新闻等)\n"
            "3. 根据话题特点,**自主决定调哪些工具**收集数据\n"
            "4. 调 2-5 个工具,不要重复调\n"
            "5. 数据收齐后,简要说明你收集到了什么\n\n"
            "调用工具的指导:\n"
            "- 板块/行业层面 → get_board_change / get_sector_fund_flow / get_global_news\n"
            "- 具体个股 → get_individual_info / get_individual_fund_flow / get_news / get_research_report\n"
            "- 业绩相关 → get_yjyg / get_financial_report\n"
            "- 资金异动 → get_sector_fund_flow / get_individual_fund_flow\n"
            "- 全局态势 → get_global_news / get_board_change\n\n"
            "## ⚠️ 财务数据获取(2026-08 新增)\n"
            "当话题是'板块共振/涨停'类异动时,请执行以下流程:\n"
            "  1. 先调 get_limit_up_pool 拿到涨停股列表\n"
            "  2. 从中挑 1-2 只代表性个股(优先龙一/成交量最大的)\n"
            "  3. **必须**调用 get_financial_report 拿到三表数据(最近 40 期)\n"
            "  4. 这样下游 Writer 就能写'X 公司 2026Q1 营收 50 亿,同比 +15%'这种硬数据\n\n"
            "即使 topic 本身没给 symbol,你也要主动从涨停池里挑一只去查财务。\n"
            "财务数据是文章的'硬支撑',没有财务数据,文章只能写空泛的板块叙事。\n\n"
            "工具返回的数据会直接用于下游写作 Agent,无需你做深度分析。"
            "你的任务只是'用合适的工具把数据拿回来'。"
        )
    
    def _build_researcher_user_prompt(self, topic: dict, industry_ctx: str, symbols: list) -> str:
        """用户提示词"""
        parts = [
            f"请对以下热点话题做深度研究,用工具收集关键数据。",
            f"",
            f"## 话题基本信息",
            f"- 主题:{topic['subject']}",
            f"- 所属行业:{topic.get('industry', '未知')}",
            f"- 评分:{topic.get('score', 0):.2f}",
            f"- 确信度:{topic.get('confidence', 'low')}",
            f"- 异动类型:{', '.join(topic.get('anomaly_types', []))}",
        ]
        if symbols:
            parts.append(f"- 相关个股:{', '.join(symbols[:5])}")
        parts.append(f"")
        parts.append(f"## 话题描述")
        parts.append(topic.get("description", ""))
        parts.append(f"")
        parts.append(f"## 已有信息")
        parts.append(industry_ctx)
        parts.append(f"")
        parts.append(f"## 任务")
        parts.append(f"请调用 2-4 个工具收集数据,完成后简要说明你收集到了什么。")
        return "\n".join(parts)
    
    # ============= 数据收集:无 LLM 模式(降级) =============
    
    def _collect_data_with_tools_only(self, topic: dict, symbols: list) -> dict:
        """无 LLM 模式:硬编码调一些关键工具"""
        data = {}
        try:
            data["board_change"] = self.tools.get_board_change()[:20]
        except Exception:
            pass
        try:
            data["global_news"] = self.tools.get_global_news()[:10]
        except Exception:
            pass
        # 涉及个股则查
        for sym in symbols[:2]:
            try:
                data.setdefault("individual_info", {})[sym] = self.tools.get_individual_info(sym)
            except Exception:
                pass
        return data
    
    # ============= 关键事实提取 =============
    
    def _extract_key_facts(self, topic: dict, tool_data: dict) -> list[str]:
        """从工具数据中提取关键事实"""
        facts = []
        anomaly_types = topic.get("anomaly_types", [])
        subject = topic["subject"]
        
        # 1. 板块/话题层面
        for anom in anomaly_types:
            if anom == "change_with_limitup":
                facts.append("板块异动频繁 + 多只涨停")
            elif anom == "board_change_with_fundflow":
                facts.append("主力资金净流入")
            elif anom == "board_resonance":
                facts.append("板块内多股共振")
            elif anom == "risk_concentration":
                facts.append("板块多股跌停,风险集中")
        
        # 2. 板块异动数据 - 匹配相关板块
        bc = tool_data.get("board_change", [])
        for item in bc[:10] if isinstance(bc, list) else []:
            if isinstance(item, dict):
                board = str(item.get("板块名称", ""))
                change_count = item.get("板块异动总次数", 0)
                pct = item.get("涨跌幅", 0)
                if board and (subject in board or board in subject):
                    facts.append(f"{board} 板块涨跌幅 {pct}%,当日异动 {change_count} 次")
                    break
        
        # 3. 资金流数据 - 匹配相关板块
        sff = tool_data.get("sector_fund_flow", [])
        for item in sff[:10] if isinstance(sff, list) else []:
            if isinstance(item, dict):
                board = str(item.get("板块名称", ""))
                amt = item.get("主力净流入-净额", 0)
                try:
                    amt_float = float(amt or 0)
                    if board and (subject in board or board in subject):
                        direction = "净流入" if amt_float > 0 else "净流出"
                        facts.append(f"{board} 主力{direction} {abs(amt_float)/1e8:.2f} 亿元")
                        break
                except (ValueError, TypeError):
                    pass
        
        # 4. 涨停股池 - 找本板块的涨停股(宽松匹配)
        lu = tool_data.get("limit_up", [])
        matched = []
        subject_lower = subject.lower()
        # 关键词扩展:电池 -> [电池, 锂电, 储能, 新能源]
        subject_keywords = [subject]
        if "电池" in subject:
            subject_keywords.extend(["电池", "锂电", "储能", "新能源"])
        elif "新能源" in subject:
            subject_keywords.extend(["新能源", "电池", "锂电", "光伏", "风电"])
        
        for item in lu[:100] if isinstance(lu, list) else []:
            if isinstance(item, dict):
                ind = str(item.get("所属行业", ""))
                ind_lower = ind.lower()
                if any(kw in ind or ind in kw for kw in subject_keywords):
                    name = item.get("名称", "")
                    if name:
                        matched.append(name)
                if len(matched) >= 5:
                    break
        if matched:
            facts.append(f"板块内涨停股:{', '.join(matched[:5])}")
        elif lu and isinstance(lu, list):
            # 没匹配到本板块,至少说今日涨停总数
            facts.append(f"今日全市场涨停 {len(lu)} 只")
        
        # 5. 个股信息
        ind_info = tool_data.get("individual_info", {})
        if isinstance(ind_info, dict):
            for sym, info in list(ind_info.items())[:2]:
                if isinstance(info, dict) and info:
                    name = info.get("股票简称", sym)
                    industry = info.get("行业", "")
                    if industry:
                        facts.append(f"{name}({sym}) 属于 {industry} 行业")
        
        # 6. 研报观点
        reports = tool_data.get("research_report", [])
        if isinstance(reports, list) and reports:
            for r in reports[:2]:
                if isinstance(r, dict):
                    title = r.get("报告名称", "")
                    rating = r.get("东财评级", "")
                    inst = r.get("机构", "")
                    if title:
                        line = f"研报:{title[:40]}"
                        if rating:
                            line += f"(评级:{rating})"
                        if inst:
                            line += f" [{inst}]"
                        facts.append(line)
                        break
        
        # 7. 个股新闻
        news = tool_data.get("news", [])
        if isinstance(news, list) and news:
            for n in news[:2]:
                if isinstance(n, dict):
                    title = n.get("新闻标题", "")
                    if title:
                        facts.append(f"新闻:{title[:60]}")
                        break
        
        # 8. 全球快讯
        gn = tool_data.get("global_news", [])
        if isinstance(gn, list):
            for item in gn[:3]:
                if isinstance(item, dict):
                    title = str(item.get("标题", ""))
                    if any(kw in title for kw in [subject[:4], "股市", "财经", "政策"]):
                        if title and len(title) > 5:
                            facts.append(f"快讯:{title[:60]}")
                            break
        
        # 9. 业绩预告
        yjyg = tool_data.get("yjyg", [])
        if isinstance(yjyg, list) and yjyg:
            for item in yjyg[:3]:
                if isinstance(item, dict):
                    name = item.get("股票简称", "")
                    change = item.get("业绩变动幅度", 0)
                    try:
                        change_float = float(change or 0)
                        if abs(change_float) > 30:
                            facts.append(f"{name} 业绩预告 {change_float:+.1f}%")
                    except (ValueError, TypeError):
                        pass

        # 10. 财务报告(2026-08 新增,Writer 写硬数据的来源)
        fin_reports = tool_data.get("financial_report", [])
        if isinstance(fin_reports, list) and fin_reports:
            # 按 report_type 分组(可能有多个报表类型)
            by_type = {}
            for fr in fin_reports:
                if not isinstance(fr, dict):
                    continue
                rtype = fr.get("report_type") or fr.get("报表类型") or "未知"
                by_type.setdefault(rtype, []).append(fr)

            for rtype, records in by_type.items():
                # 取最新一条(报告日最大)
                records_sorted = sorted(
                    records,
                    key=lambda x: str(x.get("报告日", "")),
                    reverse=True,
                )
                latest = records_sorted[0]
                report_date = latest.get("报告日", "")
                rev = latest.get("营业总收入") or latest.get("营业收入")
                np_ = latest.get("归属于母公司所有者的净利润") or latest.get("净利润")
                try:
                    if rev:
                        rev_yi = float(rev) / 1e8
                        line = f"{rtype} 最新报告期{report_date}:营收 {rev_yi:.2f} 亿元"
                        if np_:
                            np_yi = float(np_) / 1e8
                            line += f",归母净利 {np_yi:.2f} 亿元"
                        facts.append(line)
                except (ValueError, TypeError):
                    pass

        return facts
    
    # ============= 研究简报 =============
    
    def _build_research_summary(
        self,
        topic: dict,
        industry_ctx: str,
        tool_data: dict,
        key_facts: list,
    ) -> str:
        """生成研究简报文本"""
        parts = [
            f"【研究简报】{topic['subject']} 板块",
            f"",
            f"## 核心信号",
            topic.get("summary", ""),
            f"",
            f"## 行业背景",
            industry_ctx,
            f"",
            f"## 数据要点",
        ]
        if key_facts:
            for f in key_facts[:8]:
                parts.append(f"- {f}")
        else:
            parts.append("- 暂无补充数据")
        
        parts.append(f"")
        parts.append(f"## 数据来源")
        sources = []
        if tool_data.get("board_change"):
            sources.append("板块异动数据")
        if tool_data.get("sector_fund_flow"):
            sources.append("板块资金流")
        if tool_data.get("individual_info"):
            sources.append("个股基础信息")
        if tool_data.get("news"):
            sources.append("个股新闻")
        if tool_data.get("research_report"):
            sources.append("卖方研报")
        if tool_data.get("global_news"):
            sources.append("全球快讯")
        if tool_data.get("yjyg"):
            sources.append("业绩预告")
        parts.append("、".join(sources) if sources else "暂无")
        
        parts.append(f"")
        parts.append(f"## 综合评分")
        parts.append(f"评分 {topic.get('score', 0):.2f},确信度 {topic.get('confidence', 'low')}。")
        
        return "\n".join(parts)
    
    # ============= 行业背景 =============
    
    def _build_industry_context(self, industry: str | None, knowledge: dict | None) -> str:
        if not industry or not knowledge:
            return "暂无行业背景信息。"
        parts = [f"行业:{industry}。"]
        if knowledge.get("上游"):
            parts.append(f"上游:{', '.join(knowledge['上游'][:5])}。")
        if knowledge.get("中游"):
            parts.append(f"中游:{', '.join(knowledge['中游'][:5])}。")
        if knowledge.get("下游"):
            parts.append(f"下游:{', '.join(knowledge['下游'][:5])}。")
        if knowledge.get("key_players"):
            parts.append(f"龙头:{', '.join(knowledge['key_players'][:5])}。")
        if knowledge.get("policy_sensitive"):
            parts.append("该行业受政策影响较大,需关注政策动向。")
        return "".join(parts)
