"""
LLM 客户端 - DeepSeek (OpenAI 兼容协议)
=========================================

设计:
1. 优先用 config 里的 LLM_API_KEY,空则从环境变量读 DEEPSEEK_API_KEY / DEEPSEEK_API
2. 用 openai SDK(DeepSeek 是 OpenAI 兼容)
3. 支持思考模式 (thinking) 和推理强度 (reasoning_effort)
4. 重试 + 超时
5. 没有 key 时降级到 mock(返回固定字符串,方便调试)

用法:
    from tools.llm_client import get_llm, chat, chat_json
    
    llm = get_llm()
    text = chat([{"role": "user", "content": "你好"}])
    
    # JSON 模式
    data = chat_json([{"role": "user", "content": "返回 {\"a\": 1}"}])
"""

from __future__ import annotations

import os
import json
import logging
import re
import time
from typing import Any, Optional, Callable, Awaitable

import config

logger = logging.getLogger(__name__)


def _get_api_key() -> Optional[str]:
    """获取 API key - 优先级: config > env"""
    if config.LLM_API_KEY:
        return config.LLM_API_KEY
    return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API")


def is_llm_available() -> bool:
    """检查 LLM 是否可用"""
    return _get_api_key() is not None


class LLMClient:
    """DeepSeek 客户端(OpenAI 兼容)"""
    
    def __init__(self):
        self.api_key = _get_api_key()
        self.base_url = config.LLM_BASE_URL
        self.model = config.LLM_MODEL
        self.timeout = config.LLM_TIMEOUT
        self.max_retries = config.LLM_MAX_RETRIES
        self.temperature = config.LLM_TEMPERATURE
        self.max_tokens = config.LLM_MAX_TOKENS
        self.thinking = config.LLM_THINKING
        self.reasoning_effort = config.LLM_REASONING_EFFORT
        
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
                logger.info(f"LLM 初始化成功: {self.base_url} / {self.model}")
            except Exception as e:
                logger.error(f"LLM 初始化失败: {e}")
                self._client = None
        else:
            logger.warning("LLM 未配置 API key,使用 mock 模式")
    
    @property
    def available(self) -> bool:
        return self._client is not None
    
    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        allow_mock_fallback: bool = True,
        **kwargs,
    ) -> str:
        """基础对话调用,返回文本内容
        
        Args:
            messages: [{"role": "system/user/assistant", "content": "..."}]
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认最大 token
            allow_mock_fallback: LLM 调用失败时是否降级到 mock(默认 True)
                                 设为 False 时失败会抛异常
            **kwargs: 其他透传参数
        """
        if not self.available:
            if allow_mock_fallback:
                return self._mock_chat(messages)
            raise RuntimeError("LLM 不可用(无 API key 或客户端初始化失败)")
        
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # 构建请求参数
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        
        # DeepSeek 思考模式(用 extra_body 传)
        # 注意: DeepSeek 思考模式默认是开启的! 必须显式传 thinking: disabled 才关闭
        enable_thinking = kwargs.pop("enable_thinking", self.thinking)
        if enable_thinking:
            params["extra_body"] = {
                "thinking": {"type": "enabled"},
                "reasoning_effort": self.reasoning_effort,
            }
        else:
            # 显式关闭(否则 DeepSeek 默认开启)
            params["extra_body"] = {"thinking": {"type": "disabled"}}
        
        # 用户可覆盖其他参数
        params.update(kwargs)
        
        # 重试策略:
        # - 第 1 次:开启 thinking(高质量)
        # - 第 2 次:开启 thinking(再试一次,可能只是网络抖动)
        # - 第 3 次:关闭 thinking(降级保底)
        last_err = None
        for i in range(self.max_retries):
            try:
                # 第 3 次关闭 thinking(节省 token,提高成功率)
                if i >= 2 and enable_thinking:
                    params["extra_body"] = {"thinking": {"type": "disabled"}}
                
                response = self._client.chat.completions.create(**params)
                msg = response.choices[0].message
                content = msg.content or ""
                if not content:
                    reasoning = getattr(msg, "reasoning_content", "") or ""
                    if reasoning:
                        # thinking 模式特有:思考用了 token 但 content 空
                        # 可能原因:1)思考太多;2)网络中断;3)max_tokens 不够
                        raise ValueError(
                            f"thinking 模式返回空 content "
                            f"(reasoning 用了 {len(reasoning)} chars, "
                            f"已分配 max_tokens={max_tokens})"
                        )
                    raise ValueError("empty response content")
                logger.debug(f"LLM 调用成功,长度 {len(content)} 字符")
                # 落 CoT 日志(2026-08 新增)
                if getattr(config, "LLM_REASONING_LOG_ENABLED", True):
                    reasoning = getattr(msg, "reasoning_content", "") or ""
                    if reasoning:
                        logger.debug(f"🧠 chat() reasoning ({len(reasoning)} chars):\n{reasoning[:1500]}{'...(truncated)' if len(reasoning) > 1500 else ''}")
                return content
            except Exception as e:
                last_err = e
                if i < self.max_retries - 1:
                    delay = 2 ** i
                    logger.warning(f"LLM 调用失败(第 {i+1}/{self.max_retries} 次): {e}. {delay}s 后重试")
                    time.sleep(delay)
                # 即使 max_retries 到达,不再 fallback 到 mock
                # (除非 allow_mock_fallback=True)
        
        # 全部失败
        if allow_mock_fallback:
            logger.error(f"LLM 调用最终失败,降级到 mock: {last_err}")
            return self._mock_chat(messages)
        raise RuntimeError(f"LLM 调用失败(已重试 {self.max_retries} 次): {last_err}")
    
    def chat_json(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        **kwargs,
    ) -> dict:
        """JSON 模式调用 - 返回解析后的 dict
        
        自动处理 markdown 代码块包裹
        """
        text = self.chat(messages, temperature=temperature, **kwargs)
        return _parse_json(text)
    
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor: Optional[Callable[[str, dict], Any]] = None,
        max_rounds: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """带工具调用的对话 - 自动多轮循环
        
        Args:
            messages: 消息列表
            tools: OpenAI 格式的工具定义(用 ToolRegistry.to_openai_tools() 生成)
            tool_executor: 工具执行函数 (name, arguments) -> result
                          如果不传,只返回工具调用请求,不执行
            max_rounds: 最大工具调用轮数(默认 config.LLM_MAX_TOOL_ROUNDS)
            max_tokens: 输出 token 上限
            
        Returns:
            {
                "messages": [...],       # 完整的多轮对话历史
                "final_content": str,    # 最终 LLM 输出
                "tool_calls": [...],     # 所有工具调用
                "rounds": int,           # 实际轮数
            }
        """
        if not self.available:
            return {
                "messages": messages,
                "final_content": self._mock_chat(messages),
                "tool_calls": [],
                "rounds": 0,
                "mode": "mock",
            }
        
        max_rounds = max_rounds if max_rounds is not None else config.LLM_MAX_TOOL_ROUNDS
        max_tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS
        
        all_tool_calls = []
        messages = list(messages)  # 拷贝,不要改原 list
        rounds = 0
        
        for round_i in range(max_rounds):
            rounds += 1

            # pop 自定义参数,避免传给 openai client
            enable_thinking_for_tools = kwargs.pop(
                "enable_thinking_for_tools",
                getattr(config, "LLM_THINKING", True),
            )

            params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": False,
            }
            params.update(kwargs)  # 注意:kwargs 已被 pop,只剩合法参数

            # 工具调用循环:默认开 thinking(2026-08 调整,给 ReAct / Plan-and-Execute 留推理空间)
            if enable_thinking_for_tools:
                params["extra_body"] = {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": self.reasoning_effort,
                }

            # 工具参数
            params["tools"] = tools
            # tool_choice 不指定,让模型自己决定

            try:
                response = self._client.chat.completions.create(**params)
            except Exception as e:
                logger.error(f"chat_with_tools 调用失败: {e}")
                break

            msg = response.choices[0].message

            # 1. 落 CoT 日志(2026-08 新增,只落不存)
            if getattr(config, "LLM_REASONING_LOG_ENABLED", True):
                reasoning = getattr(msg, "reasoning_content", "") or ""
                if reasoning:
                    logger.debug(f"🧠 [round {rounds}] reasoning ({len(reasoning)} chars):\n{reasoning[:1500]}{'...(truncated)' if len(reasoning) > 1500 else ''}")

            # 1.5 落 tool_calls 日志(2026-08 新增)
            if getattr(config, "LLM_TOOL_CALL_LOG_ENABLED", True):
                tc_list = getattr(msg, "tool_calls", None) or []
                if tc_list:
                    tc_summary = ", ".join(
                        f"{tc.function.name}({(tc.function.arguments or '')[:60]})"
                        for tc in tc_list
                    )
                    logger.info(f"🔧 [round {rounds}] tool_calls: {tc_summary}")

            # 2. 把模型回复加入 messages
            msg_dict = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if getattr(msg, "tool_calls", None):
                # tool_calls 是 Pydantic 对象,需要转换
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(msg_dict)
            
            # 2. 检查是否要调用工具
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                # 没工具调用,这就是最终回复
                logger.info(f"chat_with_tools 完成,{rounds} 轮,{len(all_tool_calls)} 次工具调用")
                return {
                    "messages": messages,
                    "final_content": msg.content or "",
                    "tool_calls": all_tool_calls,
                    "rounds": rounds,
                    "mode": "llm",
                }
            
            # 3. 记录工具调用
            for tc in tool_calls:
                all_tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments_raw": tc.function.arguments,
                })
            
            # 4. 执行工具
            if tool_executor is None:
                logger.warning("没有 tool_executor,工具调用不会被执行")
                return {
                    "messages": messages,
                    "final_content": msg.content or "",
                    "tool_calls": all_tool_calls,
                    "rounds": rounds,
                    "mode": "llm",
                }
            
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                except json.JSONDecodeError as e:
                    logger.error(f"工具参数解析失败: {tc.function.arguments} -> {e}")
                    args = {}

                # 执行
                result = tool_executor(tc.function.name, args)

                # 落 tool_call 结果日志(2026-08 新增)
                if getattr(config, "LLM_TOOL_CALL_LOG_ENABLED", True):
                    if isinstance(result, dict) and "data" in result:
                        data = result["data"]
                        size = len(data) if isinstance(data, list) else 1
                        logger.info(f"✅ [round {rounds}] {tc.function.name} → {size} items")
                    elif isinstance(result, dict) and "error" in result:
                        logger.warning(f"❌ [round {rounds}] {tc.function.name} → error: {result['error'][:100]}")

                # 工具结果加入 messages - 在 list 层面截断,避免 JSON 字符串被腰斩
                # 这样下游 json.loads 永远不会失败
                if isinstance(result, dict) and isinstance(result.get("data"), list):
                    data_list = result["data"]
                    if len(data_list) > 20:
                        # 大列表(财报/历史K线)只保留前 5 条 + 标记
                        result = {
                            **result,
                            "data": data_list[:5] + [{"_truncated": f"original {len(data_list)} items, kept first 5 (for size limit)"}],
                        }
                content_str = json.dumps(result, ensure_ascii=False, default=str)
                # 二次保险:如果还超长,继续在 list 层面缩
                while len(content_str) > 10000 and isinstance(result, dict) and isinstance(result.get("data"), list):
                    data_list = result["data"]
                    if len(data_list) <= 2:
                        break
                    # 把 data 再截一半(去掉 _truncated 标记)
                    new_data = [x for x in data_list if not (isinstance(x, dict) and "_truncated" in x)][:max(2, len(data_list)//2)]
                    result = {**result, "data": new_data}
                    content_str = json.dumps(result, ensure_ascii=False, default=str)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content_str,
                })
        
        # 达到最大轮数还没收敛
        logger.warning(f"chat_with_tools 达到最大轮数 {max_rounds}")
        return {
            "messages": messages,
            "final_content": messages[-1].get("content", "") if messages else "",
            "tool_calls": all_tool_calls,
            "rounds": rounds,
            "mode": "llm_max_rounds_exceeded",
        }
    
    def _mock_chat(self, messages: list[dict]) -> str:
        """Mock 模式 - 没有 key 时使用,返回固定响应"""
        last_msg = messages[-1]["content"] if messages else ""
        # 返回一个标记,方便调试时知道是 mock
        return (
            f"[MOCK_LLM] 收到 {len(messages)} 条消息,最后一条长度 {len(last_msg)} 字符。"
            f"请配置 DEEPSEEK_API_KEY 环境变量以启用真实 LLM。"
        )


# ---- JSON 解析辅助 ----

def _parse_json(text: str) -> dict:
    """从 LLM 输出中解析 JSON,处理常见格式问题"""
    text = text.strip()
    
    # 1. 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 2. 处理 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    
    # 3. 尝试找第一个 { 或 [
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    
    # 4. 实在不行返回原始文本
    logger.warning(f"无法解析 LLM 输出为 JSON: {text[:200]}")
    return {"_raw": text, "_parse_failed": True}


# ---- 单例 ----

_client_instance: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance


def chat(messages: list[dict], **kwargs) -> str:
    """便捷函数:直接调用对话"""
    return get_llm().chat(messages, **kwargs)


def chat_json(messages: list[dict], **kwargs) -> dict:
    """便捷函数:JSON 模式对话"""
    return get_llm().chat_json(messages, **kwargs)
