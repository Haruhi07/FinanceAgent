"""
Tool Calls + 长输出测试
========================

测试:
1. 工具注册表 + 工具 schema
2. 单个工具直接调用
3. LLM 自动决定调用工具(chat_with_tools)
4. 多轮工具调用循环
5. 长输出写作
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_tool_registry():
    """测试工具注册表"""
    from tools import get_tool_registry
    
    reg = get_tool_registry()
    tools = reg.list_tools()
    print(f"  注册工具数: {len(tools)}")
    print(f"  工具列表: {tools[:5]}...")
    
    # 校验 schema
    assert len(tools) >= 10, f"expected >= 10 tools, got {len(tools)}"
    
    # 转为 OpenAI 格式
    openai_tools = reg.to_openai_tools()
    assert all("type" in t and "function" in t for t in openai_tools)
    print(f"  ✓ ToolRegistry OK ({len(openai_tools)} 个工具 schema 有效)")


def test_tool_direct_execution():
    """直接调工具(不走 LLM)"""
    from tools import get_tool_registry
    import asyncio
    
    reg = get_tool_registry()
    
    # 测试 get_board_change(无参数)
    result = asyncio.run(reg.execute("get_board_change", {}))
    assert result.get("success"), f"tool failed: {result}"
    assert "data" in result
    data = result["data"]
    assert isinstance(data, list)
    print(f"  get_board_change: 返回 {len(data)} 条")
    
    # 测试 get_limit_up_pool
    result = asyncio.run(reg.execute("get_limit_up_pool", {"date": ""}))
    assert result.get("success")
    data = result["data"]
    print(f"  get_limit_up_pool: 返回 {len(data)} 条")
    
    print(f"  ✓ 工具直接调用 OK")


def test_llm_with_tools():
    """测试 LLM 自动调工具"""
    from tools import get_llm, get_tool_registry, is_llm_available
    
    if not is_llm_available():
        print("  跳过(无 API key)")
        return
    
    reg = get_tool_registry()
    tools = reg.to_openai_tools()
    llm = get_llm()
    
    print("  场景 1: 简单问题(应该自动调工具)")
    messages = [
        {"role": "system", "content": "你是财经助手。用户问什么就调用合适的工具查询,不要凭空回答。"},
        {"role": "user", "content": "今天有哪些股票涨停了?给我列前 5 个。"},
    ]
    
    # 工具执行器:从注册表找
    import asyncio
    def executor(name, args):
        return asyncio.run(reg.execute(name, args))
    
    result = llm.chat_with_tools(
        messages,
        tools=tools,
        tool_executor=executor,
        max_rounds=3,
    )
    
    print(f"    轮数: {result['rounds']}")
    print(f"    工具调用数: {len(result['tool_calls'])}")
    for tc in result['tool_calls'][:3]:
        print(f"      - {tc['name']}({tc['arguments_raw'][:80]})")
    print(f"    最终输出前 200 字:")
    print(f"    {result['final_content'][:200]}")
    
    assert result['mode'] == 'llm' or 'exceeded' in result['mode']
    assert len(result['tool_calls']) >= 1, "LLM 应该至少调用 1 个工具"
    print(f"  ✓ LLM 自动调工具 OK")


def test_multi_round_tools():
    """测试多轮工具调用"""
    from tools import get_llm, get_tool_registry, is_llm_available
    
    if not is_llm_available():
        print("  跳过(无 API key)")
        return
    
    reg = get_tool_registry()
    tools = reg.to_openai_tools()
    llm = get_llm()
    
    print("  场景: 复合问题(需要调多个工具)")
    messages = [
        {"role": "system", "content": "你是财经助手。可以调用工具查询数据,综合多个工具结果回答。"},
        {"role": "user", "content": "今天涨停的股票里,有没有属于'电池'板块的?给我列出来。"},
    ]
    
    import asyncio
    def executor(name, args):
        return asyncio.run(reg.execute(name, args))
    
    result = llm.chat_with_tools(
        messages,
        tools=tools,
        tool_executor=executor,
        max_rounds=5,
    )
    
    print(f"    轮数: {result['rounds']}")
    print(f"    工具调用: {[(tc['name']) for tc in result['tool_calls']]}")
    print(f"    最终输出:")
    print(f"    {result['final_content'][:300]}")
    
    # 多轮情况下,工具调用应该 >= 1
    assert len(result['tool_calls']) >= 1
    print(f"  ✓ 多轮工具调用 OK")


def test_long_output():
    """测试长输出(Writer 用 16K tokens)"""
    from tools import get_llm, is_llm_available
    import config
    
    if not is_llm_available():
        print("  跳过(无 API key)")
        return
    
    llm = get_llm()
    
    print(f"  max_tokens = {config.LLM_WRITER_MAX_TOKENS}")
    print("  场景: 长文写作(目标 3000 字)")
    messages = [
        {"role": "system", "content": "你是财经写作专家。"},
        {"role": "user", "content": "请写一篇 3000 字左右的深度分析文章,主题:中国新能源汽车产业链投资机会。结构:导语/产业链全景/上游材料/中游电池/下游整车/竞争格局/政策环境/投资建议/风险提示。每段都要有具体数据,不要空话。"},
    ]
    
    content = llm.chat(
        messages,
        max_tokens=config.LLM_WRITER_MAX_TOKENS,
    )
    print(f"    输出长度: {len(content)} 字符 (约 {len(content)//2} 字)")
    print(f"    前 200 字:")
    print(f"    {content[:200]}")
    print(f"    ...")
    print(f"    末 200 字:")
    print(f"    {content[-200:]}")
    
    assert len(content) > 1500, f"expected > 1500 chars, got {len(content)}"
    print(f"  ✓ 长输出 OK")


def main():
    print("\n" + "="*60)
    print(" 工具调用 + 长输出测试")
    print("="*60 + "\n")
    
    print("[1] 工具注册表...")
    test_tool_registry()
    print()
    
    print("[2] 工具直接执行...")
    test_tool_direct_execution()
    print()
    
    print("[3] LLM + 单工具调用...")
    test_llm_with_tools()
    print()
    
    print("[4] LLM + 多轮工具调用...")
    test_multi_round_tools()
    print()
    
    print("[5] 长输出...")
    test_long_output()
    print()
    
    print("="*60)
    print(" 全部完成")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
