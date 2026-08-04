"""
LLM 集成测试
============

测试目标:
1. 客户端初始化(没 key / 有 key)
2. 基本对话调用
3. JSON 模式
4. 错误处理(mock 模式)
5. Writer Agent 用 LLM
6. Reviewer Agent 用 LLM-as-a-Judge

用法:
    python tests/test_llm.py              # 跑全部
    python tests/test_llm.py --check      # 只检查 key 是否可用
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_check_key():
    """检查 API key 状态"""
    from tools.llm_client import is_llm_available, get_llm
    import config
    
    print(f"  USE_LLM (config): {config.USE_LLM}")
    print(f"  LLM_MODEL: {config.LLM_MODEL}")
    print(f"  LLM_BASE_URL: {config.LLM_BASE_URL}")
    print(f"  is_llm_available(): {is_llm_available()}")
    
    llm = get_llm()
    print(f"  LLM 客户端可用: {llm.available}")
    if not llm.available:
        print("  ⚠️  LLM 不可用,可能是没配 DEEPSEEK_API_KEY")
        print("     设置方法: export DEEPSEEK_API_KEY='sk-xxx'")
        return False
    return True


def test_mock_chat():
    """测试 mock 模式对话(没 key 时)"""
    from tools.llm_client import get_llm
    
    llm = get_llm()
    if llm.available:
        print("  跳过(LLM 可用,不走 mock)")
        return
    
    response = llm.chat([
        {"role": "user", "content": "你好"}
    ])
    assert "[MOCK_LLM]" in response
    print(f"  ✓ Mock 模式 OK: {response[:50]}...")


def test_real_chat():
    """测试真实 LLM 调用(需要 key)"""
    from tools.llm_client import get_llm
    
    llm = get_llm()
    if not llm.available:
        print("  跳过(无 API key)")
        return
    
    print("  正在调用 DeepSeek API...")
    response = llm.chat([
        {"role": "system", "content": "你是一个简洁的助手,用一句话回答。"},
        {"role": "user", "content": "1+1=?"},
    ], max_tokens=100)
    
    assert response, "empty response"
    print(f"  ✓ 真实 LLM 调用 OK:")
    print(f"    响应: {response[:200]}")


def test_json_mode():
    """测试 JSON 模式"""
    from tools.llm_client import get_llm
    
    llm = get_llm()
    if not llm.available:
        print("  跳过(无 API key)")
        return
    
    print("  正在调用 JSON 模式...")
    result = llm.chat_json([
        {"role": "system", "content": "你返回 JSON。输出: {\"answer\": \"一句话回答问题\"}"},
        {"role": "user", "content": "深圳在哪里?"},
    ], max_tokens=200)
    
    assert isinstance(result, dict), f"expected dict, got {type(result)}"
    assert "answer" in result or "_raw" in result, f"unexpected keys: {list(result.keys())}"
    print(f"  ✓ JSON 模式 OK: {result}")


def test_writer_with_llm():
    """测试 Writer Agent 用 LLM 写文章"""
    from memory import get_working_memory
    from agents.writer import WriterAgent
    
    # 构造一个 mock brief - 用稳定的 ID 避免重复 append
    wm = get_working_memory()
    # 清掉之前的测试 brief
    wm.set("researches", [])
    
    wm.append("researches", {
        "brief_id": "test_brief_writer_1",
        "subject": "新能源",
        "industry": "新能源",
        "industry_context": "新能源行业是国家战略性新兴产业,涵盖锂电池、光伏、风电、储能等。",
        "industry_knowledge": {"key_players": ["宁德时代", "比亚迪"]},
        "symbol_research": [
            {"symbol": "300750", "info": {"名称": "宁德时代", "行业": "电池"}, "recent_fund_flow": []},
        ],
        "research_reports": [],
        "announcements": [],
        "key_facts": [
            "新能源板块 5 只个股涨停",
            "主力资金净流入 50 亿元",
            "政策面持续支持"
        ],
        "research_summary": "新能源板块今日出现明显异动,资金加速涌入",
        "score": 0.85,
        "confidence": "high",
    })
    
    writer = WriterAgent()
    print(f"  Writer 模式: {'LLM' if writer.use_llm else 'Mock'}")
    
    result = writer.run({})
    assert result.success, f"writer failed: {result.errors}"
    assert len(result.items) == 1, f"expected 1 item, got {len(result.items)}"
    
    article = result.items[0]
    print(f"  ✓ Writer OK:")
    print(f"    标题: {article['title'][:60]}")
    print(f"    字数: {article['word_count']}")
    print(f"    模式: {article['mode']}")
    
    return article


def test_reviewer_with_llm():
    """测试 Reviewer Agent 用 LLM 审核(直接用测试 [5] 已生成的文章,不再跑 Writer)"""
    from agents.reviewer import ReviewerAgent
    from memory import get_working_memory
    
    wm = get_working_memory()
    existing_drafts = wm.get("drafts", [])
    
    if not existing_drafts:
        # 兜底:没有就准备一个简版文章
        from datetime import datetime
        article = {
            "article_id": "test_article_fallback",
            "subject": "新能源",
            "title": "新能源板块异动观察",
            "content": (
                "今日新能源板块出现明显异动。\n\n"
                "**一、板块表现**\n\n"
                "板块多只个股涨停,资金加速涌入。\n\n"
                "**二、行业背景**\n\n"
                "新能源是国家战略性新兴产业。\n\n"
                "**五、后市展望**\n\n"
                "短期需观察持续性。\n\n"
                "**风险提示**\n\n"
                "本文基于公开数据整理,不构成投资建议。"
                "投资有风险,过往业绩不代表未来表现,入市需谨慎。\n"
            ),
            "word_count": 200,
            "mode": "mock",
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        wm.set("drafts", [article])
        existing_drafts = [article]
    
    # 拿最近一篇
    article = existing_drafts[-1]
    print(f"  准备审核: 标题='{article['title'][:40]}', 字数={article['word_count']}")
    
    reviewer = ReviewerAgent()
    print(f"  Reviewer 模式: {'LLM+Rule' if reviewer.use_llm else 'Rule only'}")
    
    result = reviewer.run({})
    assert result.success, f"reviewer failed: {result.errors}"
    assert len(result.items) == 1, f"expected 1 review, got {len(result.items)}"
    
    review = result.items[0]
    print(f"  ✓ Reviewer OK:")
    print(f"    通过: {review['passed']}")
    print(f"    分数: {review['score']}/100")
    print(f"    规则问题: {len(review.get('rule_check', {}).get('issues', []))}")
    if review.get("llm_check"):
        llm_check = review["llm_check"]
        if isinstance(llm_check, dict) and "scores" in llm_check:
            print(f"    LLM 评分: {llm_check.get('scores', {})}")
            print(f"    LLM 平均: {llm_check.get('average_score', 'N/A')}")
        elif isinstance(llm_check, dict):
            print(f"    LLM 检查: {llm_check}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只检查 key 状态")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(" LLM 集成测试")
    print("="*60 + "\n")
    
    print("[1] 检查 API key...")
    has_key = test_check_key()
    print()
    
    if args.check:
        return
    
    print("[2] Mock 模式...")
    test_mock_chat()
    print()
    
    if has_key:
        print("[3] 真实 LLM 调用...")
        test_real_chat()
        print()
        
        print("[4] JSON 模式...")
        test_json_mode()
        print()
        
        print("[5] Writer Agent (LLM)...")
        test_writer_with_llm()
        print()
        
        print("[6] Reviewer Agent (LLM-as-a-Judge)...")
        test_reviewer_with_llm()
        print()
    
    print("="*60)
    print(" 测试完成")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
