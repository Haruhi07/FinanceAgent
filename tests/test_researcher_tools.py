"""
Researcher + Tool Calls 集成测试
===================================

测试:
1. 无 LLM 模式(纯工具)
2. LLM 模式(自主调工具)
3. 端到端:Scanner → Anomaly → Hotspot → Topic → Researcher
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_researcher_no_llm():
    """无 LLM 模式(降级)"""
    from agents.researcher import ResearchAgent
    from memory import get_working_memory
    
    # 准备一个 topic
    wm = get_working_memory()
    wm.set("topics", [{
        "topic_id": "test_topic_1",
        "subject": "新能源",
        "industry": "新能源",
        "anomaly_types": ["board_resonance"],
        "symbols": ["300750", "002074"],
        "score": 0.8,
        "confidence": "high",
        "description": "新能源板块 5 只涨停,资金加速涌入",
        "summary": "新能源板块今日出现明显异动",
    }])
    
    # 强制用纯工具模式
    agent = ResearchAgent()
    agent.use_llm = False  # 强制 mock
    
    result = agent.run({})
    assert result.success, f"failed: {result.errors}"
    assert len(result.items) == 1
    
    brief = result.items[0]
    print(f"  ✓ 无 LLM 模式 OK")
    print(f"    subject: {brief['subject']}")
    print(f"    key_facts 数: {len(brief['key_facts'])}")
    print(f"    tool_data 字段: {list(brief['tool_data'].keys())}")
    return brief


def test_researcher_with_llm():
    """LLM 模式:让 LLM 自主决定调哪些工具"""
    from agents.researcher import ResearchAgent
    from memory import get_working_memory
    from tools import is_llm_available
    
    if not is_llm_available():
        print("  跳过(无 API key)")
        return None
    
    wm = get_working_memory()
    wm.set("topics", [{
        "topic_id": "test_topic_llm_1",
        "subject": "电池",
        "industry": "新能源",
        "anomaly_types": ["board_resonance", "board_change_with_fundflow"],
        "symbols": ["300750", "002074"],  # 宁德时代、国轩高科
        "score": 0.85,
        "confidence": "high",
        "description": "电池板块 3 只涨停,主力资金净流入 50 亿",
        "summary": "电池板块今日出现明显异动,资金加速涌入",
    }])
    
    agent = ResearchAgent()
    print(f"  LLM 模式: {agent.use_llm}")
    
    result = agent.run({})
    assert result.success, f"failed: {result.errors}"
    assert len(result.items) == 1
    
    brief = result.items[0]
    print(f"  ✓ LLM 模式 OK")
    print(f"    subject: {brief['subject']}")
    print(f"    LLM 工具调用数: {len(brief.get('llm_tool_calls', []))}")
    for tc in brief.get('llm_tool_calls', []):
        print(f"      - {tc['name']}({tc.get('args', '')[:60]})")
    print(f"    tool_data 字段: {list(brief['tool_data'].keys())}")
    print(f"    key_facts 数: {len(brief['key_facts'])}")
    print(f"    研究简报前 200 字:")
    print(f"    {brief['research_summary'][:200]}")
    
    assert len(brief.get('llm_tool_calls', [])) >= 1, "LLM 至少应该调 1 个工具"
    return brief


def test_end_to_end_pipeline():
    """端到端:Scanner → Anomaly → Hotspot → Topic → Researcher(LLM 模式)"""
    from agents.scanner import ScannerAgent
    from agents.anomaly_detector import AnomalyDetector
    from agents.hotspot_detector import HotspotDetector
    from agents.topic_modeler import TopicModeler
    from agents.researcher import ResearchAgent
    from memory import get_working_memory
    from tools import is_llm_available
    
    if not is_llm_available():
        print("  跳过(无 API key)")
        return
    
    print("  重置 Working Memory...")
    wm = get_working_memory()
    wm.set("signals", [])
    wm.set("anomalies", [])
    wm.set("hotspot_candidates", [])
    wm.set("topics", [])
    wm.set("researches", [])
    
    print("  1. Scanner...")
    r = ScannerAgent().run({})
    assert r.success
    print(f"    signals: {r.metrics.get('total_signals', 0)}")
    
    print("  2. Anomaly Detector...")
    r = AnomalyDetector().run({})
    assert r.success
    print(f"    anomalies: {r.metrics.get('anomaly_count', 0)}")
    
    print("  3. Hotspot Detector...")
    r = HotspotDetector().run({})
    assert r.success
    print(f"    candidates: {r.metrics.get('passed_threshold', 0)}")
    
    print("  4. Topic Modeler...")
    candidates = wm.get("hotspot_candidates", [])
    actionable = [c for c in candidates if c["score"] >= 0.55]
    if not actionable:
        print("    无 actionable 候选,跳过 Topic/Researcher")
        return
    r = TopicModeler().run({"candidates": actionable})
    assert r.success
    print(f"    topics: {r.metrics.get('topic_count', 0)}")
    
    print("  5. Researcher (LLM + Tool Calls)...")
    r = ResearchAgent().run({})
    if not r.success:
        print(f"    Researcher 失败:{r.errors}")
        return
    print(f"    researched: {r.metrics.get('researched', 0)}")
    
    # 检查产物
    briefs = wm.get("researches", [])
    for b in briefs:
        print(f"    - {b['subject']}: {len(b.get('llm_tool_calls', []))} 个工具调用, {len(b['key_facts'])} 个事实")
    
    print(f"  ✓ 端到端 OK")


def main():
    print("\n" + "="*60)
    print(" Researcher + Tool Calls 集成测试")
    print("="*60 + "\n")
    
    print("[1] 无 LLM 模式(纯工具)...")
    test_researcher_no_llm()
    print()
    
    print("[2] LLM 模式(自主调工具)...")
    test_researcher_with_llm()
    print()
    
    print("[3] 端到端流水线...")
    test_end_to_end_pipeline()
    print()
    
    print("="*60)
    print(" 全部完成")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
