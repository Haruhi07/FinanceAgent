"""
基础测试 - 不依赖真实网络
=========================

快速验证 Agent 基类、Memory、信号结构等。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from tools import Signal, SIGNAL_LIMIT_UP, SIGNAL_BOARD_CHANGE
from memory import get_working_memory, get_episodic_memory, get_semantic_memory
from agents.base import BaseAgent, AgentResult
from agents.hotspot_detector import HotspotDetector
from agents.topic_modeler import TopicModeler


class MockScanner(BaseAgent):
    """用 mock 数据跑通流程,用于快速验证"""
    name = "mock_scanner"
    
    def _run(self, input_data):
        signals = [
            Signal(
                signal_type=SIGNAL_LIMIT_UP,
                symbol="002460", name="赣锋锂业", board="能源金属",
                timestamp=datetime.now().isoformat(),
                raw={"所属行业": "能源金属"},
                score_hint=0.8,
            ).to_dict(),
            Signal(
                signal_type=SIGNAL_LIMIT_UP,
                symbol="300750", name="宁德时代", board="电池",
                timestamp=datetime.now().isoformat(),
                raw={"所属行业": "电池"},
                score_hint=0.8,
            ).to_dict(),
            Signal(
                signal_type=SIGNAL_LIMIT_UP,
                symbol="002074", name="国轩高科", board="电池",
                timestamp=datetime.now().isoformat(),
                raw={"所属行业": "电池"},
                score_hint=0.8,
            ).to_dict(),
            Signal(
                signal_type=SIGNAL_BOARD_CHANGE,
                symbol="", name="新能源", board="新能源",
                timestamp=datetime.now().isoformat(),
                raw={"板块异动总次数": 15},
                score_hint=0.6,
            ).to_dict(),
        ]
        for s in signals:
            self.wm.append("signals", s)
        return AgentResult(
            agent_name=self.name, success=True,
            items=signals, metrics={"count": len(signals)},
        )


def test_signal():
    """测试 Signal 数据结构"""
    s = Signal(
        signal_type=SIGNAL_LIMIT_UP,
        symbol="600519", name="贵州茅台", board="白酒",
        timestamp=datetime.now().isoformat(),
    )
    assert s.symbol == "600519"
    assert s.score_hint == 0.0
    d = s.to_dict()
    assert d["symbol"] == "600519"
    print("  ✓ Signal OK")


def test_working_memory():
    """测试 Working Memory"""
    wm = get_working_memory()
    wm.set("test_key", "test_value")
    assert wm.get("test_key") == "test_value"
    wm.append("test_list", {"a": 1})
    wm.append("test_list", {"a": 2})
    assert len(wm.get("test_list")) == 2
    print("  ✓ WorkingMemory OK")


def test_episodic_memory():
    """测试 Episodic Memory"""
    em = get_episodic_memory()
    case = {
        "subject": "新能源",
        "board": "新能源",
        "summary": "新能源板块爆发",
    }
    case_id = em.add_case(case)
    assert case_id
    similar = em.find_similar({"topic": "新能源", "board": "新能源"}, top_k=1)
    assert isinstance(similar, list)
    print(f"  ✓ EpisodicMemory OK (total: {em.stats()['total']})")


def test_semantic_memory():
    """测试 Semantic Memory"""
    sm = get_semantic_memory()
    industries = sm.get_all_industries()
    assert "新能源" in industries
    industry = sm.get_industry("新能源")
    assert industry is not None
    assert "中游" in industry
    found = sm.find_industry_by_keyword("宁德时代")
    assert found == "新能源", f"expected 新能源, got {found}"
    print(f"  ✓ SemanticMemory OK (industries: {len(industries)})")


def test_anomaly_and_hotspot():
    """测试异动检测 + 热点识别 + 话题建模(mock 数据)"""
    # 1. 模拟扫描
    scanner = MockScanner()
    r1 = scanner.run({})
    assert r1.success, "scanner failed"
    assert len(r1.items) == 4
    
    # 2. 异动检测
    from agents.anomaly_detector import AnomalyDetector
    anomaly = AnomalyDetector()
    r2 = anomaly.run({})
    # 至少有板块共振(电池板块 2 只涨停,够不到 3 的阈值,可能没有)
    print(f"  anomalies: {len(r2.items)}")
    
    # 3. 热点识别
    wm = get_working_memory()
    # 手动塞一个 board_resonance anomaly
    wm.append("anomalies", {
        "anomaly_type": "board_resonance",
        "board": "电池",
        "limit_up_count": 2,
        "symbols": ["宁德时代", "国轩高科"],
        "score_hint": 0.7,
        "description": "【板块共振】电池板块 2 只个股涨停",
    })
    wm.append("anomalies", {
        "anomaly_type": "change_with_limitup",
        "board": "新能源",
        "change_count": 15,
        "limit_up_count": 3,
        "symbols": ["宁德时代", "国轩高科", "赣锋锂业"],
        "score_hint": 0.9,
        "description": "【强共振】新能源 板块异动 15 次,涨停 3 只",
    })
    
    hotspot = HotspotDetector()
    r3 = hotspot.run({})
    assert r3.success, "hotspot failed"
    assert len(r3.items) >= 1, f"expected >=1 candidate, got {len(r3.items)}"
    print(f"  hotspots: {len(r3.items)}")
    for c in r3.items:
        print(f"    [{c['confidence']:4s}] {c['subject']} score={c['score']}")
    
    # 4. 话题建模
    topic = TopicModeler()
    r4 = topic.run({})
    assert r4.success
    print(f"  topics: {len(r4.items)}")
    
    print("  ✓ Hotspot + Topic pipeline OK")


def test_evaluator():
    """测试评估器"""
    wm = get_working_memory()
    # 塞一些数据
    wm.append("signals", {"signal_type": "test"})
    wm.append("anomalies", {"anomaly_type": "board_resonance"})
    
    from evaluation import get_evaluator
    ev = get_evaluator()
    result = ev.evaluate_run()
    assert "discovery" in result
    assert "content" in result
    assert "business" in result
    print(f"  ✓ Evaluator OK (summary: {result['summary']})")


def main():
    print("\n" + "="*60)
    print(" 基础测试")
    print("="*60 + "\n")
    
    test_signal()
    test_working_memory()
    test_episodic_memory()
    test_semantic_memory()
    test_anomaly_and_hotspot()
    test_evaluator()
    
    print("\n" + "="*60)
    print(" 全部通过 ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
