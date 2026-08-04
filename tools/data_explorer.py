"""
数据探索工具
============

快速验证 akshare 接口是否可用,以及返回的字段是什么样的。
方便调试。

用法:
    python -m tools.data_explorer
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import get_tools


def explore():
    t = get_tools()
    
    print("=" * 60)
    print("  akshare 接口探索")
    print("=" * 60)
    
    # 1. 涨停股池
    print("\n[1] 涨停股池 (limit_up_pool)")
    records = t.get_limit_up_pool()
    print(f"  记录数: {len(records)}")
    if records:
        print(f"  示例: {list(records[0].keys())[:8]}")
        print(f"  第一条: {records[0]}")
    
    # 2. 板块异动
    print("\n[2] 板块异动 (board_change)")
    records = t.get_board_change()
    print(f"  记录数: {len(records)}")
    if records:
        print(f"  示例: {list(records[0].keys())[:8]}")
        print(f"  前 3 条:")
        for r in records[:3]:
            print(f"    {r.get('板块名称')}:涨跌幅={r.get('涨跌幅')},异动次数={r.get('板块异动总次数')}")
    
    # 3. 板块资金流
    print("\n[3] 板块资金流 (sector_fund_flow)")
    records = t.get_sector_fund_flow(indicator="今日", sector_type="行业资金流")
    print(f"  记录数: {len(records)}")
    if records:
        print(f"  示例字段: {list(records[0].keys())[:8]}")
        # 按净流入排序看 top
        records_sorted = sorted(records, key=lambda x: float(x.get("主力净流入-净额") or 0), reverse=True)
        for r in records_sorted[:3]:
            try:
                amt = float(r.get("主力净流入-净额") or 0)
                print(f"    {r.get('板块名称')}:{amt/1e8:.2f} 亿元")
            except (ValueError, TypeError):
                pass
    
    # 4. 个股信息
    print("\n[4] 个股信息 (individual_info) - 贵州茅台 600519")
    info = t.get_individual_info("600519")
    print(f"  {info}")
    
    # 5. 业绩预告
    print("\n[5] 业绩预告 (yjyg) - 20240930")
    records = t.get_yjyg("20240930")
    print(f"  记录数: {len(records)}")
    if records:
        print(f"  示例: {records[0]}")
    
    # 6. 全局快讯
    print("\n[6] 全局快讯 (global_news)")
    records = t.get_global_news()
    print(f"  记录数: {len(records)}")
    if records:
        print(f"  示例: {records[0]}")
    
    # 7. 行业板块
    print("\n[7] 行业板块 (industry_boards)")
    records = t.get_industry_boards()
    print(f"  记录数: {len(records)}")
    if records:
        print(f"  示例: {records[0]}")
        print(f"  前 5 个: {[r.get('板块名称', '?') for r in records[:5]]}")


if __name__ == "__main__":
    explore()
