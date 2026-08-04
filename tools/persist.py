"""
Detector 结果落盘工具
====================

2026-08 新增:AnomalyDetector / HotspotDetector 把结果存为 JSON 文件
- output/anomalies/YYYYMMDD_HHMMSS.json
- output/candidates/YYYYMMDD_HHMMSS.json

这样下游 Agent (Orchestrator) 可以直接读 JSON,也能留 trace 备查
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)


def save_anomalies(anomalies: list[dict], extra: dict | None = None) -> Path:
    """存 AnomalyDetector 结果到 output/anomalies/

    Args:
        anomalies: anomaly dict 列表
        extra: 额外元数据(比如触发时间、来源 signal 数)

    Returns:
        落盘的 JSON 文件路径
    """
    return _save_json_dir(config.ANOMALIES_DIR, anomalies, extra)


def save_candidates(candidates: list[dict], extra: dict | None = None) -> Path:
    """存 HotspotDetector 结果到 output/candidates/"""
    return _save_json_dir(config.CANDIDATES_DIR, candidates, extra)


def save_brief(brief: dict, extra: dict | None = None) -> Path:
    """存单条 Researcher brief 到 output/briefs/

    2026-08-04 新增:让文章能链回它对应的研究简报。
    文件名:{ts}_{subject}_{brief_id}.json,避免重名覆盖。

    Args:
        brief: Researcher 输出的 research_brief dict
        extra: 额外元数据(可放 brief_id, topic_id 等)

    Returns:
        落盘的 JSON 文件路径
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    subject = (brief.get("subject") or "unknown").replace("/", "_").replace(" ", "_")
    brief_id = brief.get("brief_id") or f"brief_{ts}"
    path = config.BRIEFS_DIR / f"{ts}_{subject}_{brief_id}.json"
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "saved_at": ts,
        **brief,
    }
    if extra:
        payload["_extra"] = extra
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(f"已存 brief: {path.name}")
    return path


def save_briefs(briefs: list[dict], extra: dict | None = None) -> list[Path]:
    """批量存 brief(2026-08-04 新增)"""
    return [save_brief(b, extra) for b in briefs if b]


def load_brief(brief_id: str) -> dict | None:
    """按 brief_id 读 brief(2026-08-04 新增)"""
    if not brief_id:
        return None
    for f in config.BRIEFS_DIR.glob(f"*_{brief_id}.json"):
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"load_brief({brief_id}) failed: {e}")
            return None
    return None


def load_latest_briefs_for_subject(subject: str, limit: int = 5) -> list[dict]:
    """读某 subject 最近 N 条 brief"""
    subject_safe = (subject or "").replace("/", "_").replace(" ", "_")
    files = sorted(
        config.BRIEFS_DIR.glob(f"*_{subject_safe}_*.json"),
        reverse=True,
    )
    briefs = []
    for f in files[:limit]:
        try:
            briefs.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning(f"read {f.name} failed: {e}")
    return briefs


def _save_json_dir(target_dir: Path, items: list[dict], extra: dict | None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = target_dir / f"{ts}.json"
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "count": len(items),
        "items": items,
    }
    if extra:
        payload.update(extra)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(f"已存 {len(items)} 条到 {path.name}")
    return path


def load_latest_anomalies() -> list[dict]:
    """读最近一次 AnomalyDetector 输出(给 Orchestrator 用)"""
    return _load_latest(config.ANOMALIES_DIR)


def load_latest_candidates() -> list[dict]:
    """读最近一次 HotspotDetector 输出"""
    return _load_latest(config.CANDIDATES_DIR)


def _load_latest(target_dir: Path) -> list[dict]:
    if not target_dir.exists():
        return []
    files = sorted(target_dir.glob("*.json"), reverse=True)
    if not files:
        return []
    try:
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        return payload.get("items", [])
    except Exception as e:
        logger.error(f"load_latest({target_dir}) failed: {e}")
        return []
