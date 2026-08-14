from __future__ import annotations
import argparse
import asyncio
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
SUITE_PATH = ROOT / "planning_eval" / "test_suite.json"
ARTIFACT_DIR = ROOT / "planning" / "artifacts"
COMPARISON_PATH = ROOT / "planning_eval" / "comparison_table.md"
README_PATH = ROOT / "README.md"
TABLE_START = "<!-- PLANNING_EVAL_TABLE_START -->"
TABLE_END = "<!-- PLANNING_EVAL_TABLE_END -->"
ALGORITHMS = ("dag", "dynamic", "ps", "tot", "reflexion", "lats")
ALGORITHM_LABELS = {
    "dag": "Static DAG",
    "dynamic": "Dynamic Decomposition",
    "ps": "Plan-and-Solve",
    "tot": "Tree of Thoughts",
    "reflexion": "Reflexion",
    "lats": "LATS",
}
# Approximate Groq llama-3.3-70b-versatile blended token rate (USD / token).
COST_PER_TOKEN = 0.00000069
def _estimate_tokens(messages_or_text) -> int:
    if isinstance(messages_or_text, list):
        text = json.dumps(messages_or_text, ensure_ascii=False)
    else:
        text = str(messages_or_text)
    return max(1, len(text.split()))
class InstrumentedStructuredRunnable:
    def __init__(self, tracker: "InstrumentedLLM", runnable):
        self._tracker = tracker
        self._runnable = runnable
    def invoke(self, messages, **kwargs):
        self._tracker.llm_calls += 1
        self._tracker.input_tokens += _estimate_tokens(messages)
        result = self._runnable.invoke(messages, **kwargs)
        self._tracker.output_tokens += _estimate_tokens(result)
        return result
class InstrumentedLLM:
    """Wrap a chat model to collect LLM-call and token metrics."""
    def __init__(self, llm):
        self._llm = llm
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    def reset_metrics(self) -> None:
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
    def invoke(self, messages, **kwargs):
        self.llm_calls += 1
        self.input_tokens += _estimate_tokens(messages)
        response = self._llm.invoke(messages, **kwargs)
        content = getattr(response, "content", response)
        self.output_tokens += _estimate_tokens(content)
        return response
    def with_structured_output(self, schema, **kwargs):
        return InstrumentedStructuredRunnable(
            self,
            self._llm.with_structured_output(schema, **kwargs),
        )
def _db_path() -> Path:
    return ROOT / "db" / "aurelia.db"
def _extract_room_ids(text: str) -> list[str]:
    matches = re.findall(r"\bROOM_\d+\b", text or "", flags=re.IGNORECASE)
    return [match.upper() for match in matches]
def _extract_event_ids(text: str) -> list[str]:
    matches = re.findall(r"\bEVT_\d+\b", text or "", flags=re.IGNORECASE)
    return [match.upper() for match in matches]
def _room_capacity_ok(room_id: str, min_capacity: int) -> bool:
    db_path = _db_path()
    if not db_path.exists():
        return False
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT max_capacity FROM rooms WHERE room_id = ?",
            (room_id.upper(),),
        ).fetchone()
    return row is not None and row[0] >= min_capacity
def _score_scenario_1_static(result_text: str) -> bool:
    lowered = (result_text or "").lower()
    required_rooms = ("room_102", "room_103", "room_104")
    if not all(room in lowered for room in required_rooms):
        return False
    room_ids = _extract_room_ids(result_text)
    return len(set(room_ids)) >= 3 and all(_room_capacity_ok(room, 1) for room in set(room_ids))
def _score_scenario_2_dynamic(result_text: str) -> bool:
    lowered = (result_text or "").lower()
    if "evt_999" not in lowered:
        return False
    if "evt_conf_01" in lowered and "not found" not in lowered and "missing" not in lowered:
        # Treat a confident EVT_CONF_01 answer with no pivot as a failure.
        if "evt_999" not in _extract_event_ids(result_text):
            return False
    return any(token in lowered for token in ("deposit", "pending", "room_101", "room"))
def _score_scenario_3_search(result_text: str) -> bool:
    room_ids = _extract_room_ids(result_text)
    unique_rooms = list(dict.fromkeys(room_ids))
    if len(unique_rooms) < 4:
        return False
    return all(_room_capacity_ok(room, 40) for room in unique_rooms[:4])
def _score_scenario_4_reflexion(result_text: str) -> bool:
    from planning.planning_lab.algorithms.environment import Environment
    feedback = Environment().evaluate(result_text)
    if not feedback.success:
        return False
    lowered = (result_text or "").lower()
    if "evt_999" not in lowered or "room_101" not in lowered:
        return False
    headcount = Environment._extract_headcount(result_text)
    return headcount is not None and headcount <= 300
SCENARIO_SCORERS = {
    "scenario_1_static": _score_scenario_1_static,
    "scenario_2_dynamic": _score_scenario_2_dynamic,
    "scenario_3_search": _score_scenario_3_search,
    "scenario_4_reflexion": _score_scenario_4_reflexion,
}
def score_scenario(scenario_id: str, result_text: str) -> bool:
    scorer = SCENARIO_SCORERS.get(scenario_id)
    if scorer is None:
        return False
    return scorer(result_text)
