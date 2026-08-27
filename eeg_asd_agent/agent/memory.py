from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eeg_asd_agent.schemas import (
    AgentTurnDecision,
    AssessmentCaseState,
    Observation,
)


class SQLiteAgentMemory:
    """Local case and turn memory for pause/resume and context recall.

    Full case state stays local in SQLite. Only ``snapshot_for_llm`` is allowed
    into Agent context, and that projection never contains raw EEG, feature
    values, local paths, or internal ToolResult objects.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    decision_json TEXT NOT NULL,
                    observation_json TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(case_id, turn_index)
                );
                """
            )

    def save_state(self, state: AssessmentCaseState) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cases(case_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (state.case_id, state.model_dump_json(), now),
            )

    def load_state(self, case_id: str) -> AssessmentCaseState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            return None
        return AssessmentCaseState.model_validate_json(str(row["state_json"]))

    def record_turn(
        self,
        case_id: str,
        turn_index: int,
        decision: AgentTurnDecision,
        observation: Observation | None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        observation_json = (
            observation.model_dump_json() if observation is not None else None
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO turns(
                    case_id, turn_index, decision_json, observation_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(case_id, turn_index) DO UPDATE SET
                    decision_json=excluded.decision_json,
                    observation_json=excluded.observation_json,
                    created_at=excluded.created_at
                """,
                (
                    case_id,
                    turn_index,
                    decision.model_dump_json(),
                    observation_json,
                    now,
                ),
            )

    def next_turn_index(self, case_id: str) -> int:
        """Return a monotonically increasing index across resumed runs."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(turn_index), 0) AS last_index FROM turns WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        return int(row["last_index"]) + 1

    def turn_count(self, case_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM turns WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        return int(row["count"])

    def snapshot_for_llm(self, case_id: str, *, limit: int = 6) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT turn_index, decision_json, observation_json
                FROM turns
                WHERE case_id = ?
                ORDER BY turn_index DESC
                LIMIT ?
                """,
                (case_id, limit),
            ).fetchall()
        turns: list[dict[str, Any]] = []
        for row in reversed(rows):
            decision = AgentTurnDecision.model_validate_json(str(row["decision_json"]))
            observation = (
                Observation.model_validate_json(str(row["observation_json"]))
                if row["observation_json"] is not None
                else None
            )
            turns.append(
                {
                    "turn_index": int(row["turn_index"]),
                    "tool_name": decision.action.tool_name,
                    "reason": decision.action.reason[:300],
                    "goal_satisfied": decision.goal_satisfied,
                    "observation": (
                        {
                            "observation_id": observation.observation_id,
                            "tool_name": observation.tool_name,
                            "status": observation.status.value,
                            "warnings": observation.warnings,
                            "limitations": observation.limitations,
                        }
                        if observation is not None
                        else None
                    ),
                }
            )
        return {"case_id": case_id, "recent_turns": turns}

    def export_summary(self, case_id: str) -> dict[str, Any]:
        state = self.load_state(case_id)
        return {
            "case_id": case_id,
            "stored": state is not None,
            "execution_status": state.execution_status.value if state else None,
            "step_count": state.step_count if state else 0,
            "plan_revision": state.agent_plan.revision if state and state.agent_plan else None,
            "turn_count": self.turn_count(case_id),
            "recent_turns": self.snapshot_for_llm(case_id)["recent_turns"],
        }
