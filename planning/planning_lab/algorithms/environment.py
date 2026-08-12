import os
import random
import re
import sqlite3

from ..models import EnvironmentFeedback


class Environment:
    """Grounded evaluator that checks the database rather than simulating a score."""

    def __init__(
        self,
        success_threshold: float = 0.6,
        rng: random.Random | None = None,
    ):
        if not 0.0 <= success_threshold <= 1.0:
            raise ValueError("success_threshold must be between zero and one")
        self.success_threshold = success_threshold
        self.rng = rng or random.Random()

    @property
    def db_path(self) -> str:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        return os.path.join(repo_root, "db", "aurelia.db")

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()

    @staticmethod
    def _extract_room_ids(state: str) -> list[str]:
        matches = re.findall(r"\bROOM_\d+\b", state or "", flags=re.IGNORECASE)
        return [match.upper() for match in matches]

    @staticmethod
    def _extract_event_ids(state: str) -> list[str]:
        matches = re.findall(r"\bEVT_\d+\b", state or "", flags=re.IGNORECASE)
        return [match.upper() for match in matches]

    @staticmethod
    def _extract_headcount(state: str) -> int | None:
        patterns = [
            r"requested_headcount\s*[:=]?\s*(\d+)",
            r"headcount\s*[:=]?\s*(\d+)",
            r"(\d+)\s*(?:guests?|people|attendees)\b",
            r"(?:guest|people|attendees)\s*(?:count|total)?\s*[:=]?\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, (state or ""), flags=re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None

    def evaluate(self, state: str) -> EnvironmentFeedback:
        state_text = str(state or "")
        normalized_state = self._normalize_text(state_text)
        details: list[str] = []
        violations = 0

        if not os.path.exists(self.db_path):
            return EnvironmentFeedback(
                success=False,
                score=0.0,
                details=[f"Database not found at {self.db_path}."],
            )

        room_ids = self._extract_room_ids(state_text)
        event_ids = self._extract_event_ids(state_text)
        requested_headcount = self._extract_headcount(state_text)
        ingredient_matches: list[str] = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if room_ids:
                room_id = room_ids[0]
                cursor.execute(
                    "SELECT max_capacity, fire_code_status FROM rooms WHERE room_id = ?",
                    (room_id,),
                )
                room = cursor.fetchone()
                if room is None:
                    violations += 1
                    details.append(f"Room {room_id} is not present in the database.")
                else:
                    max_capacity, fire_status = room
                    if requested_headcount is not None and requested_headcount > max_capacity:
                        violations += 1
                        if fire_status == "STRICT_ENFORCEMENT":
                            details.append(
                                f"Room {room_id} has a strict fire-code maximum of {max_capacity}; "
                                f"requested headcount {requested_headcount} exceeds it."
                            )
                        else:
                            details.append(
                                f"Room {room_id} capacity is {max_capacity}; requested headcount "
                                f"{requested_headcount} exceeds the room limit."
                            )
            elif event_ids and requested_headcount is not None:
                details.append(
                    "No room identifier was provided, so the booking cannot be verified against the DB."
                )
                violations += 1

            cursor.execute("SELECT name, is_nut_free FROM safe_ingredients")
            for ingredient_name, is_nut_free in cursor.fetchall():
                normalized_name = self._normalize_text(ingredient_name)
                if normalized_name and normalized_name in normalized_state:
                    ingredient_matches.append(ingredient_name)
                    if not bool(is_nut_free):
                        violations += 1
                        details.append(
                            f"Ingredient '{ingredient_name}' is not nut-free in the database; "
                            "a severe nut-allergy plan cannot use it."
                        )

        if not room_ids and not event_ids and not ingredient_matches and not requested_headcount:
            violations += 1
            details.append(
                "No database-grounded booking, room, headcount, or ingredient facts were found in the candidate state."
            )

        if not details:
            details.append("Verified against the live database constraints: no violations found.")

        score = 1.0 if violations == 0 else max(0.0, 1.0 - (0.65 * violations))
        success = violations == 0 and score >= self.success_threshold

        return EnvironmentFeedback(
            success=success,
            score=round(score, 4),
            details=details,
        )
