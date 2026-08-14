"""
Centralized Planning Router for Aurelia BEO Assistant (Issue 6.2).
Maps sub-tasks to the optimal planning algorithm (PS, ToT, Reflexion, LATS)
based on task search topology, branching factors, deterministic constraints, 
and failure risk levels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class PlanningAlgorithm(str, Enum):
    """Available planning and search algorithms in the Aurelia planning lab."""
    PLAN_AND_SOLVE = "plan_and_solve"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    REFLEXION = "reflexion"
    LATS = "lats"


class TaskSearchTopology(str, Enum):
    """Search space characteristics of a given sub-task."""
    LINEAR_DETERMINISTIC = "linear_deterministic"       # Single path, arithmetic, direct lookup
    COMBINATORIAL_SELECTION = "combinatorial_selection" # Multiple discrete alternatives / permutations
    ITERATIVE_CORRECTION = "iterative_correction"       # Single path with clear violation feedback loop
    HIGH_STAKES_TREE = "high_stakes_tree"               # Multi-step state space with high failure penalty


@dataclass
class RoutingDecision:
    """Audit record capturing why a sub-task was assigned to an algorithm."""
    task_instruction: str
    algorithm: PlanningAlgorithm
    topology: TaskSearchTopology
    confidence: float
    rationale: str
    detected_signals: list[str] = field(default_factory=list)
    suggested_params: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_instruction": self.task_instruction,
            "algorithm": self.algorithm.value,
            "topology": self.topology.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "detected_signals": self.detected_signals,
            "suggested_params": self.suggested_params,
            "timestamp": self.timestamp,
        }


class PlanningRouter:
    """
    Evaluates sub-task instructions and contextual constraints to dispatch
    the task to the most appropriate planning algorithm with transparent justification.
    """

    # Keyword signal catalogs for heuristic feature extraction
    COMBINATORIAL_SIGNALS = (
        "select a combination", "choose rooms", "distinct rooms", "parallel tracks",
        "breakout", "options", "combinations", "schedule slots", "assign rooms",
        "permutation", "candidate sets"
    )

    HIGH_RISK_SIGNALS = (
        "allergy", "allergic", "severe nut", "vegan", "cross-contamination",
        "life safety", "strict_enforcement", "vip", "financial penalty",
        "non-refundable", "high stakes", "critical safety"
    )

    CORRECTION_SIGNALS = (
        "retry", "exceed", "fire code cap", "room_101", "300", "over capacity",
        "adjust headcount", "feedback loop", "refine booking", "recover from"
    )

    DETERMINISTIC_SIGNALS = (
        "calculate", "sum", "compute", "deposit", "deposit status", "check balance",
        "is under budget", "budget", "verify deposit", "math", "total cost", "difference",
        "view status", "fetch template", "read policy"
    )

    def __init__(self):
        self.decision_log: list[RoutingDecision] = []

    def route_subtask(
        self,
        instruction: str,
        context: Optional[dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Routes an individual sub-task to the best planning algorithm with inline rationale.
        """
        text = instruction.lower()
        ctx = context or {}
        detected_signals: list[str] = []

        # ----------------------------------------------------------------------
        # 1. LANGUAGE AGENT TREE SEARCH (LATS) ROUTING LOGIC
        # ----------------------------------------------------------------------
        # Justification / Rationale:
        # LATS uses Monte Carlo Tree Search (UCT exploration, rollout evaluation,
        # and backpropagation of grounded environment rewards). It incurs the highest
        # token and latency overhead.
        #
        # Mapping Criteria:
        # - High-stakes tasks with life-safety risks (e.g., severe anaphylactic allergies).
        # - Complex multi-constraint optimization where a single mistake violates hard DB rules.
        # - Requires branching exploration AND value backpropagation to guarantee safety.
        # ----------------------------------------------------------------------
        has_high_risk = any(sig in text for sig in self.HIGH_RISK_SIGNALS) or ctx.get("is_high_risk", False)
        requires_deep_search = ("multi-course" in text or "itinerary" in text or "safety" in text)

        if has_high_risk and (requires_deep_search or "menu" in text or "vip" in text):
            matched = [s for s in self.HIGH_RISK_SIGNALS if s in text]
            detected_signals.extend(matched)
            
            decision = RoutingDecision(
                task_instruction=instruction,
                algorithm=PlanningAlgorithm.LATS,
                topology=TaskSearchTopology.HIGH_STAKES_TREE,
                confidence=0.95,
                rationale=(
                    "Sub-task involves mission-critical life-safety or strict VIP constraints "
                    "(e.g., severe allergies/cross-contamination). Routed to LATS for MCTS-driven "
                    "exploration with external environment feedback and value backpropagation."
                ),
                detected_signals=detected_signals,
                suggested_params={"iterations": 2, "n_actions": 2, "exploration_weight": 1.414}
            )
            self._record(decision)
            return decision

        # ----------------------------------------------------------------------
        # 2. TREE OF THOUGHTS (ToT) ROUTING LOGIC
        # ----------------------------------------------------------------------
        # Justification / Rationale:
        # ToT excels at combinatorial constraint satisfaction problems where the agent
        # must explore multiple discrete candidate states (e.g., selecting K distinct
        # rooms out of 150 candidates without overlapping or exceeding capacities).
        #
        # Mapping Criteria:
        # - Discrete combination generation across a database catalog.
        # - Beam search over candidate sets with immediate database pruning.
        # ----------------------------------------------------------------------
        has_combinatorial = any(sig in text for sig in self.COMBINATORIAL_SIGNALS)
        if has_combinatorial:
            matched = [s for s in self.COMBINATORIAL_SIGNALS if s in text]
            detected_signals.extend(matched)

            decision = RoutingDecision(
                task_instruction=instruction,
                algorithm=PlanningAlgorithm.TREE_OF_THOUGHTS,
                topology=TaskSearchTopology.COMBINATORIAL_SELECTION,
                confidence=0.92,
                rationale=(
                    "Sub-task requires combinatorial selection across discrete room/resource options. "
                    "Routed to Tree of Thoughts (ToT) with grounded DB beam search to evaluate and "
                    "prune candidate combinations."
                ),
                detected_signals=detected_signals,
                suggested_params={"depth": 2, "beam_width": 2}
            )
            self._record(decision)
            return decision

        # ----------------------------------------------------------------------
        # 3. REFLEXION ROUTING LOGIC
        # ----------------------------------------------------------------------
        # Justification / Rationale:
        # Reflexion is designed for single-trajectory tasks that are likely to encounter
        # deterministic environment violations (e.g., trying to book 350 guests into
        # a 300-capacity ballroom). It uses verbal episodic memory of past failed trials
        # to self-correct in subsequent attempts.
        #
        # Mapping Criteria:
        # - Single-path action execution requiring environment feedback on failure.
        # - Iterative trial-and-error where the error message provides direct corrective guidance.
        # ----------------------------------------------------------------------
        has_correction = any(sig in text for sig in self.CORRECTION_SIGNALS) or ctx.get("expected_retries", False)
        if has_correction:
            matched = [s for s in self.CORRECTION_SIGNALS if s in text]
            detected_signals.extend(matched)

            decision = RoutingDecision(
                task_instruction=instruction,
                algorithm=PlanningAlgorithm.REFLEXION,
                topology=TaskSearchTopology.ITERATIVE_CORRECTION,
                confidence=0.88,
                rationale=(
                    "Sub-task targets operations prone to capacity/policy violations. "
                    "Routed to Reflexion to leverage bounded episodic verbal memory and "
                    "environment feedback across retries without branching tree overhead."
                ),
                detected_signals=detected_signals,
                suggested_params={"max_trials": 3, "memory_size": 2}
            )
            self._record(decision)
            return decision

        # ----------------------------------------------------------------------
        # 4. PLAN-AND-SOLVE (PS) ROUTING LOGIC (Default & Linear Tasks)
        # ----------------------------------------------------------------------
        # Justification / Rationale:
        # Plan-and-Solve is a zero-branching, single-pass step-by-step reasoning technique.
        # It is optimal for deterministic calculations, arithmetic checks, and simple
        # data syntheses where branching tree search produces zero added value and wastes tokens.
        #
        # Mapping Criteria:
        # - Budget checks, deposit calculations, arithmetic verification.
        # - Linear single-pass lookup + reasoning steps.
        # ----------------------------------------------------------------------
        matched = [s for s in self.DETERMINISTIC_SIGNALS if s in text]
        detected_signals.extend(matched)

        decision = RoutingDecision(
            task_instruction=instruction,
            algorithm=PlanningAlgorithm.PLAN_AND_SOLVE,
            topology=TaskSearchTopology.LINEAR_DETERMINISTIC,
            confidence=0.85,
            rationale=(
                "Sub-task is deterministic, arithmetic, or sequential single-pass reasoning. "
                "Routed to Plan-and-Solve (PS) to minimize latency and token consumption "
                "while enforcing explicit step-by-step calculation separation."
            ),
            detected_signals=detected_signals,
            suggested_params={}
        )
        self._record(decision)
        return decision

    def _record(self, decision: RoutingDecision) -> None:
        """Appends the decision to the internal audit log."""
        self.decision_log.append(decision)

    def get_decision_log(self) -> list[dict[str, Any]]:
        """Returns the full audit trail as serializable dictionaries."""
        return [d.to_dict() for d in self.decision_log]

    def clear_log(self) -> None:
        """Clears the decision log."""
        self.decision_log.clear()