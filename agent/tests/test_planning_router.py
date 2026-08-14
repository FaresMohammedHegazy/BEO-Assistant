"""
Unit test suite for the Centralized Planning Router.
Validates that sub-tasks are routed to the appropriate algorithm
(PS, ToT, Reflexion, LATS) with verifiable rationales and signal detection.
"""

import os
import sys
import unittest

# Ensure the project root directory is on the Python path for direct execution
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.planning_router import PlanningAlgorithm, PlanningRouter, TaskSearchTopology


class TestPlanningRouter(unittest.TestCase):
    def setUp(self):
        self.router = PlanningRouter()

    def test_route_to_plan_and_solve_for_budget_math(self):
        task = "Calculate the total deposit required for EVT_CONF_01 and verify if it is under budget."
        decision = self.router.route_subtask(task)

        self.assertEqual(decision.algorithm, PlanningAlgorithm.PLAN_AND_SOLVE)
        self.assertEqual(decision.topology, TaskSearchTopology.LINEAR_DETERMINISTIC)
        self.assertIn("deposit", decision.detected_signals)
        self.assertIn("Plan-and-Solve", decision.rationale)

    def test_route_to_tree_of_thoughts_for_breakout_room_selection(self):
        task = "Select a combination of 4 distinct rooms starting at ROOM_102 for parallel breakout tracks."
        decision = self.router.route_subtask(task)

        self.assertEqual(decision.algorithm, PlanningAlgorithm.TREE_OF_THOUGHTS)
        self.assertEqual(decision.topology, TaskSearchTopology.COMBINATORIAL_SELECTION)
        self.assertIn("select a combination", decision.detected_signals)
        self.assertEqual(decision.suggested_params.get("beam_width"), 2)

    def test_route_to_reflexion_for_fire_code_capacity_retry(self):
        task = "Draft a booking summary for EVT_999 into ROOM_101 with 350 attendees, adjusting if exceeding fire code cap."
        decision = self.router.route_subtask(task)

        self.assertEqual(decision.algorithm, PlanningAlgorithm.REFLEXION)
        self.assertEqual(decision.topology, TaskSearchTopology.ITERATIVE_CORRECTION)
        self.assertIn("Reflexion", decision.rationale)
        self.assertEqual(decision.suggested_params.get("max_trials"), 3)

    def test_route_to_lats_for_critical_allergy_vip_menu(self):
        task = "Draft a VIP multi-course custom menu for Eleanor Vance with a severe nut allergy and strict vegan requirements."
        decision = self.router.route_subtask(task)

        self.assertEqual(decision.algorithm, PlanningAlgorithm.LATS)
        self.assertEqual(decision.topology, TaskSearchTopology.HIGH_STAKES_TREE)
        self.assertIn("allergy", decision.detected_signals)
        self.assertIn("LATS", decision.rationale)
        self.assertEqual(decision.suggested_params.get("iterations"), 2)

    def test_decision_log_recording(self):
        self.router.route_subtask("Compute deposit math")
        self.router.route_subtask("Choose rooms for breakout tracks")

        log = self.router.get_decision_log()
        self.assertEqual(len(log), 2)
        self.assertEqual(log[0]["algorithm"], "plan_and_solve")
        self.assertEqual(log[1]["algorithm"], "tree_of_thoughts")


if __name__ == "__main__":
    unittest.main()