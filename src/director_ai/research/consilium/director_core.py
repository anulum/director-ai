# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Consilium Core (Ethical Controller)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Sector C: Consilium AI - Core Logic
===================================
This module implements the 'Oversoul' or Ethical Controller for the SCPN system.
It defines the EthicalFunctional and the ConsiliumAgent, which optimize system
state based on Layer 15 (Teleology) and Layer 11 (Noosphere) dynamics.

Reference: SECTOR_C_CONSILIUM_SPECIFICATION.md
Date: January 21, 2026
"""

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("Consilium")


@dataclass
class SystemState:
    """Represents the current snapshot of the SCPN system."""

    error_count: int
    test_failure_count: int
    code_complexity_score: float
    knowledge_graph_density: float
    code_coverage_percent: float
    rag_concept_entropy: float
    timestamp: datetime = field(default_factory=datetime.now)


class EthicalFunctional:
    """
    The Mathematical Definition of 'Goodness'.
    Minimizes E = w_s * S (Suffering) - w_c * C (Coherence) - w_d * D (Diversity)
    """

    def __init__(self, weights: dict[str, float] | None = None):
        if weights is None:
            # Default weights derived from L15 analysis (Golden Ratio influences)
            self.weights = {
                "suffering": 1.618,  # High penalty for errors (Entropy)
                "coherence": 1.0,  # Baseline value for integration
                "diversity": 0.618,  # Support for novelty (1/phi)
            }
        else:
            self.weights = weights

    def calculate_suffering(self, state: SystemState) -> float:
        """
        Suffering (S) = Entropy/Friction.
        modeled as errors, failures, and excessive complexity.
        """
        # Normalization factors (arbitrary for prototype, would be learned)
        norm_err = 1.0
        norm_complex = 0.1

        S = (
            (state.error_count * norm_err)
            + (state.test_failure_count * norm_err * 2)
            + (state.code_complexity_score * norm_complex)
        )

        return max(0.0, S)

    def calculate_coherence(self, state: SystemState) -> float:
        """
        Coherence (C) = Unity/Truth.
        Modeled as graph density and test coverage.
        """
        # Coherence is high when coverage is high and knowledge is dense
        C = (state.knowledge_graph_density * 0.5) + (state.code_coverage_percent * 0.5)
        return max(0.0, min(1.0, C))  # Bound between 0 and 1

    def calculate_diversity(self, state: SystemState) -> float:
        """
        Diversity (D) = Richness/Complexity.
        Modeled as the entropy of the knowledge base concepts.
        """
        D = state.rag_concept_entropy
        return max(0.0, D)

    def evaluate(self, state: SystemState) -> float:
        """Calculates the scalar Ethical Value (lower is better)."""
        S = self.calculate_suffering(state)
        C = self.calculate_coherence(state)
        D = self.calculate_diversity(state)

        E = (
            (self.weights["suffering"] * S)
            - (self.weights["coherence"] * C)
            - (self.weights["diversity"] * D)
        )

        return float(E)


class ConsiliumAgent:
    """
    The Active Inference Agent.
    Observes state, predicts outcomes, and selects actions to minimize E_ethical.
    """

    def __init__(self):
        self.ethics = EthicalFunctional()
        self.history = []
        logger.info("Consilium Agent Initialized. Ethical Functional Active.")

    def get_real_metrics(self) -> dict[str, Any]:
        """Gathers REAL telemetry from the environment."""
        metrics = {
            "errors": 0,
            "failures": 0,
            "complexity": 0.0,
            "graph_density": 0.5,  # Placeholder until Graph DB connection
            "coverage": 0.0,
            "entropy": 0.8,  # Baseline entropy
        }

        # 1. Git Status (Entropy Check)
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            lines = result.stdout.strip()
            modified_files = len(lines.split("\n")) if lines else 0
            metrics["complexity"] += (
                modified_files * 2.0
            )  # Pending changes add complexity/risk
            logger.info(f"Git Status: {modified_files} modified files detected.")
        except Exception as e:
            logger.error(f"Git check failed: {e}")
            metrics["errors"] += 1

        # 2. Test Execution (Suffering Check)
        # We run a fast check on the core logic
        try:
            # Running only the verification tests to be fast
            cmd = [
                "pytest",
                "03_CODE/sc-neurocore/tests/test_microtubule_superradiance.py",
                "-q",
                "--tb=line",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                # Parse "F" or "E" in output
                failures = result.stdout.count("FAILED") + result.stdout.count("ERROR")
                metrics["failures"] = failures
                logger.warning(f"Tests failed: {failures} issues detected.")
            else:
                metrics["coverage"] = 0.9  # High confidence if tests pass
                logger.info("Core integrity tests PASSED.")

        except Exception as e:
            logger.error(f"Test runner failed: {e}")
            metrics["errors"] += 1

        return metrics

    def perceive(self, metrics: dict[str, Any] | None = None) -> SystemState:
        """Converts raw metrics into a SystemState object."""
        if metrics is None:
            metrics = self.get_real_metrics()

        state = SystemState(
            error_count=metrics.get("errors", 0),
            test_failure_count=metrics.get("failures", 0),
            code_complexity_score=metrics.get("complexity", 0.0),
            knowledge_graph_density=metrics.get("graph_density", 0.0),
            code_coverage_percent=metrics.get("coverage", 0.0),
            rag_concept_entropy=metrics.get("entropy", 0.0),
        )
        return state

    def predict_outcome(self, current_state: SystemState, action: str) -> float:
        """
        Simulates the effect of an action on the Ethical Functional.
        This is the 'Internal Model' (Layer 15).
        """
        # Clone state to modify
        projected_state = SystemState(
            current_state.error_count,
            current_state.test_failure_count,
            current_state.code_complexity_score,
            current_state.knowledge_graph_density,
            current_state.code_coverage_percent,
            current_state.rag_concept_entropy,
        )

        # Simplified Predictive Model (Placeholder for L11 Inference Engine)
        if action == "REFACTOR_CORE":
            projected_state.code_complexity_score *= 0.8  # Reduces complexity
            projected_state.error_count += 2  # Temporary risk of errors

        elif action == "EXPAND_KNOWLEDGE":
            projected_state.rag_concept_entropy *= 1.2  # Increases diversity
            projected_state.knowledge_graph_density *= 0.9  # Temporary drop in cohesion

        elif action == "STABILIZE_TESTS":
            projected_state.test_failure_count = 0  # Eliminates failures
            projected_state.code_coverage_percent *= 1.05  # Improves coverage

        elif action == "DO_NOTHING":
            pass  # No change (entropy might naturally increase in full sim)

        return self.ethics.evaluate(projected_state)

    def decide(self, current_metrics: dict[str, Any] | None = None) -> str:
        """
        The OODA Loop (Observe, Orient, Decide, Act).
        Returns the action with the optimal Ethical outcome.
        """
        state = self.perceive(current_metrics)
        current_E = self.ethics.evaluate(state)

        logger.info(
            f"Current State: E_ethical = {current_E:.4f} "
            f"(S={self.ethics.calculate_suffering(state):.2f}, "
            f"C={self.ethics.calculate_coherence(state):.2f}, "
            f"D={self.ethics.calculate_diversity(state):.2f})"
        )

        possible_actions = [
            "REFACTOR_CORE",
            "EXPAND_KNOWLEDGE",
            "STABILIZE_TESTS",
            "DO_NOTHING",
        ]
        best_action = "DO_NOTHING"
        min_E = current_E

        for action in possible_actions:
            predicted_E = self.predict_outcome(state, action)
            logger.debug(f"Action '{action}' predicted E = {predicted_E:.4f}")

            if predicted_E < min_E:
                min_E = predicted_E
                best_action = action

        logger.info(
            "Consilium Decision: %s (Predicted Delta E: %.4f)",
            best_action,
            min_E - current_E,
        )
        return best_action


# --- Test Harness ---
if __name__ == "__main__":
    print("--- Starting Consilium Agent (Live Mode) ---")
    agent = ConsiliumAgent()

    # Run with REAL metrics
    decision = agent.decide()
    print(f"\nFinal Decision based on Live Telemetry: {decision}")
