# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Coverage Push Tests (v3.10.1)

"""Targeted tests for coverage gaps: types, exceptions, stats, doc_chunker,
doc_registry edge cases, CLI ingest paths, config profiles."""

from __future__ import annotations

import time

import pytest

# ── Types module (_clamp, CoherenceScore properties) ────────────────


class TestClampFunction:
    def test_clamp_normal(self):
        from director_ai.core.types import _clamp

        assert _clamp(0.5) == 0.5
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0

    def test_clamp_below_range(self):
        from director_ai.core.types import _clamp

        assert _clamp(-0.5) == 0.0

    def test_clamp_above_range(self):
        from director_ai.core.types import _clamp

        assert _clamp(1.5) == 1.0

    def test_clamp_nan(self):
        from director_ai.core.types import _clamp

        assert _clamp(float("nan")) == 0.0

    def test_clamp_positive_inf(self):
        from director_ai.core.types import _clamp

        assert _clamp(float("inf")) == 1.0

    def test_clamp_negative_inf(self):
        from director_ai.core.types import _clamp

        assert _clamp(float("-inf")) == 0.0

    def test_clamp_custom_bounds(self):
        from director_ai.core.types import _clamp

        assert _clamp(5.0, 0.0, 10.0) == 5.0
        assert _clamp(-1.0, 0.0, 10.0) == 0.0
        assert _clamp(11.0, 0.0, 10.0) == 10.0


class TestCoherenceScoreProperties:
    def _make_score(self, **kwargs):
        from director_ai.core.types import CoherenceScore

        defaults = {"score": 0.8, "approved": True, "h_logical": 0.1, "h_factual": 0.2}
        defaults.update(kwargs)
        return CoherenceScore(**defaults)

    def test_claims_empty_without_evidence(self):
        s = self._make_score()
        assert s.claims == []

    def test_attributions_empty_without_evidence(self):
        s = self._make_score()
        assert s.attributions == []

    def test_claim_coverage_none_without_evidence(self):
        s = self._make_score()
        assert s.claim_coverage is None

    def test_unsupported_claims_empty(self):
        s = self._make_score()
        assert s.unsupported_claims == []

    def test_claim_provenance_empty(self):
        s = self._make_score()
        assert s.claim_provenance() == []

    def test_claims_with_evidence(self):
        from director_ai.core.types import (
            ClaimAttribution,
            CoherenceScore,
            EvidenceChunk,
            ScoringEvidence,
        )

        ev = ScoringEvidence(
            chunks=[EvidenceChunk(text="src", distance=0.1)],
            nli_premise="p",
            nli_hypothesis="h",
            nli_score=0.9,
            claims=["claim A", "claim B"],
            attributions=[
                ClaimAttribution(
                    claim="claim A",
                    claim_index=0,
                    source_sentence="src",
                    source_index=0,
                    divergence=0.1,
                    supported=True,
                ),
                ClaimAttribution(
                    claim="claim B",
                    claim_index=1,
                    source_sentence="src2",
                    source_index=1,
                    divergence=0.8,
                    supported=False,
                ),
            ],
            claim_coverage=0.5,
        )
        s = CoherenceScore(
            score=0.6, approved=True, h_logical=0.2, h_factual=0.3, evidence=ev
        )
        assert len(s.claims) == 2
        assert len(s.attributions) == 2
        assert s.claim_coverage == 0.5
        assert len(s.unsupported_claims) == 1
        assert s.unsupported_claims[0].claim == "claim B"
        prov = s.claim_provenance()
        assert len(prov) == 2
        assert prov[0]["supported"] is True
        assert prov[1]["supported"] is False


class TestHaltEvidence:
    def test_creation(self):
        from director_ai.core.types import EvidenceChunk, HaltEvidence

        he = HaltEvidence(
            reason="threshold",
            last_score=0.2,
            evidence_chunks=[EvidenceChunk(text="chunk", distance=0.3)],
            suggested_action="retry",
        )
        assert he.reason == "threshold"
        assert len(he.evidence_chunks) == 1


class TestReviewResult:
    def test_creation(self):
        from director_ai.core.types import ReviewResult

        rr = ReviewResult(
            output="The sky is blue.",
            coherence=None,
            halted=False,
            candidates_evaluated=3,
        )
        assert not rr.halted
        assert rr.candidates_evaluated == 3
        assert rr.fallback_used is False


# ── Exceptions ──────────────────────────────────────────────────────


class TestExceptions:
    def test_base_exception(self):
        from director_ai.core.exceptions import DirectorAIError

        with pytest.raises(DirectorAIError):
            raise DirectorAIError("test")

    def test_coherence_error_is_subclass(self):
        from director_ai.core.exceptions import CoherenceError, DirectorAIError

        assert issubclass(CoherenceError, DirectorAIError)

    def test_kernel_halt_error(self):
        from director_ai.core.exceptions import KernelHaltError

        e = KernelHaltError("emergency")
        assert "emergency" in str(e)

    def test_generator_error(self):
        from director_ai.core.exceptions import GeneratorError

        assert issubclass(GeneratorError, Exception)

    def test_validation_error_is_value_error(self):
        from director_ai.core.exceptions import ValidationError

        assert issubclass(ValidationError, ValueError)
        with pytest.raises(ValueError):
            raise ValidationError("bad input")

    def test_dependency_error(self):
        from director_ai.core.exceptions import DependencyError

        e = DependencyError("torch not installed")
        assert "torch" in str(e)

    def test_physics_error(self):
        from director_ai.core.exceptions import PhysicsError

        assert issubclass(PhysicsError, Exception)

    def test_numerical_error_is_physics_error(self):
        from director_ai.core.exceptions import NumericalError, PhysicsError

        assert issubclass(NumericalError, PhysicsError)

    def test_hallucination_error(self):
        from director_ai.core.exceptions import HallucinationError
        from director_ai.core.types import CoherenceScore

        score = CoherenceScore(score=0.15, approved=False, h_logical=0.8, h_factual=0.9)
        e = HallucinationError("What?", "Wrong answer", score)
        assert e.query == "What?"
        assert e.response == "Wrong answer"
        assert e.score is score
        assert "0.150" in str(e)


# ── StatsStore ──────────────────────────────────────────────────────


class TestStatsStore:
    def test_record_and_summary(self, tmp_path):
        from director_ai.core.stats import StatsStore

        db = tmp_path / "test_stats.db"
        store = StatsStore(db_path=db)

        store.record_review(
            approved=True, score=0.9, h_logical=0.1, h_factual=0.2, latency_ms=5.0
        )
        store.record_review(approved=False, score=0.2, latency_ms=3.0, halted=True)
        store.record_review(approved=True, score=0.85, latency_ms=4.0)

        summary = store.summary()
        assert summary["total"] == 3
        assert summary["approved"] == 2
        assert summary["rejected"] == 1
        assert summary["halted"] == 1
        assert summary["avg_score"] is not None
        assert summary["avg_latency_ms"] is not None
        store.close()

    def test_summary_with_since(self, tmp_path):
        from director_ai.core.stats import StatsStore

        db = tmp_path / "stats2.db"
        store = StatsStore(db_path=db)
        store.record_review(approved=True, score=0.9)

        future = time.time() + 3600
        summary = store.summary(since=future)
        assert summary["total"] == 0
        store.close()

    def test_empty_summary(self, tmp_path):
        from director_ai.core.stats import StatsStore

        db = tmp_path / "empty.db"
        store = StatsStore(db_path=db)
        summary = store.summary()
        assert summary["total"] == 0
        assert summary["avg_score"] is None
        store.close()

    def test_hourly_breakdown(self, tmp_path):
        from director_ai.core.stats import StatsStore

        db = tmp_path / "hourly.db"
        store = StatsStore(db_path=db)
        store.record_review(approved=True, score=0.9)
        store.record_review(approved=False, score=0.2)

        breakdown = store.hourly_breakdown(days=1)
        assert isinstance(breakdown, list)
        if breakdown:
            assert "total" in breakdown[0]
            assert "avg_score" in breakdown[0]
        store.close()

    def test_default_path(self):
        from director_ai.core.stats import StatsStore

        # Just verify it doesn't crash on init (will use default path)
        store = StatsStore()
        store.close()


# ── DocChunker edge cases ───────────────────────────────────────────


class TestDocChunkerEdgeCases:
    def test_force_split_long_word(self):
        from director_ai.core.retrieval.doc_chunker import ChunkConfig, split

        text = "A" * 500
        chunks = split(text, ChunkConfig(chunk_size=100, overlap=10))
        assert len(chunks) >= 5
        for chunk in chunks:
            assert len(chunk) <= 110  # chunk_size + some overlap

    def test_no_overlap(self):
        from director_ai.core.retrieval.doc_chunker import ChunkConfig, split

        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = split(text, ChunkConfig(chunk_size=25, overlap=0))
        assert len(chunks) >= 2

    def test_single_sentence(self):
        from director_ai.core.retrieval.doc_chunker import ChunkConfig, split

        result = split("Short.", ChunkConfig(chunk_size=100))
        assert result == ["Short."]

    def test_empty_text(self):
        from director_ai.core.retrieval.doc_chunker import split

        assert split("") == []

    def test_within_chunk_size(self):
        from director_ai.core.retrieval.doc_chunker import ChunkConfig, split

        text = "Fits in one chunk."
        assert split(text, ChunkConfig(chunk_size=1000)) == [text]


class TestDocChunkerShim:
    def test_shim_resolves(self):
        from director_ai.core.doc_chunker import ChunkConfig, split

        assert callable(split)
        result = split("Test.", ChunkConfig(chunk_size=100))
        assert result == ["Test."]


# ── DocRegistry edge cases ──────────────────────────────────────────


class TestDocRegistryEdgeCases:
    def test_update_nonexistent_raises(self):
        from director_ai.core.retrieval.doc_registry import DocRegistry

        reg = DocRegistry()
        with pytest.raises(KeyError):
            reg.update("nonexistent", ["c0"])

    def test_delete_nonexistent_returns_none(self):
        from director_ai.core.retrieval.doc_registry import DocRegistry

        reg = DocRegistry()
        assert reg.delete("nonexistent") is None

    def test_get_wrong_tenant_returns_none(self):
        from director_ai.core.retrieval.doc_registry import DocRegistry

        reg = DocRegistry()
        reg.register("d1", "f.txt", "tenant_a", ["c0"])
        assert reg.get("d1", "tenant_b") is None

    def test_list_empty_tenant(self):
        from director_ai.core.retrieval.doc_registry import DocRegistry

        reg = DocRegistry()
        assert reg.list_for_tenant("nobody") == []

    def test_doc_record_fields(self):
        from director_ai.core.retrieval.doc_registry import DocRegistry

        reg = DocRegistry()
        rec = reg.register("d1", "source.pdf", "t1", ["c0", "c1", "c2"])
        assert rec.doc_id == "d1"
        assert rec.source == "source.pdf"
        assert rec.tenant_id == "t1"
        assert rec.chunk_count == 3
        assert len(rec.chunk_ids) == 3
        assert rec.created_at > 0
        assert rec.updated_at >= rec.created_at


# ── Config profiles ─────────────────────────────────────────────────


class TestConfigProfiles:
    def test_all_valid_profiles(self):
        from director_ai.core.config import DirectorConfig

        profiles = [
            "medical",
            "finance",
            "legal",
            "creative",
            "customer_support",
            "summarization",
            "fast",
            "thorough",
            "research",
            "lite",
        ]
        for profile in profiles:
            cfg = DirectorConfig.from_profile(profile)
            assert cfg.coherence_threshold > 0
            assert cfg.hard_limit >= 0

    def test_from_env_defaults(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig.from_env()
        assert hasattr(cfg, "coherence_threshold")
        assert hasattr(cfg, "hard_limit")
        assert hasattr(cfg, "use_nli")

    def test_build_scorer(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig.from_profile("fast")
        scorer = cfg.build_scorer()
        assert scorer is not None

    def test_build_store(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig.from_profile("fast")
        store = cfg.build_store()
        assert store is not None


# ── EvidenceChunk ───────────────────────────────────────────────────


class TestEvidenceChunk:
    def test_defaults(self):
        from director_ai.core.types import EvidenceChunk

        ec = EvidenceChunk(text="some text", distance=0.5)
        assert ec.source == ""
        assert ec.distance == 0.5

    def test_with_source(self):
        from director_ai.core.types import EvidenceChunk

        ec = EvidenceChunk(text="text", distance=0.1, source="file.txt")
        assert ec.source == "file.txt"


# ── ScoringEvidence defaults ────────────────────────────────────────


class TestScoringEvidence:
    def test_defaults(self):
        from director_ai.core.types import ScoringEvidence

        ev = ScoringEvidence(
            chunks=[], nli_premise="p", nli_hypothesis="h", nli_score=0.9
        )
        assert ev.chunk_scores is None
        assert ev.claim_coverage is None
        assert ev.claims is None
        assert ev.attributions is None
        assert ev.token_count is None
        assert ev.estimated_cost_usd is None
        assert ev.premise_chunk_count == 1
        assert ev.hypothesis_chunk_count == 1


# ── Verified scorer basic ───────────────────────────────────────────


class TestVerifiedScorerBasic:
    def test_verify_correct_claim(self):
        from director_ai.core.verified_scorer import VerifiedScorer

        vs = VerifiedScorer()
        result = vs.verify(
            "Water boils at 100 degrees.", "Water boils at 100 degrees Celsius."
        )
        assert result.overall_score >= 0
        assert isinstance(result.claims, list)

    def test_verify_hallucinated_claim(self):
        from director_ai.core.verified_scorer import VerifiedScorer

        vs = VerifiedScorer()
        result = vs.verify(
            "The moon is made of green cheese.",
            "Water boils at 100 degrees Celsius.",
        )
        assert result.overall_score >= 0


# ── CLI edge cases ──────────────────────────────────────────────────


class TestCLIEdgeCases:
    def test_command_regex_rejects_special_chars(self):
        from director_ai.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["--weird"])
        # --weird starts with -, not matching ^[a-z]
        assert exc.value.code == 1

    def test_help_command(self, capsys):
        from director_ai.cli import main

        main(["help"])
        out = capsys.readouterr().out
        assert "Commands:" in out

    def test_empty_args_help(self, capsys):
        from director_ai.cli import main

        main([])
        out = capsys.readouterr().out
        assert "Director-Class AI CLI" in out

    def test_version_format(self, capsys):
        from director_ai.cli import main

        main(["version"])
        out = capsys.readouterr().out
        parts = out.strip().split()[-1].split(".")
        assert len(parts) == 3


# ── Lazy imports ────────────────────────────────────────────────────


class TestLazyImports:
    def test_coherence_scorer_importable(self):
        from director_ai import CoherenceScorer

        assert CoherenceScorer is not None

    def test_ground_truth_store_importable(self):
        from director_ai import GroundTruthStore

        assert GroundTruthStore is not None

    def test_vector_store_importable(self):
        from director_ai import VectorGroundTruthStore

        assert VectorGroundTruthStore is not None

    def test_voice_guard_importable(self):
        from director_ai import VoiceGuard

        assert VoiceGuard is not None

    def test_director_config_importable(self):
        from director_ai import DirectorConfig

        assert DirectorConfig is not None

    def test_coherence_score_importable(self):
        from director_ai import CoherenceScore

        assert CoherenceScore is not None

    def test_streaming_kernel_importable(self):
        from director_ai import StreamingKernel

        assert StreamingKernel is not None

    def test_batch_processor_importable(self):
        from director_ai import BatchProcessor

        assert BatchProcessor is not None

    def test_review_result_importable(self):
        from director_ai import ReviewResult

        assert ReviewResult is not None

    def test_halt_evidence_importable(self):
        from director_ai import HaltEvidence

        assert HaltEvidence is not None

    def test_version(self):
        import director_ai

        assert director_ai.__version__
        parts = director_ai.__version__.split(".")
        assert len(parts) == 3

    def test_invalid_attr_raises(self):
        import director_ai

        with pytest.raises(AttributeError):
            _ = director_ai.NonExistentClass
