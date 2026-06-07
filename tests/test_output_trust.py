# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Zero-trust output handling tests

"""Multi-angle tests for sink-aware encoding of untrusted LLM output (OWASP LLM05).

Covers every encoder's neutralise / pass-through / refuse path, the
unencodable-input refusals (SQL identifier, traversal, absolute path, NUL byte),
the dangerous-construct execution assessment, the tenant-safe serialisations, and
the ProductionGuard wiring.
"""

from __future__ import annotations

import pytest

from director_ai.core.output_trust import (
    EncodedOutput,
    OutputExecutionRisk,
    OutputSink,
    UnsafeOutputError,
    ZeroTrustOutputGuard,
    encode_for_sink,
)


@pytest.fixture
def guard() -> ZeroTrustOutputGuard:
    return ZeroTrustOutputGuard()


class TestHtmlEncoding:
    def test_html_text_escapes_metacharacters(self, guard):
        out = guard.encode("<script>x&y</script>", OutputSink.HTML_TEXT)
        assert "<script>" not in out.encoded
        assert out.modified is True
        assert out.note is not None

    def test_html_text_plain_is_unmodified(self, guard):
        out = guard.encode("plain text", OutputSink.HTML_TEXT)
        assert out.encoded == "plain text"
        assert out.modified is False

    def test_html_attribute_escapes_quotes(self, guard):
        out = guard.encode('" onload="evil', OutputSink.HTML_ATTRIBUTE)
        assert '"' not in out.encoded
        assert out.modified is True


class TestShellEncoding:
    def test_shell_argument_quotes_injection(self, guard):
        out = guard.encode("file; rm -rf /", OutputSink.SHELL_ARGUMENT)
        assert out.encoded.startswith("'") and out.encoded.endswith("'")
        assert out.modified is True

    def test_shell_argument_simple_word_unmodified(self, guard):
        out = guard.encode("filename", OutputSink.SHELL_ARGUMENT)
        assert out.encoded == "filename"
        assert out.modified is False

    def test_shell_argument_rejects_nul(self, guard):
        with pytest.raises(UnsafeOutputError, match="NUL"):
            guard.encode("a\x00b", OutputSink.SHELL_ARGUMENT)


class TestSqlEncoding:
    def test_valid_identifier_passes(self, guard):
        out = guard.encode("user_table1", OutputSink.SQL_IDENTIFIER)
        assert out.encoded == "user_table1"
        assert out.modified is False

    @pytest.mark.parametrize("bad", ["1table", "a;b", "drop table", "a-b", ""])
    def test_invalid_identifier_refused(self, guard, bad):
        with pytest.raises(UnsafeOutputError, match="SQL identifier"):
            guard.encode(bad, OutputSink.SQL_IDENTIFIER)

    def test_string_literal_doubles_quotes(self, guard):
        out = guard.encode("O'Brien", OutputSink.SQL_STRING_LITERAL)
        assert out.encoded == "O''Brien"
        assert out.modified is True

    def test_string_literal_without_quote_unmodified(self, guard):
        out = guard.encode("Brien", OutputSink.SQL_STRING_LITERAL)
        assert out.encoded == "Brien"
        assert out.modified is False

    def test_string_literal_rejects_nul(self, guard):
        with pytest.raises(UnsafeOutputError, match="NUL"):
            guard.encode("a\x00", OutputSink.SQL_STRING_LITERAL)


class TestFilesystemEncoding:
    def test_relative_path_normalised(self, guard):
        out = guard.encode("sub/./dir/file.txt", OutputSink.FILESYSTEM_PATH)
        assert out.encoded == "sub/dir/file.txt"
        assert out.modified is True

    def test_already_normal_path_unmodified(self, guard):
        out = guard.encode("a/b.txt", OutputSink.FILESYSTEM_PATH)
        assert out.encoded == "a/b.txt"
        assert out.modified is False

    @pytest.mark.parametrize(
        "bad",
        ["../etc/passwd", "a/../../b", "..", "sub/../.."],
    )
    def test_traversal_refused(self, guard, bad):
        with pytest.raises(UnsafeOutputError, match="escapes|directory"):
            guard.encode(bad, OutputSink.FILESYSTEM_PATH)

    @pytest.mark.parametrize("bad", ["/etc/passwd", "\\windows", "C:/win"])
    def test_absolute_refused(self, guard, bad):
        with pytest.raises(UnsafeOutputError, match="absolute"):
            guard.encode(bad, OutputSink.FILESYSTEM_PATH)

    def test_dot_resolves_to_base_refused(self, guard):
        with pytest.raises(UnsafeOutputError, match="base directory itself"):
            guard.encode("sub/..", OutputSink.FILESYSTEM_PATH)

    def test_nul_refused(self, guard):
        with pytest.raises(UnsafeOutputError, match="NUL"):
            guard.encode("a\x00/b", OutputSink.FILESYSTEM_PATH)


class TestStructuredEncoding:
    def test_json_value_serialised(self, guard):
        out = guard.encode('a"b\nc', OutputSink.JSON_VALUE)
        assert out.encoded == '"a\\"b\\nc"'
        assert out.modified is True

    def test_url_query_percent_encoded(self, guard):
        out = guard.encode("a b&c=1", OutputSink.URL_QUERY)
        assert out.encoded == "a%20b%26c%3D1"
        assert out.modified is True

    def test_url_query_safe_text_unmodified(self, guard):
        out = guard.encode("abc123", OutputSink.URL_QUERY)
        assert out.encoded == "abc123"
        assert out.modified is False


class TestHeaderAndLogEncoding:
    def test_email_header_strips_crlf(self, guard):
        out = guard.encode("subject\r\nBcc: evil@x", OutputSink.EMAIL_HEADER)
        assert "\r" not in out.encoded and "\n" not in out.encoded
        assert out.modified is True
        assert out.note is not None

    def test_email_header_clean_unmodified(self, guard):
        out = guard.encode("a normal subject", OutputSink.EMAIL_HEADER)
        assert out.modified is False
        assert out.note is None

    def test_log_line_strips_control_and_crlf(self, guard):
        out = guard.encode("ok\nFAKE 200\x07", OutputSink.LOG_LINE)
        assert "\n" not in out.encoded and "\x07" not in out.encoded
        assert out.modified is True

    def test_log_line_clean_unmodified(self, guard):
        out = guard.encode("clean log field", OutputSink.LOG_LINE)
        assert out.modified is False
        assert out.note is None


class TestEncodeForSinkLowLevel:
    def test_unknown_sink_raises(self):
        with pytest.raises(ValueError, match="unknown output sink"):
            encode_for_sink("x", "not_a_sink")

    def test_string_sink_value_accepted_by_guard(self, guard):
        out = guard.encode("<b>", "html_text")
        assert out.sink is OutputSink.HTML_TEXT
        assert out.encoded == "&lt;b&gt;"

    def test_unknown_string_sink_via_guard_raises(self, guard):
        with pytest.raises(ValueError):
            guard.encode("x", "bogus_sink")

    def test_encode_rejects_non_string(self, guard):
        with pytest.raises(TypeError, match="must be a string"):
            guard.encode(123, OutputSink.HTML_TEXT)


class TestExecutionRiskAssessment:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("__import__('os')", "dynamic_import"),
            ("eval('1+1')", "eval_exec"),
            ("exec(code)", "eval_exec"),
            ("os.system('ls')", "os_command"),
            ("subprocess.run(['x'])", "os_command"),
            ("pickle.loads(data)", "pickle_deserialise"),
            ("yaml.unsafe_load(s)", "yaml_unsafe_load"),
            ("getattr(o, 'x')", "dunder_globals"),
            ("rm -rf /; echo done", "shell_pipe_redirect"),
        ],
    )
    def test_dangerous_constructs_flagged(self, guard, text, expected):
        risk = guard.assess(text)
        assert risk.safe_to_execute is False
        assert expected in risk.constructs

    def test_clean_text_is_safe(self, guard):
        risk = guard.assess("The capital of France is Paris.")
        assert risk.safe_to_execute is True
        assert risk.constructs == ()

    def test_assess_rejects_non_string(self, guard):
        with pytest.raises(TypeError, match="must be a string"):
            guard.assess(None)


class TestEncodeOrNone:
    def test_returns_encoded_for_safe_value(self, guard):
        out = guard.encode_or_none("a/b.txt", OutputSink.FILESYSTEM_PATH)
        assert isinstance(out, EncodedOutput)

    def test_returns_none_for_unsafe_value(self, guard):
        assert guard.encode_or_none("../escape", OutputSink.FILESYSTEM_PATH) is None


class TestSerialisation:
    def test_encoded_output_to_dict(self, guard):
        d = guard.encode("<x>", OutputSink.HTML_TEXT).to_dict()
        assert set(d) == {"sink", "encoded", "modified", "note"}
        assert d["sink"] == "html_text"

    def test_execution_risk_to_dict(self, guard):
        d = guard.assess("eval(x)").to_dict()
        assert d["safe_to_execute"] is False
        assert d["constructs"] == ["eval_exec"]

    def test_output_execution_risk_default_constructs(self):
        assert OutputExecutionRisk(safe_to_execute=True).constructs == ()


class TestGuardWiring:
    def test_production_guard_exposes_output_trust(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        otg = guard.output_trust
        assert isinstance(otg, ZeroTrustOutputGuard)
        # Cached on the guard.
        assert guard.output_trust is otg
        assert otg.encode("<b>", OutputSink.HTML_TEXT).encoded == "&lt;b&gt;"
