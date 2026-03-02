# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Security Hardening Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from director_ai.core.sanitizer import InputSanitizer


class TestPathTraversal:
    def test_unix_traversal_blocked(self):
        san = InputSanitizer()
        result = san.check("Load file ../../etc/passwd")
        assert result.blocked
        assert result.pattern == "path_traversal"

    def test_windows_traversal_blocked(self):
        san = InputSanitizer()
        result = san.check("Read ..\\..\\windows\\system32\\config")
        assert result.blocked

    def test_url_encoded_traversal_blocked(self):
        san = InputSanitizer()
        result = san.check("..%2F..%2Fetc%2Fshadow")
        assert result.blocked


class TestNullByteStripping:
    def test_scrub_removes_null_bytes(self):
        assert "\x00" not in InputSanitizer.scrub("hello\x00world")

    def test_scrub_preserves_content(self):
        assert InputSanitizer.scrub("hello\x00world") == "helloworld"


class TestLargeInput:
    def test_oversized_input_blocked(self):
        san = InputSanitizer(max_length=100)
        result = san.check("A" * 200)
        assert result.blocked
        assert result.pattern == "length"

    def test_exactly_at_limit_passes(self):
        san = InputSanitizer(max_length=100)
        result = san.check("A" * 100)
        assert not result.blocked


class TestYAMLInjection:
    def test_python_object_blocked(self):
        san = InputSanitizer()
        result = san.check("!!python/object:os.system ['rm -rf /']")
        assert result.blocked
        assert result.pattern == "yaml_json_injection"

    def test_unsafe_load_blocked(self):
        san = InputSanitizer()
        result = san.check("yaml.unsafe_load(data)")
        assert result.blocked

    def test_dunder_import_blocked(self):
        san = InputSanitizer()
        result = san.check("__import__('os').system('whoami')")
        assert result.blocked


class TestControlCharInjection:
    def test_escape_sequence_blocked(self):
        san = InputSanitizer()
        result = san.check("Normal text\x1b[31m RED ALERT")
        assert result.blocked
        assert result.pattern == "control_char_injection"

    def test_scrub_strips_control_chars(self):
        cleaned = InputSanitizer.scrub("hello\x1bworld\x7fend")
        assert "\x1b" not in cleaned
        assert "\x7f" not in cleaned


class TestBidiOverride:
    def test_bidi_override_blocked(self):
        san = InputSanitizer()
        result = san.check("Price: \u202e 99.1$ normal text")
        assert result.blocked
        assert result.pattern == "bidi_override"


class TestExtraPatterns:
    def test_custom_pattern_registered(self):
        san = InputSanitizer(extra_patterns=[("custom_bad", r"EVIL_PATTERN")])
        result = san.check("Contains EVIL_PATTERN here")
        assert result.blocked
        assert result.pattern == "custom_bad"

    def test_clean_input_passes(self):
        san = InputSanitizer()
        result = san.check("The quick brown fox jumps over the lazy dog.")
        assert not result.blocked
