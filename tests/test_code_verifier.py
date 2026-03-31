# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for code output verification."""

from __future__ import annotations

from director_ai.core.verification.code_verifier import verify_code


class TestPythonSyntax:
    def test_valid_python(self):
        r = verify_code("x = 1 + 2\nprint(x)")
        assert r.syntax_valid is True
        assert r.error_count == 0

    def test_invalid_python(self):
        r = verify_code("def foo(:\n  pass")
        assert r.syntax_valid is False
        assert "SyntaxError" in r.parse_error

    def test_empty_code(self):
        r = verify_code("")
        assert r.syntax_valid is True

    def test_multiline_function(self):
        code = "def greet(name):\n    return f'Hello, {name}!'"
        r = verify_code(code)
        assert r.syntax_valid is True


class TestJsonSyntax:
    def test_valid_json(self):
        r = verify_code('{"key": "value"}', language="json")
        assert r.syntax_valid is True

    def test_invalid_json(self):
        r = verify_code("{key: value}", language="json")
        assert r.syntax_valid is False

    def test_unsupported_language(self):
        r = verify_code("SELECT * FROM users", language="sql")
        assert r.syntax_valid is True
        assert "not supported" in r.parse_error


class TestImportChecking:
    def test_stdlib_import(self):
        r = verify_code("import os\nimport json")
        assert r.unknown_imports == []

    def test_known_package(self):
        r = verify_code("import numpy as np\nimport pandas as pd")
        assert r.unknown_imports == []

    def test_unknown_import(self):
        r = verify_code("import nonexistent_package_xyz")
        assert "nonexistent_package_xyz" in r.unknown_imports
        assert r.error_count == 1

    def test_from_import(self):
        r = verify_code("from fake_module import something")
        assert "fake_module" in r.unknown_imports

    def test_custom_known_modules(self):
        r = verify_code("import my_company_lib", known_modules={"my_company_lib"})
        assert r.unknown_imports == []

    def test_mixed_imports(self):
        r = verify_code("import os\nimport fake_lib\nimport json")
        assert r.unknown_imports == ["fake_lib"]


class TestHallucinatedAPIs:
    def test_known_api(self):
        manifest = {"pd": {"read_csv", "DataFrame", "merge"}}
        r = verify_code(
            "import pandas as pd\ndf = pd.read_csv('data.csv')",
            api_manifest=manifest,
        )
        assert r.hallucinated_apis == []

    def test_hallucinated_api(self):
        manifest = {"pd": {"read_csv", "DataFrame"}}
        r = verify_code(
            "import pandas as pd\ndf = pd.read_quantum_csv('data.csv')",
            api_manifest=manifest,
        )
        assert "pd.read_quantum_csv" in r.hallucinated_apis

    def test_no_manifest(self):
        r = verify_code("import pandas as pd\ndf = pd.anything()")
        assert r.hallucinated_apis == []

    def test_builtin_call_not_flagged(self):
        manifest = {"pd": {"read_csv"}}
        r = verify_code("print('hello')\nlen([1,2,3])", api_manifest=manifest)
        assert r.hallucinated_apis == []

    def test_multiple_hallucinated_apis(self):
        manifest = {"np": {"array", "zeros"}}
        code = "import numpy as np\na = np.quantum_sort([1,2])\nb = np.neural_fft(a)"
        r = verify_code(code, api_manifest=manifest)
        assert len(r.hallucinated_apis) == 2


class TestCombined:
    def test_syntax_error_skips_analysis(self):
        r = verify_code("import os\ndef broken(:", api_manifest={"os": {"path"}})
        assert r.syntax_valid is False
        assert r.unknown_imports == []
        assert r.hallucinated_apis == []
