# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for actor.py â€” LLMGenerator stream_tokens fallback."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import patch

from director_ai.core.actor import LLMGenerator


class TestLLMGeneratorStreamFallback:
    def test_stream_tokens_no_httpx(self):
        gen = LLMGenerator(api_url="http://localhost:1234/v1/completions")

        async def run():
            tokens = []
            with (
                patch.dict(sys.modules, {"httpx": None}),
                patch.object(gen, "generate_candidates") as mock_gc,
            ):
                mock_gc.return_value = [{"text": "hello world test"}]
                async for tok in gen.stream_tokens("test"):
                    tokens.append(tok)
            return tokens

        tokens = asyncio.run(run())
        assert len(tokens) == 3
        assert tokens[0] == "hello"
