# Zero-Trust Output Handling

Treat every model output as untrusted until it has been encoded for the exact
context it is about to enter. The same string is harmless as JSON, an XSS vector
in HTML, command injection in a shell, and path traversal on a filesystem — the
danger is the **sink**, not the text. This is the OWASP-LLM05 posture: nothing the
model produces is rendered, executed, or persisted raw.

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig
from director_ai.core.output_trust import OutputSink

guard = ProductionGuard(DirectorConfig()).output_trust

# Encode the same model output for two different destinations.
html = guard.encode("<script>alert(1)</script>", OutputSink.HTML_TEXT)
print(html.encoded)   # &lt;script&gt;alert(1)&lt;/script&gt;

arg = guard.encode("file; rm -rf /", OutputSink.SHELL_ARGUMENT)
print(arg.encoded)    # 'file; rm -rf /'  (a single, inert shell argument)
```

`encode()` returns an `EncodedOutput`:

| Field | Meaning |
|-------|---------|
| `sink` | The `OutputSink` the value was encoded for. |
| `encoded` | The safe rendering. |
| `modified` | Whether anything had to be neutralised. |
| `note` | A short tenant-safe description of what was neutralised. |

`to_dict()` is tenant-safe — sink, encoded value, flag, and note only; never the
prompt or retrieval context.

## Sinks

| `OutputSink` | Encoding |
|--------------|----------|
| `HTML_TEXT` / `HTML_ATTRIBUTE` | HTML-escape `& < > " '`. |
| `SHELL_ARGUMENT` | POSIX-quote into one inert argument. |
| `SQL_IDENTIFIER` | **Validated** against `[A-Za-z_][A-Za-z0-9_]*`, never escaped. |
| `SQL_STRING_LITERAL` | ANSI single-quote doubling (a bound parameter is still preferred). |
| `FILESYSTEM_PATH` | Normalised to a relative path; absolute paths and `..` traversal are refused. |
| `JSON_VALUE` | Serialised as a JSON string literal. |
| `URL_QUERY` | Percent-encoded. |
| `EMAIL_HEADER` | CR/LF header injection stripped. |
| `LOG_LINE` | CR/LF and control characters stripped. |

Values that **cannot** be made safe — an invalid SQL identifier, an absolute or
traversing path, a NUL byte — raise `UnsafeOutputError` rather than emit something
that only looks safe. Use `encode_or_none()` to drop such a field instead of
handling the exception per call site.

## Refusing code execution

Generated text is data, not code. `assess()` flags constructs that must never be
handed to `exec`/`eval` or an unsandboxed deserialiser:

```python
risk = guard.assess("__import__('os').system('rm -rf /')")
print(risk.safe_to_execute)   # False
print(risk.constructs)        # ('dynamic_import', 'shell_pipe_redirect')
```

`safe_to_execute` is `True` only when no dangerous category matches. Flagged
categories: dynamic import, `eval`/`exec`, OS commands, pickle/`yaml.load`
deserialisation, reflective `globals`/`getattr`, and shell pipe/redirect
metacharacters. `assess()` never executes or transforms the text — it only
reports — so it composes with the per-sink `encode()` step.
