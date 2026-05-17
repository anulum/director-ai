# AutoGen

Guard AutoGen/AG2 group-chat messages before normal auto-reply functions run.
The integration uses AutoGen's `register_reply()` hook shape and does not
import AutoGen at module import time.

```python
from director_ai.agentic import AgentProfile, SwarmGuardian
from director_ai.integrations.autogen_swarm import (
    AutoGenReplyGuard,
    GroupChatGuardian,
)

guardian = SwarmGuardian(hallucination_threshold=0.5)
guardian.register_agent(AgentProfile.for_role("researcher", agent_id="researcher"))
guardian.register_agent(AgentProfile.for_role("reviewer", agent_id="reviewer"))

chat_guard = GroupChatGuardian(guardian)
reply_guard = AutoGenReplyGuard(chat_guard)

reply_guard.install(
    reviewer_agent,
    trigger=["researcher", None],
    config={"chat_context": "Verified project facts and source excerpts."},
)
```

When the latest incoming message passes, the hook returns `(False, None)` so
AutoGen continues to the recipient's normal reply functions. When the message
is suppressed, the hook returns `(True, {"role": "assistant", "content": ...})`
and the sender can be quarantined by `GroupChatGuardian`.

## Manual Filtering

For custom orchestration loops, call the filter directly:

```python
result = chat_guard.filter_message(
    sender="researcher",
    message="Paris is the capital of France.",
    chat_context="Paris is the capital of France topic.",
    recipients=["reviewer"],
)

if result.suppressed:
    print(result.reason)
```

## Statistics

```python
print(chat_guard.stats)
# {"messages": 10, "suppressed": 1, "passed": 9}
```

## Behaviour

| Event | Hook result |
|-------|-------------|
| Grounded message | `(False, None)` |
| Suppressed message | `(True, suppression_reply)` |
| Already quarantined sender | `(True, suppression_reply)` |

Use a custom `suppression_reply` if the AutoGen conversation should receive a
domain-specific fallback message:

```python
reply_guard = AutoGenReplyGuard(
    chat_guard,
    suppression_reply={"role": "assistant", "content": "Message blocked."},
)
```
