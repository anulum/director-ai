# Director-AI Wire Schemas

Single source of truth for every message that crosses a process
boundary in the Director-AI stack: Python server ↔ Go gateway,
any-language SDK ↔ gateway, auditor ↔ persisted audit records.

Business-logic types that live inside one language (internal
dataclasses, cache structures) stay in their native code.

## Layout

```
schemas/
├── proto/director/v1/director.proto   # frozen v1 messages + services
├── generate.sh                        # regenerate Python and Go stubs
└── README.md                          # this file
```

Generated output lives outside this directory:

- Python: `src/director_ai/proto/director/v1/`
- Go:     `gateway/go/proto/director/v1/`

Both are committed so the day-to-day dev loop does not need a proto
toolchain. The generator script is the authoritative way to refresh
them — never hand-edit the generated files.

## Regenerating

```bash
# prerequisites: protoc (3.21+), protoc-gen-go, protoc-gen-go-grpc,
# Python grpcio-tools (pinned to the project's grpcio version)
bash schemas/generate.sh
```

The script:

1. finds every `*.proto` under `schemas/proto/`
2. emits `*_pb2.py`, `*_pb2.pyi`, `*_pb2_grpc.py` into the Python
   tree and rewrites imports from the proto-package form
   (`from director.v1 import ...`) to the Python-package form
   (`from director_ai.proto.director.v1 import ...`)
3. emits `*.pb.go` and `*_grpc.pb.go` into the Go tree

## Versioning rule

- `director.v1` is frozen. Existing field numbers and names do not
  change. New fields land as additive entries.
- Breaking changes move to `director.v2`, with a migration note
  in `CHANGELOG.md` and a parallel build period.
- The `option go_package` path must track the version segment.

## Testing

Round-trip tests exist in both ecosystems so a regeneration that
silently drops or renames a field fails loud:

- Python: `tests/test_proto_roundtrip.py` (27 cases)
- Go:     `gateway/go/proto/director/v1/roundtrip_test.go` (5 cases)

Run locally:

```bash
# Python
make test        # or: pytest tests/test_proto_roundtrip.py

# Go
cd gateway/go && go test ./...
```

## Services

`director.proto` currently declares two services:

- `CoherenceScoring` — unary `ScoreClaim` and bidi streaming
  `ScoreStream` for token-level halt decisions.
- `ChatGateway` — OpenAI-compatible chat completion RPCs, unary
  and streaming.

The services are a convenience wrapper — the **messages** are the
contract. Any transport (gRPC, HTTP/JSON via gRPC-Gateway, raw
protobuf over a message bus) can use the same structs.

## Why protobuf over JSON

- A strict schema catches "the Go gateway forwarded an extra field
  the Python server does not validate" at generation time, not at
  3 AM in production logs.
- Zero-copy decoding in Go and Rust; the Python side is not a
  bottleneck, so `protobuf` is fine there too.
- The `.proto` file doubles as documentation that does not rot.

Not a fit: operator-facing config and audit logs, where a
human-readable JSON/JSONL still wins.
