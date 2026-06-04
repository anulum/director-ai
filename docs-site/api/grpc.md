<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — gRPC server API
-->

# gRPC Server

Director-AI ships two gRPC surfaces:

- `DirectorService`, the legacy Python transport implemented by
  `director_ai.grpc_server` from `proto/director.proto`.
- `director.v1.CoherenceScoring`, the gateway-facing scoring service
  implemented by `director_ai.grpc_scoring` from
  `schemas/proto/director/v1/director.proto`.

Install the optional transport dependencies before using either service:

```bash
pip install director-ai[grpc]
```

## Legacy DirectorService

`create_grpc_server()` serves `DirectorService` on port `50051` by default.
Use it when an existing Python client already depends on the legacy proto.

=== "CLI"

    ```bash
    director-ai serve --transport grpc --port 50051 --workers 4
    ```

=== "Python"

    ```python
    from director_ai.grpc_server import create_grpc_server

    server = create_grpc_server(port=50051)
    server.start()
    server.wait_for_termination()
    ```

=== "With TLS"

    ```python
    server = create_grpc_server(
        port=50051,
        tls_cert_path="/path/to/cert.pem",
        tls_key_path="/path/to/key.pem",
    )
    ```

### DirectorService RPCs

| Method | Request | Response | Description |
|--------|---------|----------|-------------|
| `Review` | `ReviewRequest` | `ReviewResponse` | Score a single prompt/response pair |
| `Process` | `ProcessRequest` | `ProcessResponse` | Generate and score one guarded response |
| `ReviewBatch` | `BatchReviewRequest` | `BatchReviewResponse` | Score a bounded batch of prompt/response pairs |
| `StreamTokens` | `StreamRequest` | stream `TokenEvent` | Stream generated tokens with coherence metadata |

```protobuf
service DirectorService {
  rpc Review (ReviewRequest) returns (ReviewResponse);
  rpc Process (ProcessRequest) returns (ProcessResponse);
  rpc ReviewBatch (BatchReviewRequest) returns (BatchReviewResponse);
  rpc StreamTokens (StreamRequest) returns (stream TokenEvent);
}
```

### Legacy Python Client

```python
import grpc
from director_ai import director_pb2, director_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = director_pb2_grpc.DirectorServiceStub(channel)

response = stub.Review(
    director_pb2.ReviewRequest(
        prompt="What is the capital of France?",
        response="Paris.",
    )
)
print(f"Approved: {response.approved}, coherence: {response.coherence:.3f}")
```

## CoherenceScoring

`director_ai.grpc_scoring.serve()` serves `director.v1.CoherenceScoring` on
port `50052` by default. This is the low-latency scoring contract used by the
gateway-facing protocol and preserves tenant/request metadata on the wire.

```python
from director_ai.grpc_scoring import serve

server, port = serve(listen_addr="[::]:50052")
server.wait_for_termination()
```

### CoherenceScoring RPCs

| Method | Request | Response | Description |
|--------|---------|----------|-------------|
| `ScoreClaim` | `ScoreClaimRequest` | `ScoreClaimResponse` | Score one claim against optional supporting documents |
| `ScoreStream` | stream `ScoreTokenRequest` | stream `ScoreTokenResponse` | Score accumulated candidate output per token or chunk |

```protobuf
service CoherenceScoring {
  rpc ScoreClaim(ScoreClaimRequest) returns (ScoreClaimResponse);
  rpc ScoreStream(stream ScoreTokenRequest)
      returns (stream ScoreTokenResponse);
}
```

## create_grpc_server()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `DirectorConfig \| None` | `None` | Configuration |
| `max_workers` | `int` | `4` | Thread pool size |
| `port` | `int` | `50051` | Listen port |
| `tls_cert_path` | `str \| None` | `None` | TLS certificate path |
| `tls_key_path` | `str \| None` | `None` | TLS key path |

## Full API

::: director_ai.grpc_server.create_grpc_server

::: director_ai.grpc_scoring.serve
