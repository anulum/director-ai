# gRPC Server

Protocol Buffers-based gRPC service for low-latency, high-throughput scoring. Supports TLS and bidirectional streaming.

```bash
pip install director-ai[grpc]
```

## Starting the Server

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

## RPC Methods

| Method | Request | Response | Description |
|--------|---------|----------|-------------|
| `Review` | `ReviewRequest` | `ReviewResponse` | Score a single pair |
| `ReviewBatch` | `BatchRequest` | `BatchResponse` | Batch scoring |
| `StreamReview` | stream `TokenChunk` | `StreamResult` | Streaming oversight |
| `HealthCheck` | `Empty` | `HealthResponse` | Server health |

## Proto Definition

```protobuf
service DirectorService {
  rpc Review (ReviewRequest) returns (ReviewResponse);
  rpc ReviewBatch (BatchRequest) returns (BatchResponse);
  rpc StreamReview (stream TokenChunk) returns (StreamResult);
  rpc HealthCheck (google.protobuf.Empty) returns (HealthResponse);
}
```

## Python Client

```python
import grpc
from director_ai.proto import director_pb2, director_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = director_pb2_grpc.DirectorServiceStub(channel)

response = stub.Review(director_pb2.ReviewRequest(
    prompt="What is the capital?",
    response="Paris.",
))
print(f"Approved: {response.approved}, Score: {response.score:.3f}")
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
