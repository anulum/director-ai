# Docker Deployment

## Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    director-ai[server,minicheck,vector,embeddings]

COPY config.yaml .
COPY data/ data/

EXPOSE 8080
CMD ["uvicorn", "director_ai.server:app", "--host", "0.0.0.0", "--port", "8080"]
```

## Docker Compose

```yaml
services:
  director-ai:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DIRECTOR_USE_NLI=true
      - DIRECTOR_NLI_MODEL=lytang/MiniCheck-DeBERTa-L
      - DIRECTOR_VECTOR_BACKEND=chroma
      - DIRECTOR_CHROMA_PERSIST_DIR=/data/chroma
      - DIRECTOR_METRICS_ENABLED=true
    volumes:
      - chroma_data:/data/chroma
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  chroma_data:
```

## GPU Support

For GPU-accelerated NLI, use the NVIDIA base image:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install director-ai[server,nli,quantize,vector]
```
