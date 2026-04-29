# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - CORS reverse proxy examples

# CORS Reverse Proxy

CORS protects browser callers. It does not replace caller credentials, tenant binding, rate limits, or private routing for operational endpoints.

Director-AI defaults to no browser origins. Enable browser access only for exact user interface origins:

```bash
DIRECTOR_CORS_ORIGINS=https://app.example.com,https://admin.example.com \
director-ai serve
```

Keep service-to-service clients outside CORS. Command-line tools, backend workers, Prometheus, and internal schedulers should use private network routes and caller credentials.

## Header Contract

The application middleware allows:

- Methods: `GET`, `POST`, `PUT`, `DELETE`, `OPTIONS`
- Headers: `Authorization`, `Content-Type`, `X-API-Key`, `X-Request-ID`, `X-Tenant-ID`, `X-KB-Key-ID`, `X-KB-Signature`
- Credentials: disabled at the CORS layer; use explicit request headers instead

Reverse proxies should mirror this contract. Do not add wildcard origins at the proxy when the application has exact origins.

## Nginx

Use a map with exact origins and return 403 for browser origins outside the map.
Define the origin map in the `http` block:

```nginx
map $http_origin $director_cors_origin {
    default "";
    "https://app.example.com" $http_origin;
    "https://admin.example.com" $http_origin;
}
```

Use the map in the public server:

```nginx
server {
    listen 443 ssl http2;
    server_name api.example.com;

    if ($http_origin != "") {
        set $director_has_origin 1;
    }
    if ($director_cors_origin = "") {
        set $director_has_origin "${director_has_origin}0";
    }
    if ($director_has_origin = "10") {
        return 403;
    }

    add_header Access-Control-Allow-Origin $director_cors_origin always;
    add_header Vary Origin always;
    add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Authorization, Content-Type, X-API-Key, X-Request-ID, X-Tenant-ID, X-KB-Key-ID, X-KB-Signature" always;
    add_header Access-Control-Max-Age 600 always;

    location / {
        if ($request_method = OPTIONS) {
            return 204;
        }
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Request-ID $request_id;
    }
}
```

## Traefik

Use accessControlAllowOriginList with exact browser origins.

```yaml
http:
  middlewares:
    director-cors:
      headers:
        accessControlAllowOriginList:
          - https://app.example.com
          - https://admin.example.com
        accessControlAllowMethods:
          - GET
          - POST
          - PUT
          - DELETE
          - OPTIONS
        accessControlAllowHeaders:
          - Authorization
          - Content-Type
          - X-API-Key
          - X-Request-ID
          - X-Tenant-ID
          - X-KB-Key-ID
          - X-KB-Signature
        accessControlAllowCredentials: false
        accessControlMaxAge: 600
        addVaryHeader: true
```

## Ingress Nginx

Use cors-allow-origin with an exact comma-separated allowlist.

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://app.example.com, https://admin.example.com"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"
    nginx.ingress.kubernetes.io/cors-allow-headers: "Authorization, Content-Type, X-API-Key, X-Request-ID, X-Tenant-ID, X-KB-Key-ID, X-KB-Signature"
    nginx.ingress.kubernetes.io/cors-allow-credentials: "false"
    nginx.ingress.kubernetes.io/cors-max-age: "600"
```

## Checks

- `DIRECTOR_CORS_ORIGINS` is empty for non-browser deployments.
- Browser deployments list only exact `https://` origins.
- Reverse proxy CORS and application CORS use the same origin list.
- Operational endpoints stay behind private routes as described in [Public Endpoint Exposure](public-endpoints.md).
- Preflight requests from unknown origins receive a denial response.
