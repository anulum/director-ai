// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — passthrough HTTP proxy to the upstream LLM API

// Package proxy forwards client requests to an upstream OpenAI-
// compatible endpoint. No scoring integration yet — that is Phase 3.
// The point of this phase is the Go hop itself: terminating TLS,
// counting requests, and streaming server-sent events without buffering.
package proxy

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Client is a thin HTTP client that forwards one request to the
// upstream API. It is safe for concurrent use.
type Client struct {
	upstream    *url.URL
	httpClient  *http.Client
	forwardAuth bool
}

// New builds a Client pointing at ``upstreamURL``. ``timeout`` is the
// per-request total timeout; set to zero to disable.
func New(upstreamURL string, timeout time.Duration) (*Client, error) {
	u, err := url.Parse(upstreamURL)
	if err != nil {
		return nil, fmt.Errorf("parse upstream: %w", err)
	}
	if u.Scheme != "https" && u.Scheme != "http" {
		return nil, fmt.Errorf("unsupported upstream scheme: %q", u.Scheme)
	}
	return &Client{
		upstream: u,
		httpClient: &http.Client{
			Timeout: timeout,
		},
		forwardAuth: true,
	}, nil
}

// ForwardAuthHeader toggles whether the client forwards the incoming
// ``Authorization`` header upstream. Production deployments that
// replace the inbound key with a service account should set this to
// false and supply their own header via Inject.
func (c *Client) ForwardAuthHeader(on bool) { c.forwardAuth = on }

// Forward proxies the incoming request to the upstream, copying the
// status code, content-type, and body through. It honours streaming
// responses by flushing as soon as data arrives.
func (c *Client) Forward(w http.ResponseWriter, r *http.Request) {
	upstreamURL := *c.upstream
	upstreamURL.Path = strings.TrimSuffix(upstreamURL.Path, "/") + r.URL.Path
	upstreamURL.RawQuery = r.URL.RawQuery

	ctx := r.Context()
	// Cap the request at the HTTP client's configured timeout if the
	// caller didn't already supply a deadline.
	if c.httpClient.Timeout > 0 {
		if _, ok := ctx.Deadline(); !ok {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, c.httpClient.Timeout)
			defer cancel()
		}
	}

	outReq, err := http.NewRequestWithContext(ctx, r.Method, upstreamURL.String(), r.Body)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "upstream request build failed")
		return
	}
	copyRequestHeaders(outReq, r, c.forwardAuth)

	resp, err := c.httpClient.Do(outReq)
	if err != nil {
		writeError(w, http.StatusBadGateway, "upstream unreachable: "+err.Error())
		return
	}
	defer resp.Body.Close()

	copyResponseHeaders(w, resp)
	w.WriteHeader(resp.StatusCode)

	// Streaming path: flush as we go so SSE clients see chunks
	// immediately. Buffer size is a balance between allocation churn
	// (too small) and streaming latency (too large).
	buf := make([]byte, 4096)
	flusher, _ := w.(http.Flusher)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if readErr == io.EOF {
			return
		}
		if readErr != nil {
			return
		}
	}
}

// Rewritable request headers the gateway forwards. Hop-by-hop
// headers per RFC 7230 § 6.1 are stripped, as is Host so the
// upstream's TLS name is matched correctly.
var hopByHopHeaders = map[string]struct{}{
	"Connection":          {},
	"Keep-Alive":          {},
	"Proxy-Authenticate":  {},
	"Proxy-Authorization": {},
	"Te":                  {},
	"Trailer":             {},
	"Transfer-Encoding":   {},
	"Upgrade":             {},
}

func copyRequestHeaders(dst, src *http.Request, forwardAuth bool) {
	for k, values := range src.Header {
		if _, hop := hopByHopHeaders[k]; hop {
			continue
		}
		if !forwardAuth && strings.EqualFold(k, "Authorization") {
			continue
		}
		for _, v := range values {
			dst.Header.Add(k, v)
		}
	}
	if dst.Header.Get("X-Forwarded-For") == "" && src.RemoteAddr != "" {
		dst.Header.Set("X-Forwarded-For", src.RemoteAddr)
	}
	dst.Header.Set("X-Forwarded-Proto", schemeFromScheme(src))
}

func copyResponseHeaders(w http.ResponseWriter, src *http.Response) {
	for k, values := range src.Header {
		if _, hop := hopByHopHeaders[k]; hop {
			continue
		}
		for _, v := range values {
			w.Header().Add(k, v)
		}
	}
}

func schemeFromScheme(r *http.Request) string {
	if r.TLS != nil {
		return "https"
	}
	return "http"
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = fmt.Fprintf(w, `{"error":{"message":%q,"type":"gateway_error"}}`, msg)
}
