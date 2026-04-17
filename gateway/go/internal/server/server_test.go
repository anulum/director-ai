// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — server integration tests

package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anulum/director-ai/gateway/internal/audit"
	"github.com/anulum/director-ai/gateway/internal/config"
)

func newTestServer(t *testing.T, upstreamURL string, keys []string) (*Components, *bytes.Buffer) {
	t.Helper()
	cfg := &config.Config{
		ListenAddr:      ":0",
		UpstreamURL:     upstreamURL,
		UpstreamTimeout: 5 * time.Second,
		APIKeys:         keys,
		AuditSalt:       []byte("test-salt"),
		RateLimitRPM:    6000, // high enough not to interfere with fast unit tests
		RateLimitBurst:  1000,
	}
	var logBuf bytes.Buffer
	c, err := Build(cfg, audit.New(&logBuf))
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	return c, &logBuf
}

func TestServer_HealthIsOpen(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	c, _ := newTestServer(t, upstream.URL, []string{"sk-a"})
	srv := httptest.NewServer(c.Handler)
	defer srv.Close()

	for _, path := range []string{"/health", "/healthz", "/ready"} {
		resp, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Errorf("%s: status = %d", path, resp.StatusCode)
		}
		_, _ = io.ReadAll(resp.Body)
		resp.Body.Close()
	}
}

func TestServer_UnauthorisedWithoutKey(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	c, _ := newTestServer(t, upstream.URL, []string{"sk-a"})
	srv := httptest.NewServer(c.Handler)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader("{}"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d; want 401", resp.StatusCode)
	}
}

func TestServer_ForwardsWithValidKey(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"echo":` + string(body) + `}`))
	}))
	defer upstream.Close()

	c, logBuf := newTestServer(t, upstream.URL, []string{"sk-a"})
	srv := httptest.NewServer(c.Handler)

	req, _ := http.NewRequest("POST", srv.URL+"/v1/chat/completions", strings.NewReader(`{"ok":1}`))
	req.Header.Set("X-API-Key", "sk-a")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(body), `"ok":1`) {
		t.Errorf("body echo missing: %s", body)
	}
	if resp.Header.Get("X-Request-ID") == "" {
		t.Errorf("X-Request-ID not set")
	}

	// httptest.Server.Close blocks until every in-flight handler
	// goroutine has returned, which is the cleanest way to force
	// the trailing audit write to become visible without a sleep or
	// explicit barrier.
	srv.Close()

	// Audit line should exist with populated fields.
	line := strings.TrimRight(logBuf.String(), "\n")
	if line == "" {
		t.Fatal("audit log empty")
	}
	var rec audit.Record
	if err := json.Unmarshal([]byte(line), &rec); err != nil {
		t.Fatalf("audit decode: %v (line=%q)", err, line)
	}
	if rec.Path != "/v1/chat/completions" || rec.Status != 200 {
		t.Errorf("audit record = %+v", rec)
	}
	if rec.APIKeyFingerprint == "" {
		t.Errorf("expected fingerprint in audit record")
	}
}

func TestServer_RateLimitTriggers429(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	cfg := &config.Config{
		ListenAddr:      ":0",
		UpstreamURL:     upstream.URL,
		UpstreamTimeout: time.Second,
		APIKeys:         []string{"sk-a"},
		AuditSalt:       []byte("salt"),
		RateLimitRPM:    60,
		RateLimitBurst:  1,
	}
	c, err := Build(cfg, audit.New(io.Discard))
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(c.Handler)
	defer srv.Close()

	client := srv.Client()
	doRequest := func() int {
		req, _ := http.NewRequest("POST", srv.URL+"/v1/chat/completions", nil)
		req.Header.Set("X-API-Key", "sk-a")
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		return resp.StatusCode
	}
	if code := doRequest(); code != http.StatusOK {
		t.Fatalf("first request = %d", code)
	}
	if code := doRequest(); code != http.StatusTooManyRequests {
		t.Errorf("second request = %d; want 429", code)
	}
}

func TestServer_RequestIDEchoesClientValue(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	c, _ := newTestServer(t, upstream.URL, nil) // no-auth mode
	srv := httptest.NewServer(c.Handler)
	defer srv.Close()

	req, _ := http.NewRequest("POST", srv.URL+"/v1/chat/completions", nil)
	req.Header.Set("X-Request-ID", "client-supplied-id")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if got := resp.Header.Get("X-Request-ID"); got != "client-supplied-id" {
		t.Errorf("X-Request-ID = %q", got)
	}
}
