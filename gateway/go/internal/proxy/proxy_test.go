// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — proxy tests

package proxy

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestNew_RejectsUnsupportedScheme(t *testing.T) {
	if _, err := New("ftp://example.com", time.Second); err == nil {
		t.Error("expected error for ftp:// upstream")
	}
}

func TestForward_PassthroughBodyAndStatus(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Upstream", "yes")
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"echo":` + strconv.Quote(string(body)) + `}`))
	}))
	defer upstream.Close()

	c, err := New(upstream.URL, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`hello`))
	w := httptest.NewRecorder()
	c.Forward(w, r)
	if w.Code != http.StatusCreated {
		t.Errorf("status = %d; want 201", w.Code)
	}
	if got := w.Result().Header.Get("X-Upstream"); got != "yes" {
		t.Errorf("header not forwarded; got %q", got)
	}
	if !strings.Contains(w.Body.String(), `"hello"`) {
		t.Errorf("body not echoed: %s", w.Body.String())
	}
}

func TestForward_ForwardsPathAndQuery(t *testing.T) {
	var seen *http.Request
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Clone(r.Context())
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	c, _ := New(upstream.URL, time.Second)
	r := httptest.NewRequest("GET", "/v1/models?offset=3", nil)
	w := httptest.NewRecorder()
	c.Forward(w, r)
	if seen == nil {
		t.Fatal("upstream did not observe request")
	}
	if seen.URL.Path != "/v1/models" {
		t.Errorf("upstream path = %q", seen.URL.Path)
	}
	if seen.URL.RawQuery != "offset=3" {
		t.Errorf("upstream query = %q", seen.URL.RawQuery)
	}
}

func TestForward_StreamsChunksWithFlush(t *testing.T) {
	chunks := []string{"data: one\n\n", "data: two\n\n"}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		for _, c := range chunks {
			_, _ = w.Write([]byte(c))
			flusher.Flush()
			time.Sleep(5 * time.Millisecond)
		}
	}))
	defer upstream.Close()

	c, _ := New(upstream.URL, 5*time.Second)
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	c.Forward(w, r)
	body := w.Body.String()
	for _, chunk := range chunks {
		if !strings.Contains(body, chunk) {
			t.Errorf("chunk %q missing from body %q", chunk, body)
		}
	}
}

func TestForward_UpstreamErrorReturns502(t *testing.T) {
	// Point at a closed listener so Do returns an error.
	c, _ := New("http://127.0.0.1:1", 500*time.Millisecond)
	r := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()
	c.Forward(w, r)
	if w.Code != http.StatusBadGateway {
		t.Errorf("status = %d; want 502", w.Code)
	}
}

func TestForward_DoesNotForwardAuthWhenDisabled(t *testing.T) {
	var seenAuth string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	c, _ := New(upstream.URL, time.Second)
	c.ForwardAuthHeader(false)
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("Authorization", "Bearer secret-downstream")
	w := httptest.NewRecorder()
	c.Forward(w, r)
	if seenAuth != "" {
		t.Errorf("Authorization leaked upstream: %q", seenAuth)
	}
}

func TestForward_ForwardsAuthByDefault(t *testing.T) {
	var seenAuth string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	c, _ := New(upstream.URL, time.Second)
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("Authorization", "Bearer downstream")
	w := httptest.NewRecorder()
	c.Forward(w, r)
	if seenAuth != "Bearer downstream" {
		t.Errorf("Authorization not forwarded: %q", seenAuth)
	}
}

func TestForward_StripsHopByHopHeaders(t *testing.T) {
	var seen http.Header
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Clone()
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	c, _ := New(upstream.URL, time.Second)
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("Connection", "close")
	r.Header.Set("Te", "trailers")
	r.Header.Set("Upgrade", "h2c")
	w := httptest.NewRecorder()
	c.Forward(w, r)
	for _, h := range []string{"Connection", "Te", "Upgrade"} {
		if seen.Get(h) != "" {
			t.Errorf("hop-by-hop header %q leaked: %q", h, seen.Get(h))
		}
	}
}
