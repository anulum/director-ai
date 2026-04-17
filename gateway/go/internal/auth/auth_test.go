// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — auth middleware tests

package auth

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func ok(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func TestFingerprint_DeterministicAndSaltSensitive(t *testing.T) {
	m := New([]string{"sk-a"}, []byte("salt-1"))
	fp1 := m.Fingerprint("sk-a")
	fp2 := m.Fingerprint("sk-a")
	if fp1 != fp2 {
		t.Errorf("same key+salt must produce stable fingerprint")
	}
	m2 := New([]string{"sk-a"}, []byte("salt-2"))
	if m2.Fingerprint("sk-a") == fp1 {
		t.Errorf("fingerprint must depend on salt")
	}
	if len(fp1) != 16 {
		t.Errorf("fingerprint len = %d; want 16 (hex chars)", len(fp1))
	}
}

func TestHandler_ExemptPathsBypassAuth(t *testing.T) {
	m := New([]string{"sk-a"}, []byte("salt"))
	h := m.Handler(http.HandlerFunc(ok))
	for _, p := range []string{"/", "/health", "/healthz", "/ready", "/metrics"} {
		r := httptest.NewRequest("GET", p, nil)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != http.StatusOK {
			t.Errorf("%s: expected 200, got %d", p, w.Code)
		}
	}
}

func TestHandler_NoKeysMeansNoAuth(t *testing.T) {
	m := New(nil, []byte("salt"))
	h := m.Handler(http.HandlerFunc(ok))
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200 in no-auth mode; got %d", w.Code)
	}
}

func TestHandler_RejectsMissingKey(t *testing.T) {
	m := New([]string{"sk-a"}, []byte("salt"))
	h := m.Handler(http.HandlerFunc(ok))
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401; got %d", w.Code)
	}
}

func TestHandler_RejectsWrongKey(t *testing.T) {
	m := New([]string{"sk-good"}, []byte("salt"))
	h := m.Handler(http.HandlerFunc(ok))
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("X-API-Key", "sk-bad")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401; got %d", w.Code)
	}
}

func TestHandler_AcceptsAPIKeyHeader(t *testing.T) {
	m := New([]string{"sk-good"}, []byte("salt"))
	h := m.Handler(http.HandlerFunc(ok))
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("X-API-Key", "sk-good")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200; got %d", w.Code)
	}
}

func TestHandler_AcceptsBearerHeader(t *testing.T) {
	m := New([]string{"sk-good"}, []byte("salt"))
	h := m.Handler(http.HandlerFunc(ok))
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("Authorization", "Bearer sk-good")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200; got %d", w.Code)
	}
}

func TestHandler_FingerprintPropagates(t *testing.T) {
	m := New([]string{"sk-good"}, []byte("salt"))
	var seen string
	captured := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = FingerprintFromContext(r)
		w.WriteHeader(http.StatusOK)
	})
	h := m.Handler(captured)
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("X-API-Key", "sk-good")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if len(seen) != 16 {
		t.Errorf("fingerprint len = %d", len(seen))
	}
	if seen != m.Fingerprint("sk-good") {
		t.Errorf("context fingerprint mismatch")
	}
}

func TestSetKeys_RuntimeRotation(t *testing.T) {
	m := New([]string{"sk-old"}, []byte("salt"))
	h := m.Handler(http.HandlerFunc(ok))
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("X-API-Key", "sk-old")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("precondition: expected 200")
	}
	m.SetKeys([]string{"sk-new"})
	r2 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r2.Header.Set("X-API-Key", "sk-old")
	w2 := httptest.NewRecorder()
	h.ServeHTTP(w2, r2)
	if w2.Code != http.StatusUnauthorized {
		t.Errorf("old key should have been rotated out; got %d", w2.Code)
	}
}

func TestExtractKey_PrefersBearer(t *testing.T) {
	r := httptest.NewRequest("POST", "/", nil)
	r.Header.Set("Authorization", "Bearer  sk-via-bearer ")
	r.Header.Set("X-API-Key", "sk-via-header")
	if got := extractKey(r); got != "sk-via-bearer" {
		t.Errorf("got %q", got)
	}
}

func TestExtractKey_CaseInsensitiveScheme(t *testing.T) {
	r := httptest.NewRequest("POST", "/", nil)
	r.Header.Set("Authorization", "BEARER sk-upper")
	if got := extractKey(r); got != "sk-upper" {
		t.Errorf("got %q", got)
	}
}

func TestFingerprintFromContext_EmptyWhenUnset(t *testing.T) {
	r := httptest.NewRequest("GET", "/", nil)
	if got := FingerprintFromContext(r); got != "" {
		t.Errorf("got %q; want empty", got)
	}
}

func TestReject_WritesJSON(t *testing.T) {
	w := httptest.NewRecorder()
	reject(w, "test message")
	if w.Code != http.StatusUnauthorized {
		t.Errorf("status = %d", w.Code)
	}
	if ct := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("content-type = %q", ct)
	}
	if !strings.Contains(w.Body.String(), "test message") {
		t.Errorf("body = %q", w.Body.String())
	}
}
