// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — token-bucket tests

package ratelimit

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestAllow_ZeroRPMMeansUnlimited(t *testing.T) {
	l := New(0, 0)
	for i := 0; i < 10000; i++ {
		if allowed, _ := l.Allow("x"); !allowed {
			t.Fatalf("zero RPM must never block; blocked at i=%d", i)
		}
	}
}

func TestAllow_BurstThenBlock(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	clock := func() time.Time { return t0 }
	l := NewWithClock(60, 3, clock)
	for i := 0; i < 3; i++ {
		if allowed, _ := l.Allow("k"); !allowed {
			t.Fatalf("burst of 3 should succeed; blocked at %d", i)
		}
	}
	if allowed, retry := l.Allow("k"); allowed {
		t.Fatal("4th request should block")
	} else if retry <= 0 {
		t.Errorf("retry should be positive; got %v", retry)
	}
}

func TestAllow_RefillsOverTime(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	clock := func() time.Time { return now }
	l := NewWithClock(60, 2, clock)

	l.Allow("k")
	l.Allow("k")
	if allowed, _ := l.Allow("k"); allowed {
		t.Fatal("precondition: bucket empty")
	}
	// 60 RPM = 1 token per second. Advance 1.2s → one refilled token.
	now = now.Add(1200 * time.Millisecond)
	if allowed, _ := l.Allow("k"); !allowed {
		t.Errorf("expected refill after 1.2s")
	}
}

func TestAllow_CapsAtBurst(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	clock := func() time.Time { return now }
	l := NewWithClock(60, 2, clock)
	// Long idle → bucket should cap at `burst`, not grow unbounded.
	now = now.Add(time.Hour)
	for i := 0; i < 2; i++ {
		if allowed, _ := l.Allow("k"); !allowed {
			t.Fatalf("token %d should be allowed after cap", i)
		}
	}
	if allowed, _ := l.Allow("k"); allowed {
		t.Errorf("3rd token should block — cap must be burst=2")
	}
}

func TestAllow_SeparateBucketsPerKey(t *testing.T) {
	l := New(60, 1)
	if allowed, _ := l.Allow("alice"); !allowed {
		t.Fatal("alice first token should pass")
	}
	if allowed, _ := l.Allow("bob"); !allowed {
		t.Fatal("bob first token should pass — not sharing alice's bucket")
	}
	if allowed, _ := l.Allow("alice"); allowed {
		t.Error("alice 2nd token should block (burst=1)")
	}
}

func TestHandler_RateLimited429(t *testing.T) {
	l := New(60, 2)
	handler := l.Handler(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}), func(*http.Request) string { return "bucket-a" })

	for i := 0; i < 2; i++ {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, httptest.NewRequest("POST", "/", nil))
		if w.Code != http.StatusOK {
			t.Fatalf("burst slot %d: status %d", i, w.Code)
		}
	}
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, httptest.NewRequest("POST", "/", nil))
	if w.Code != http.StatusTooManyRequests {
		t.Errorf("expected 429; got %d", w.Code)
	}
	if retry := w.Result().Header.Get("Retry-After"); retry == "" {
		t.Error("Retry-After header missing")
	} else if n, _ := strconv.Atoi(retry); n < 1 {
		t.Errorf("Retry-After = %q; want >= 1", retry)
	}
	if !strings.Contains(w.Body.String(), "rate limit") {
		t.Errorf("body = %q", w.Body.String())
	}
}

func TestHandler_FallsBackToRemoteAddr(t *testing.T) {
	l := New(60, 1)
	handler := l.Handler(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}), func(*http.Request) string { return "" })

	r1 := httptest.NewRequest("POST", "/", nil)
	r1.RemoteAddr = "10.0.0.1:1234"
	w1 := httptest.NewRecorder()
	handler.ServeHTTP(w1, r1)
	if w1.Code != http.StatusOK {
		t.Fatalf("first request: %d", w1.Code)
	}
	r2 := httptest.NewRequest("POST", "/", nil)
	r2.RemoteAddr = "10.0.0.1:5678" // same host
	w2 := httptest.NewRecorder()
	handler.ServeHTTP(w2, r2)
	if w2.Code != http.StatusTooManyRequests {
		t.Errorf("same remote host should share bucket; got %d", w2.Code)
	}
	r3 := httptest.NewRequest("POST", "/", nil)
	r3.RemoteAddr = "10.0.0.2:1234" // different host
	w3 := httptest.NewRecorder()
	handler.ServeHTTP(w3, r3)
	if w3.Code != http.StatusOK {
		t.Errorf("different remote host should have own bucket; got %d", w3.Code)
	}
}

func TestClientAddr_HandlesXForwardedFor(t *testing.T) {
	r := httptest.NewRequest("POST", "/", nil)
	r.Header.Set("X-Forwarded-For", "203.0.113.5, 10.0.0.1")
	if got := clientAddr(r); got != "203.0.113.5" {
		t.Errorf("got %q", got)
	}
}

func TestClientAddr_StripsPort(t *testing.T) {
	r := httptest.NewRequest("POST", "/", nil)
	r.RemoteAddr = "10.0.0.1:54321"
	if got := clientAddr(r); got != "10.0.0.1" {
		t.Errorf("got %q", got)
	}
}

func TestAllow_ConcurrentAccess(t *testing.T) {
	l := New(60, 100)
	var wg sync.WaitGroup
	allowed := 0
	var mu sync.Mutex
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if ok, _ := l.Allow("shared"); ok {
				mu.Lock()
				allowed++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	// Exactly burst tokens should pass before refills kick in.
	// Allow tiny refill slack (1s walls at 60 RPM = 1 extra).
	if allowed < 100 || allowed > 102 {
		t.Errorf("concurrent allowed=%d; want ≈100", allowed)
	}
}

func TestReset_ClearsBuckets(t *testing.T) {
	l := New(60, 1)
	l.Allow("k")
	if ok, _ := l.Allow("k"); ok {
		t.Fatal("precondition: bucket empty")
	}
	l.Reset()
	if ok, _ := l.Allow("k"); !ok {
		t.Errorf("Reset should restore the bucket")
	}
}
