// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — per-key token-bucket rate limiter

// Package ratelimit implements a token-bucket rate limiter keyed by
// API-key fingerprint. An unauthenticated request is bucketed by
// remote address so that a misconfigured client with no key still
// cannot DoS the gateway.
//
// Storage is in-memory and per-process — fine for a single gateway
// replica. A shared Redis-backed implementation is a v2 concern.
package ratelimit

import (
	"net"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// Limiter enforces a per-key rate budget using the token-bucket
// algorithm. Zero-value Limiter is unusable; construct with New.
type Limiter struct {
	rpm     int
	burst   int
	clock   func() time.Time
	mu      sync.Mutex
	buckets map[string]*bucket
}

// bucket tracks the running token count for one client.
type bucket struct {
	tokens float64
	last   time.Time
}

// New returns a Limiter that allows ``rpm`` requests per minute with
// up to ``burst`` tokens on the gauge at any time. ``rpm <= 0``
// disables rate limiting entirely — the Handler becomes a passthrough.
func New(rpm, burst int) *Limiter {
	return &Limiter{
		rpm:     rpm,
		burst:   burst,
		clock:   time.Now,
		buckets: make(map[string]*bucket),
	}
}

// NewWithClock is for tests: it lets callers inject a deterministic
// clock so the bucket fills at a known rate.
func NewWithClock(rpm, burst int, clock func() time.Time) *Limiter {
	l := New(rpm, burst)
	l.clock = clock
	return l
}

// Allow decrements one token from the bucket keyed by ``id``. Returns
// (allowed, retryAfter). ``retryAfter`` is only meaningful when
// ``allowed`` is false.
func (l *Limiter) Allow(id string) (bool, time.Duration) {
	if l.rpm <= 0 {
		return true, 0
	}
	now := l.clock()
	refill := float64(l.rpm) / 60.0

	l.mu.Lock()
	defer l.mu.Unlock()
	b, ok := l.buckets[id]
	if !ok {
		b = &bucket{tokens: float64(l.burst), last: now}
		l.buckets[id] = b
	}
	elapsed := now.Sub(b.last).Seconds()
	if elapsed > 0 {
		b.tokens += elapsed * refill
		if b.tokens > float64(l.burst) {
			b.tokens = float64(l.burst)
		}
		b.last = now
	}
	if b.tokens >= 1.0 {
		b.tokens -= 1.0
		return true, 0
	}
	missing := 1.0 - b.tokens
	wait := time.Duration(missing/refill*float64(time.Second)) + time.Millisecond
	return false, wait
}

// Reset removes every bucket. Useful for tests and for a future
// admin-only "flush" endpoint.
func (l *Limiter) Reset() {
	l.mu.Lock()
	l.buckets = make(map[string]*bucket)
	l.mu.Unlock()
}

// Handler wraps next with rate limiting. The bucket key is the
// audit fingerprint when present, otherwise the remote address.
func (l *Limiter) Handler(next http.Handler, fingerprintFn func(*http.Request) string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := fingerprintFn(r)
		if id == "" {
			id = clientAddr(r)
		}
		if allowed, retry := l.Allow(id); !allowed {
			w.Header().Set("Retry-After", strconv.Itoa(int(retry.Seconds())+1))
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":"rate limit exceeded"}`))
			return
		}
		next.ServeHTTP(w, r)
	})
}

func clientAddr(r *http.Request) string {
	if r == nil {
		return "unknown"
	}
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		// X-Forwarded-For may list multiple hops; take the first one.
		for i := 0; i < len(xff); i++ {
			if xff[i] == ',' {
				return xff[:i]
			}
		}
		return xff
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}
