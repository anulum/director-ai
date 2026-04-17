// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — gateway API-key authentication

// Package auth validates API keys in exactly the same shape as the
// Python middleware (``Authorization: Bearer`` or ``X-API-Key``) and
// produces audit fingerprints that match ``audit_salt.get_audit_salt``
// on the Python side. Any fingerprint persisted by the Go gateway is
// interchangeable with one persisted by the Python server.
package auth

import (
	"crypto/hmac"
	"crypto/sha512"
	"encoding/hex"
	"net/http"
	"strings"
	"sync"
)

// Context key type — unexported so callers cannot forge values.
type ctxKey int

const (
	// KeyFingerprint is the context key under which a successfully
	// authenticated request carries its key fingerprint. Downstream
	// middleware (rate limit, audit) reads it with
	// ``r.Context().Value(KeyFingerprint)``.
	KeyFingerprint ctxKey = iota
	// KeyAuthenticated flags whether the request cleared auth. Always
	// true when set; absence means unauthenticated.
	KeyAuthenticated
)

// ExemptPaths matches the Python server list. Health and metrics
// endpoints must stay reachable without a key.
var exemptPaths = map[string]struct{}{
	"/":        {},
	"/health":  {},
	"/healthz": {},
	"/ready":   {},
	"/metrics": {},
}

// Middleware validates keys against ``keys``. When ``keys`` is empty
// the middleware is a no-op (dev mode) — the caller is responsible
// for logging that choice.
type Middleware struct {
	keys       [][]byte
	auditSalt  []byte
	keysLock   sync.RWMutex
}

// New constructs a middleware from the decoded keys and audit salt.
// ``keys`` may be empty (no-auth mode); passing ``nil`` is equivalent.
func New(keys []string, auditSalt []byte) *Middleware {
	m := &Middleware{auditSalt: auditSalt}
	m.SetKeys(keys)
	return m
}

// SetKeys replaces the valid key list atomically. Safe for concurrent
// readers running through Handler.
func (m *Middleware) SetKeys(keys []string) {
	buf := make([][]byte, 0, len(keys))
	for _, k := range keys {
		if trimmed := strings.TrimSpace(k); trimmed != "" {
			buf = append(buf, []byte(trimmed))
		}
	}
	m.keysLock.Lock()
	m.keys = buf
	m.keysLock.Unlock()
}

// Fingerprint returns the truncated salted SHA-512 HMAC of ``key``,
// matching ``director_ai.middleware.api_key._hash_key`` on the Python
// side (16 hex chars).
func (m *Middleware) Fingerprint(key string) string {
	mac := hmac.New(sha512.New, m.auditSalt)
	mac.Write([]byte(key))
	return hex.EncodeToString(mac.Sum(nil))[:16]
}

// Handler wraps ``next`` with API-key validation. Exempt paths are
// forwarded unchanged. Invalid or missing keys return 401 JSON.
func (m *Middleware) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, ok := exemptPaths[r.URL.Path]; ok {
			next.ServeHTTP(w, r)
			return
		}
		m.keysLock.RLock()
		hasKeys := len(m.keys) > 0
		m.keysLock.RUnlock()
		if !hasKeys {
			next.ServeHTTP(w, r)
			return
		}

		provided := extractKey(r)
		if provided == "" || !m.validate(provided) {
			reject(w, "invalid or missing API key")
			return
		}

		fp := m.Fingerprint(provided)
		ctx := r.Context()
		ctx = withFingerprint(ctx, fp)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (m *Middleware) validate(provided string) bool {
	candidate := []byte(provided)
	m.keysLock.RLock()
	defer m.keysLock.RUnlock()
	for _, valid := range m.keys {
		if hmac.Equal(candidate, valid) {
			return true
		}
	}
	return false
}

func extractKey(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if strings.HasPrefix(strings.ToLower(auth), "bearer ") {
		return strings.TrimSpace(auth[len("bearer "):])
	}
	return strings.TrimSpace(r.Header.Get("X-API-Key"))
}

func reject(w http.ResponseWriter, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnauthorized)
	_, _ = w.Write([]byte(`{"error":"` + msg + `"}`))
}

// FingerprintFromContext returns the audit fingerprint attached by a
// prior Handler run, or the empty string if no authentication
// occurred on this request.
func FingerprintFromContext(r *http.Request) string {
	if fp, ok := r.Context().Value(KeyFingerprint).(string); ok {
		return fp
	}
	return ""
}
