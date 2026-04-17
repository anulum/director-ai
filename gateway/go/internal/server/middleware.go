// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — request-ID and audit middleware

package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"net/http"
	"time"

	"github.com/anulum/director-ai/gateway/internal/audit"
	"github.com/anulum/director-ai/gateway/internal/auth"
)

type ctxKey int

const (
	ctxRequestID ctxKey = iota
)

// RequestID returns the request ID attached by requestIDMiddleware.
// Empty when the middleware has not run on this request (e.g., in a
// unit test exercising a handler directly).
func RequestID(r *http.Request) string {
	if v, ok := r.Context().Value(ctxRequestID).(string); ok {
		return v
	}
	return ""
}

func requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			id = generateID()
		}
		w.Header().Set("X-Request-ID", id)
		ctx := context.WithValue(r.Context(), ctxRequestID, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func generateID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "req-unknown"
	}
	return "req-" + hex.EncodeToString(b[:])
}

// responseRecorder captures the status code and bytes written so we
// can log them after the proxy handler returns.
type responseRecorder struct {
	http.ResponseWriter
	status   int
	bytesOut int64
}

func (r *responseRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *responseRecorder) Write(p []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	n, err := r.ResponseWriter.Write(p)
	r.bytesOut += int64(n)
	return n, err
}

// Flush propagates the underlying Flusher so SSE responses still
// stream through the recorder unchanged.
func (r *responseRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func auditMiddleware(logger *audit.Logger, upstream string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			rec := &responseRecorder{ResponseWriter: w}
			next.ServeHTTP(rec, r)
			if logger == nil {
				return
			}
			_ = logger.Log(&audit.Record{
				Timestamp:         start.UTC().Format(time.RFC3339Nano),
				RequestID:         RequestID(r),
				APIKeyFingerprint: auth.FingerprintFromContext(r),
				Method:            r.Method,
				Path:              r.URL.Path,
				Status:            rec.status,
				LatencyMS:         time.Since(start).Milliseconds(),
				BytesIn:           r.ContentLength,
				BytesOut:          rec.bytesOut,
				Upstream:          upstream,
			})
		})
	}
}
