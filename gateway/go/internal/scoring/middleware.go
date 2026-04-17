// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — scoring response middleware

package scoring

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	directorv1 "github.com/anulum/director-ai/gateway/proto/director/v1"
)

// Scorer is the subset of Client methods the middleware depends on.
// Extracted so tests can inject a stub without wiring gRPC.
type Scorer interface {
	ScoreClaim(
		ctx context.Context,
		claim string,
		documents []string,
		tenantID, requestID string,
		threshold float32,
	) (*directorv1.CoherenceVerdict, int64, error)
}

// Middleware runs ScoreClaim against the assistant response once the
// upstream handler returns. The verdict is attached to the response
// headers; requests with ``Accept: text/event-stream`` bypass scoring
// because SSE output is not buffered here (streaming mode is a
// future-phase concern).
type Middleware struct {
	Scorer  Scorer
	Timeout time.Duration
	// ThresholdHeader, when set, lets clients tune the threshold per
	// request via an HTTP header (e.g. "X-Coherence-Threshold"). An
	// empty string keeps the server default.
	ThresholdHeader string
}

// Handler wraps next with post-response scoring. The middleware
// buffers non-streaming responses so it can read the assistant
// message, calls ScoreClaim with it, and either:
//
//   - adds ``X-Coherence-Score`` and ``X-Coherence-Halted`` headers
//     and forwards the body unchanged (halted=false), or
//   - rewrites the response as 422 JSON (halted=true, default),
//     signalling a hallucination.
//
// A zero-value Middleware returns next unchanged. Use ``Enabled`` to
// check that a scorer was supplied before wiring.
func (m *Middleware) Handler(next http.Handler) http.Handler {
	if !m.Enabled() {
		return next
	}
	timeout := m.Timeout
	if timeout <= 0 {
		timeout = 2 * time.Second
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if wantsStream(r) {
			next.ServeHTTP(w, r)
			return
		}
		rec := &bufferRecorder{header: http.Header{}}
		next.ServeHTTP(rec, r)
		// Surface the upstream body verbatim whenever we cannot
		// inspect it — e.g. an error status or an unexpected
		// content type.
		if rec.status >= 400 || !looksLikeChatCompletion(rec.header) {
			rec.flushTo(w)
			return
		}
		claim := extractAssistantContent(rec.body.Bytes())
		if claim == "" {
			rec.flushTo(w)
			return
		}
		threshold := parseThresholdHeader(r, m.ThresholdHeader)
		ctx, cancel := context.WithTimeout(r.Context(), timeout)
		defer cancel()
		verdict, _, err := m.Scorer.ScoreClaim(
			ctx,
			claim,
			nil,
			r.Header.Get("X-Tenant-ID"),
			r.Header.Get("X-Request-ID"),
			threshold,
		)
		if err != nil {
			// Scoring is an optional augmentation. If it fails the
			// gateway still forwards the response; clients relying
			// on halt behaviour must observe ``X-Coherence-Error``.
			w.Header().Set("X-Coherence-Error", truncate(err.Error(), 200))
			rec.flushTo(w)
			return
		}
		w.Header().Set(
			"X-Coherence-Score",
			strconv.FormatFloat(float64(verdict.GetScore()), 'f', 4, 64),
		)
		if verdict.GetHalted() {
			w.Header().Set("X-Coherence-Halted", "true")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnprocessableEntity)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"error": map[string]any{
					"type":    "hallucination_detected",
					"message": firstNonEmpty(verdict.GetMessage(), "coherence below threshold"),
					"score":   verdict.GetScore(),
				},
			})
			return
		}
		w.Header().Set("X-Coherence-Halted", "false")
		rec.flushTo(w)
	})
}

// Enabled reports whether the middleware is wired to a real scorer.
func (m *Middleware) Enabled() bool {
	return m != nil && m.Scorer != nil
}

// bufferRecorder captures the full response so the middleware can
// re-emit it conditionally. The recorder only buffers when the
// upstream produces a bounded body; streaming requests take a
// separate path.
type bufferRecorder struct {
	header http.Header
	body   bytes.Buffer
	status int
}

func (r *bufferRecorder) Header() http.Header { return r.header }

func (r *bufferRecorder) WriteHeader(code int) {
	if r.status == 0 {
		r.status = code
	}
}

func (r *bufferRecorder) Write(p []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	return r.body.Write(p)
}

func (r *bufferRecorder) flushTo(w http.ResponseWriter) {
	for k, values := range r.header {
		for _, v := range values {
			w.Header().Add(k, v)
		}
	}
	if r.status == 0 {
		r.status = http.StatusOK
	}
	w.WriteHeader(r.status)
	_, _ = w.Write(r.body.Bytes())
}

func wantsStream(r *http.Request) bool {
	if r.Method != http.MethodPost {
		return false
	}
	accept := r.Header.Get("Accept")
	return strings.Contains(strings.ToLower(accept), "text/event-stream")
}

func looksLikeChatCompletion(header http.Header) bool {
	return strings.Contains(strings.ToLower(header.Get("Content-Type")), "application/json")
}

// chatCompletion is a minimal subset of the OpenAI response shape —
// only the fields we need to pull the assistant text out.
type chatCompletion struct {
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func extractAssistantContent(body []byte) string {
	var payload chatCompletion
	if err := json.Unmarshal(body, &payload); err != nil {
		return ""
	}
	for _, c := range payload.Choices {
		if c.Message.Role == "assistant" && c.Message.Content != "" {
			return c.Message.Content
		}
	}
	return ""
}

func parseThresholdHeader(r *http.Request, headerName string) float32 {
	if headerName == "" {
		return 0
	}
	raw := r.Header.Get(headerName)
	if raw == "" {
		return 0
	}
	f, err := strconv.ParseFloat(raw, 32)
	if err != nil || f <= 0 || f >= 1 {
		return 0
	}
	return float32(f)
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
