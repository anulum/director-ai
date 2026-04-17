// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — scoring middleware tests with a stub scorer

package scoring

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	directorv1 "github.com/anulum/director-ai/gateway/proto/director/v1"
)

type stubScorer struct {
	verdict   *directorv1.CoherenceVerdict
	err       error
	lastClaim string
}

func (s *stubScorer) ScoreClaim(
	_ context.Context, claim string, _ []string, _, _ string, _ float32,
) (*directorv1.CoherenceVerdict, int64, error) {
	s.lastClaim = claim
	return s.verdict, 3, s.err
}

func chatResponse(w http.ResponseWriter, status int, content string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":      "chatcmpl-1",
		"object":  "chat.completion",
		"choices": []map[string]any{{"index": 0, "message": map[string]string{"role": "assistant", "content": content}, "finish_reason": "stop"}},
	})
}

func TestHandler_PassesWhenNotHalted(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		chatResponse(w, 200, "Paris is the capital of France.")
	})
	scorer := &stubScorer{verdict: &directorv1.CoherenceVerdict{Score: 0.88, Halted: false}}
	mw := (&Middleware{Scorer: scorer}).Handler(upstream)
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("{}"))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mw.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("status = %d; want 200", w.Code)
	}
	if got := w.Header().Get("X-Coherence-Score"); got != "0.8800" {
		t.Errorf("X-Coherence-Score = %q", got)
	}
	if got := w.Header().Get("X-Coherence-Halted"); got != "false" {
		t.Errorf("X-Coherence-Halted = %q", got)
	}
	if !strings.Contains(w.Body.String(), "Paris is the capital") {
		t.Errorf("body should be forwarded unchanged; got %q", w.Body.String())
	}
	if scorer.lastClaim != "Paris is the capital of France." {
		t.Errorf("lastClaim = %q", scorer.lastClaim)
	}
}

func TestHandler_Halts422WhenVerdictHalted(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		chatResponse(w, 200, "Fictional claim.")
	})
	scorer := &stubScorer{verdict: &directorv1.CoherenceVerdict{
		Score:      0.11,
		Halted:     true,
		HaltReason: directorv1.HaltReason_HALT_REASON_COHERENCE_BELOW_THRESHOLD,
		Message:    "below threshold",
	}}
	mw := (&Middleware{Scorer: scorer}).Handler(upstream)
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mw.ServeHTTP(w, r)
	if w.Code != http.StatusUnprocessableEntity {
		t.Errorf("status = %d; want 422", w.Code)
	}
	if got := w.Header().Get("X-Coherence-Halted"); got != "true" {
		t.Errorf("X-Coherence-Halted = %q", got)
	}
	var payload map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
		t.Fatalf("body JSON: %v", err)
	}
	errBlob, ok := payload["error"].(map[string]any)
	if !ok {
		t.Fatalf("error payload = %v", payload)
	}
	if errBlob["type"] != "hallucination_detected" {
		t.Errorf("type = %v", errBlob["type"])
	}
}

func TestHandler_PassesThroughOnScorerError(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		chatResponse(w, 200, "ok")
	})
	scorer := &stubScorer{err: errors.New("network blip")}
	mw := (&Middleware{Scorer: scorer}).Handler(upstream)
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	mw.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("status = %d", w.Code)
	}
	if got := w.Header().Get("X-Coherence-Error"); got == "" {
		t.Errorf("X-Coherence-Error missing")
	}
}

func TestHandler_BypassesStreamingRequests(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: one\n\n"))
	})
	scorer := &stubScorer{verdict: &directorv1.CoherenceVerdict{Score: 0.9, Halted: false}}
	mw := (&Middleware{Scorer: scorer}).Handler(upstream)
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("Accept", "text/event-stream")
	w := httptest.NewRecorder()
	mw.ServeHTTP(w, r)
	if scorer.lastClaim != "" {
		t.Errorf("scoring should be skipped for SSE; claim=%q", scorer.lastClaim)
	}
}

func TestHandler_PassesThrough5xx(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(`{"error":"upstream"}`))
	})
	scorer := &stubScorer{}
	mw := (&Middleware{Scorer: scorer}).Handler(upstream)
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	mw.ServeHTTP(w, r)
	if w.Code != http.StatusBadGateway {
		t.Errorf("status = %d", w.Code)
	}
	if scorer.lastClaim != "" {
		t.Errorf("scoring should skip non-2xx; claim=%q", scorer.lastClaim)
	}
}

func TestHandler_SkipsNonJSONContentType(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(200)
		_, _ = w.Write([]byte("plain text"))
	})
	scorer := &stubScorer{}
	mw := (&Middleware{Scorer: scorer}).Handler(upstream)
	r := httptest.NewRequest("POST", "/anywhere", nil)
	w := httptest.NewRecorder()
	mw.ServeHTTP(w, r)
	if scorer.lastClaim != "" {
		t.Errorf("scoring should skip non-JSON body; claim=%q", scorer.lastClaim)
	}
	if w.Body.String() != "plain text" {
		t.Errorf("body = %q", w.Body.String())
	}
}

func TestHandler_ParsesThresholdHeader(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		chatResponse(w, 200, "Some claim.")
	})
	captured := struct {
		threshold float32
	}{}
	scorer := &thresholdRecorder{verdict: &directorv1.CoherenceVerdict{Score: 0.9}, captured: &captured}
	mw := (&Middleware{Scorer: scorer, ThresholdHeader: "X-Coherence-Threshold"}).Handler(upstream)
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("X-Coherence-Threshold", "0.72")
	w := httptest.NewRecorder()
	mw.ServeHTTP(w, r)
	if captured.threshold != 0.72 {
		t.Errorf("threshold = %v", captured.threshold)
	}
}

func TestEnabled(t *testing.T) {
	var nilM *Middleware
	if nilM.Enabled() {
		t.Error("nil middleware should report disabled")
	}
	if (&Middleware{}).Enabled() {
		t.Error("middleware without scorer should report disabled")
	}
	if !(&Middleware{Scorer: &stubScorer{}}).Enabled() {
		t.Error("middleware with scorer should report enabled")
	}
}

func TestHandler_NilMiddlewareIsPassthrough(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(204)
	})
	h := (&Middleware{}).Handler(upstream) // Scorer nil
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 204 {
		t.Errorf("passthrough expected; got %d", w.Code)
	}
}

// thresholdRecorder captures the threshold passed through so we can
// assert parseThresholdHeader() wiring.
type thresholdRecorder struct {
	verdict  *directorv1.CoherenceVerdict
	captured *struct{ threshold float32 }
}

func (r *thresholdRecorder) ScoreClaim(
	_ context.Context, _ string, _ []string, _, _ string, threshold float32,
) (*directorv1.CoherenceVerdict, int64, error) {
	r.captured.threshold = threshold
	return r.verdict, 1, nil
}
