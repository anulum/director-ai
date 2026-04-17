// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — risk middleware for the gateway

package risk

import (
	"encoding/json"
	"io"
	"net/http"
	"strconv"

	"github.com/anulum/director-ai/gateway/internal/auth"
)

// Middleware refuses obvious attack prompts and throttles tenants
// whose sliding-window risk budget is exhausted. The decision is
// stamped onto response headers so downstream auditors see why a
// request was accepted or rejected:
//
//   - X-Risk-Score     — combined risk in [0, 1]
//   - X-Risk-Backend   — chosen scorer backend (rules/embed/nli)
//   - X-Risk-Action    — allow / reject
//   - X-Risk-Reason    — human-readable reason
//   - X-Risk-Remaining — remaining budget for the tenant
//
// The middleware extracts the prompt from the request body when the
// payload is a JSON object with a ``messages`` array (OpenAI chat)
// or a ``prompt`` field (legacy completion). If the body is not
// JSON or the prompt cannot be found, the middleware passes the
// request through untouched — failing-open preserves compatibility
// with clients that use an unexpected wire shape.
type Middleware struct {
	Scorer            *Scorer
	Budget            *Budget
	RulesThreshold    float64
	EmbedThreshold    float64
	RejectThreshold   float64
	TenantFromRequest func(*http.Request) string
}

// Defaults mirror ``director_ai.core.routing.RiskRouter``.
const (
	DefaultRulesThreshold  = 0.2
	DefaultEmbedThreshold  = 0.55
	DefaultRejectThreshold = 0.92
)

// NewMiddleware returns a wired middleware with sane defaults. The
// tenant resolver falls back to ``auth.FingerprintFromContext`` so a
// gateway with API-key auth binds risk to the key's fingerprint;
// unauthenticated deployments pass empty tenant IDs, which is
// still acceptable because the budget is per-key.
func NewMiddleware(scorer *Scorer, budget *Budget) *Middleware {
	return &Middleware{
		Scorer:            scorer,
		Budget:            budget,
		RulesThreshold:    DefaultRulesThreshold,
		EmbedThreshold:    DefaultEmbedThreshold,
		RejectThreshold:   DefaultRejectThreshold,
		TenantFromRequest: auth.FingerprintFromContext,
	}
}

// Handler wraps next with risk scoring. Enabled reports whether
// the middleware will do anything; callers that did not configure
// a scorer should skip the wrap entirely.
func (m *Middleware) Enabled() bool {
	return m != nil && m.Scorer != nil && m.Budget != nil
}

// Handler returns the net/http-compatible handler.
func (m *Middleware) Handler(next http.Handler) http.Handler {
	if !m.Enabled() {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		prompt := m.extractPrompt(r)
		if prompt == "" {
			// Could not find a prompt — pass through. The downstream
			// scoring middleware will catch a malformed body.
			next.ServeHTTP(w, r)
			return
		}
		components := m.Scorer.Score(prompt)
		tenant := ""
		if m.TenantFromRequest != nil {
			tenant = m.TenantFromRequest(r)
		}
		backend := m.selectBackend(components.Combined)
		w.Header().Set("X-Risk-Backend", backend)
		w.Header().Set(
			"X-Risk-Score",
			strconv.FormatFloat(components.Combined, 'f', 4, 64),
		)

		if components.Combined >= m.RejectThreshold {
			entry := m.Budget.Snapshot(tenant)
			w.Header().Set("X-Risk-Action", "reject")
			w.Header().Set(
				"X-Risk-Reason",
				"risk "+strconv.FormatFloat(components.Combined, 'f', 3, 64)+
					" >= reject_threshold",
			)
			w.Header().Set(
				"X-Risk-Remaining",
				strconv.FormatFloat(entry.Remaining, 'f', 3, 64),
			)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnprocessableEntity)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"error": map[string]any{
					"type":    "risk_reject",
					"message": "prompt risk above reject threshold",
					"score":   components.Combined,
				},
			})
			return
		}
		entry := m.Budget.Reserve(tenant, components.Combined)
		w.Header().Set(
			"X-Risk-Remaining",
			strconv.FormatFloat(entry.Remaining, 'f', 3, 64),
		)
		if entry.Exhausted() {
			w.Header().Set("X-Risk-Action", "reject")
			w.Header().Set("X-Risk-Reason", "budget exhausted")
			w.Header().Set("Retry-After", strconv.Itoa(int(entry.WindowSeconds)+1))
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"error": map[string]any{
					"type":    "risk_budget_exhausted",
					"message": "tenant risk budget exhausted for window",
				},
			})
			return
		}
		w.Header().Set("X-Risk-Action", "allow")
		w.Header().Set("X-Risk-Reason", "within band")
		next.ServeHTTP(w, r)
	})
}

func (m *Middleware) selectBackend(risk float64) string {
	if risk < m.RulesThreshold {
		return "rules"
	}
	if risk < m.EmbedThreshold {
		return "embed"
	}
	return "nli"
}

type chatCompletionEnvelope struct {
	Prompt   string `json:"prompt,omitempty"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages,omitempty"`
}

// extractPrompt peels the user prompt from the body without
// consuming the caller's request — the body is replaced with a
// buffer over the original bytes so downstream handlers still see
// the full payload.
func (m *Middleware) extractPrompt(r *http.Request) string {
	if r.Body == nil {
		return ""
	}
	buf, err := io.ReadAll(r.Body)
	if err != nil {
		return ""
	}
	_ = r.Body.Close()
	r.Body = _replayReader(buf)
	if len(buf) == 0 {
		return ""
	}
	var env chatCompletionEnvelope
	if err := json.Unmarshal(buf, &env); err != nil {
		return ""
	}
	if env.Prompt != "" {
		return env.Prompt
	}
	// Prefer the last user message — that's the request text.
	for i := len(env.Messages) - 1; i >= 0; i-- {
		msg := env.Messages[i]
		if msg.Role == "user" && msg.Content != "" {
			return msg.Content
		}
	}
	// Fallback: the first non-empty message.
	for _, msg := range env.Messages {
		if msg.Content != "" {
			return msg.Content
		}
	}
	return ""
}
