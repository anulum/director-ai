// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — gateway HTTP server assembly

// Package server wires the gateway middleware chain: request-ID →
// audit → auth → rate limit → proxy. The chain is deliberately short;
// every piece can be tested in isolation.
package server

import (
	"net/http"
	"time"

	"github.com/anulum/director-ai/gateway/internal/audit"
	"github.com/anulum/director-ai/gateway/internal/auth"
	"github.com/anulum/director-ai/gateway/internal/config"
	"github.com/anulum/director-ai/gateway/internal/proxy"
	"github.com/anulum/director-ai/gateway/internal/ratelimit"
	"github.com/anulum/director-ai/gateway/internal/risk"
	"github.com/anulum/director-ai/gateway/internal/scoring"
)

// Components bundles the wired middleware so callers can assemble a
// server once and reuse it for tests.
type Components struct {
	Config  *config.Config
	Auth    *auth.Middleware
	Rate    *ratelimit.Limiter
	Proxy   *proxy.Client
	Audit   *audit.Logger
	Scoring *scoring.Client
	Risk    *risk.Middleware
	Handler http.Handler
}

// Build builds the middleware chain from a resolved config.
// ``auditWriter``, when non-nil, overrides the file sink — useful in
// tests that capture log output.
func Build(cfg *config.Config, auditLogger *audit.Logger) (*Components, error) {
	prx, err := proxy.New(cfg.UpstreamURL, cfg.UpstreamTimeout)
	if err != nil {
		return nil, err
	}
	authMW := auth.New(cfg.APIKeys, cfg.AuditSalt)
	rate := ratelimit.New(cfg.RateLimitRPM, cfg.RateLimitBurst)

	var scoringClient *scoring.Client
	var scoringMW *scoring.Middleware
	if cfg.ScoringAddr != "" {
		scoringClient, err = scoring.Dial(cfg.ScoringAddr, cfg.ScoringTimeout)
		if err != nil {
			return nil, err
		}
		scoringMW = &scoring.Middleware{
			Scorer:          scoringClient,
			Timeout:         cfg.ScoringTimeout,
			ThresholdHeader: "X-Coherence-Threshold",
		}
	}

	var riskMW *risk.Middleware
	if cfg.RiskEnabled {
		budget, err := risk.NewBudget(
			cfg.RiskAllowance, cfg.RiskWindowSeconds, nil,
		)
		if err != nil {
			return nil, err
		}
		riskMW = risk.NewMiddleware(risk.NewScorer(), budget)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", health)
	mux.HandleFunc("/healthz", health)
	mux.HandleFunc("/ready", health)
	mux.Handle("/v1/", http.HandlerFunc(prx.Forward))

	// Middleware order: outer → inner. Request flows from the
	// outermost wrapper inward; the audit middleware must run after
	// auth so it can read the fingerprint the latter stamps onto the
	// request context, but it must wrap the rate limiter so 429s get
	// logged too. Scoring, when enabled, wraps the proxy so it sees
	// the assistant message after the upstream returns. Risk runs
	// between auth and rate so obvious attacks are rejected before
	// the token bucket or the upstream call, and the budget
	// reservation key is the authenticated fingerprint. Final chain:
	//   requestID → auth → audit → risk → rate → scoring → mux
	var handler http.Handler = mux
	if scoringMW.Enabled() {
		handler = scoringMW.Handler(handler)
	}
	handler = rate.Handler(handler, auth.FingerprintFromContext)
	if riskMW.Enabled() {
		handler = riskMW.Handler(handler)
	}
	handler = auditMiddleware(auditLogger, cfg.UpstreamURL)(handler)
	handler = authMW.Handler(handler)
	handler = requestIDMiddleware(handler)

	return &Components{
		Config:  cfg,
		Auth:    authMW,
		Rate:    rate,
		Proxy:   prx,
		Audit:   auditLogger,
		Scoring: scoringClient,
		Risk:    riskMW,
		Handler: handler,
	}, nil
}

// Run starts an HTTP server on the configured listen address and
// blocks until the server exits.
func Run(cfg *config.Config, auditLogger *audit.Logger) error {
	c, err := Build(cfg, auditLogger)
	if err != nil {
		return err
	}
	srv := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           c.Handler,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       60 * time.Second,
		WriteTimeout:      0, // zero = streaming responses may run long
		IdleTimeout:       120 * time.Second,
	}
	return srv.ListenAndServe()
}

func health(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}
