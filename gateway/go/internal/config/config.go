// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — gateway configuration loader

// Package config resolves gateway settings from the environment. Kept
// deliberately flat: each field has one env var, one flag override,
// and an explicit default. No YAML, no TOML, no struct tags.
package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config is the fully resolved gateway configuration.
type Config struct {
	ListenAddr        string
	UpstreamURL       string
	UpstreamTimeout   time.Duration
	APIKeys           []string
	AuditSalt         []byte
	AuditLogPath      string
	RateLimitRPM      int
	RateLimitBurst    int
	AllowHTTPUpstream bool
	ScoringAddr       string
	ScoringTimeout    time.Duration
}

const (
	defaultListenAddr      = ":8080"
	defaultUpstreamURL     = "https://api.openai.com"
	defaultUpstreamTimeout = 30 * time.Second
	defaultRateLimitRPM    = 600
	defaultRateLimitBurst  = 60
	defaultScoringTimeout  = 2 * time.Second
	legacyAuditSalt        = "director-ai-audit-v1"
)

// Load reads the configuration from environment variables and returns
// an error if anything is malformed. Unset values fall back to
// safe defaults; empty "DIRECTOR_API_KEYS" means "no auth" (dev mode
// only) and is accepted but logged by the caller.
func Load() (*Config, error) {
	cfg := &Config{
		ListenAddr:        envOr("DIRECTOR_LISTEN_ADDR", defaultListenAddr),
		UpstreamURL:       envOr("DIRECTOR_UPSTREAM_URL", defaultUpstreamURL),
		UpstreamTimeout:   defaultUpstreamTimeout,
		RateLimitRPM:      defaultRateLimitRPM,
		RateLimitBurst:    defaultRateLimitBurst,
		AllowHTTPUpstream: envBool("DIRECTOR_ALLOW_HTTP_UPSTREAM"),
		AuditLogPath:      os.Getenv("DIRECTOR_AUDIT_LOG"),
		ScoringAddr:       os.Getenv("DIRECTOR_SCORING_ADDR"),
		ScoringTimeout:    defaultScoringTimeout,
	}

	if raw := os.Getenv("DIRECTOR_SCORING_TIMEOUT_MS"); raw != "" {
		ms, err := strconv.Atoi(raw)
		if err != nil || ms <= 0 {
			return nil, fmt.Errorf("DIRECTOR_SCORING_TIMEOUT_MS: %q", raw)
		}
		cfg.ScoringTimeout = time.Duration(ms) * time.Millisecond
	}

	if raw := os.Getenv("DIRECTOR_UPSTREAM_TIMEOUT_SECONDS"); raw != "" {
		secs, err := strconv.Atoi(raw)
		if err != nil || secs <= 0 {
			return nil, fmt.Errorf("DIRECTOR_UPSTREAM_TIMEOUT_SECONDS: %q", raw)
		}
		cfg.UpstreamTimeout = time.Duration(secs) * time.Second
	}

	if raw := os.Getenv("DIRECTOR_RATE_LIMIT_RPM"); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n < 0 {
			return nil, fmt.Errorf("DIRECTOR_RATE_LIMIT_RPM: %q", raw)
		}
		cfg.RateLimitRPM = n
	}

	if raw := os.Getenv("DIRECTOR_RATE_LIMIT_BURST"); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("DIRECTOR_RATE_LIMIT_BURST: %q", raw)
		}
		cfg.RateLimitBurst = n
	}

	if raw := os.Getenv("DIRECTOR_API_KEYS"); raw != "" {
		cfg.APIKeys = splitKeys(raw)
	}

	salt := resolveAuditSalt()
	if len(salt) == 0 {
		return nil, errors.New("audit salt resolved to empty value")
	}
	cfg.AuditSalt = salt

	if strings.HasPrefix(cfg.UpstreamURL, "http://") && !cfg.AllowHTTPUpstream {
		return nil, fmt.Errorf(
			"non-HTTPS upstream %q — set DIRECTOR_ALLOW_HTTP_UPSTREAM=1 to override",
			cfg.UpstreamURL,
		)
	}

	return cfg, nil
}

func splitKeys(raw string) []string {
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if trimmed := strings.TrimSpace(p); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func resolveAuditSalt() []byte {
	if explicit := os.Getenv("DIRECTOR_AUDIT_SALT"); explicit != "" {
		return []byte(explicit)
	}
	if path := os.Getenv("DIRECTOR_AUDIT_SALT_FILE"); path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		trimmed := strings.TrimSpace(string(data))
		if trimmed == "" {
			return nil
		}
		return []byte(trimmed)
	}
	return []byte(legacyAuditSalt)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envBool(key string) bool {
	switch strings.ToLower(os.Getenv(key)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}
