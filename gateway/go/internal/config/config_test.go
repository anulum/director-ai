// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — config resolver tests

package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// isolateEnv removes every DIRECTOR_* variable so one test's leaks do
// not contaminate another's expectations. The Go test runner reuses
// a process, so os.Unsetenv is the only safe reset.
func isolateEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		"DIRECTOR_LISTEN_ADDR",
		"DIRECTOR_UPSTREAM_URL",
		"DIRECTOR_UPSTREAM_TIMEOUT_SECONDS",
		"DIRECTOR_RATE_LIMIT_RPM",
		"DIRECTOR_RATE_LIMIT_BURST",
		"DIRECTOR_API_KEYS",
		"DIRECTOR_AUDIT_SALT",
		"DIRECTOR_AUDIT_SALT_FILE",
		"DIRECTOR_AUDIT_LOG",
		"DIRECTOR_ALLOW_HTTP_UPSTREAM",
	} {
		t.Setenv(k, "")
		os.Unsetenv(k)
	}
}

func TestLoad_Defaults(t *testing.T) {
	isolateEnv(t)
	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.ListenAddr != defaultListenAddr {
		t.Errorf("ListenAddr = %q; want %q", cfg.ListenAddr, defaultListenAddr)
	}
	if cfg.UpstreamURL != defaultUpstreamURL {
		t.Errorf("UpstreamURL = %q; want %q", cfg.UpstreamURL, defaultUpstreamURL)
	}
	if cfg.UpstreamTimeout != defaultUpstreamTimeout {
		t.Errorf("UpstreamTimeout = %v; want %v", cfg.UpstreamTimeout, defaultUpstreamTimeout)
	}
	if cfg.RateLimitRPM != defaultRateLimitRPM {
		t.Errorf("RateLimitRPM = %d; want %d", cfg.RateLimitRPM, defaultRateLimitRPM)
	}
	if len(cfg.APIKeys) != 0 {
		t.Errorf("expected no API keys by default; got %v", cfg.APIKeys)
	}
	if string(cfg.AuditSalt) != legacyAuditSalt {
		t.Errorf("AuditSalt = %q; want legacy default", cfg.AuditSalt)
	}
}

func TestLoad_EnvOverrides(t *testing.T) {
	isolateEnv(t)
	t.Setenv("DIRECTOR_LISTEN_ADDR", ":9090")
	t.Setenv("DIRECTOR_UPSTREAM_URL", "https://api.anthropic.com")
	t.Setenv("DIRECTOR_UPSTREAM_TIMEOUT_SECONDS", "15")
	t.Setenv("DIRECTOR_RATE_LIMIT_RPM", "1200")
	t.Setenv("DIRECTOR_RATE_LIMIT_BURST", "200")
	t.Setenv("DIRECTOR_API_KEYS", "sk-a, sk-b ,,sk-c")
	t.Setenv("DIRECTOR_AUDIT_SALT", "deployment-xyz")
	t.Setenv("DIRECTOR_AUDIT_LOG", "/tmp/audit.jsonl")
	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.ListenAddr != ":9090" {
		t.Errorf("ListenAddr = %q", cfg.ListenAddr)
	}
	if cfg.UpstreamURL != "https://api.anthropic.com" {
		t.Errorf("UpstreamURL = %q", cfg.UpstreamURL)
	}
	if cfg.UpstreamTimeout != 15*time.Second {
		t.Errorf("UpstreamTimeout = %v", cfg.UpstreamTimeout)
	}
	if cfg.RateLimitRPM != 1200 {
		t.Errorf("RateLimitRPM = %d", cfg.RateLimitRPM)
	}
	if cfg.RateLimitBurst != 200 {
		t.Errorf("RateLimitBurst = %d", cfg.RateLimitBurst)
	}
	want := []string{"sk-a", "sk-b", "sk-c"}
	if len(cfg.APIKeys) != 3 {
		t.Fatalf("APIKeys = %v; want %v", cfg.APIKeys, want)
	}
	for i, k := range want {
		if cfg.APIKeys[i] != k {
			t.Errorf("APIKeys[%d] = %q; want %q", i, cfg.APIKeys[i], k)
		}
	}
	if string(cfg.AuditSalt) != "deployment-xyz" {
		t.Errorf("AuditSalt = %q", cfg.AuditSalt)
	}
	if cfg.AuditLogPath != "/tmp/audit.jsonl" {
		t.Errorf("AuditLogPath = %q", cfg.AuditLogPath)
	}
}

func TestLoad_AuditSaltFromFile(t *testing.T) {
	isolateEnv(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "salt")
	if err := os.WriteFile(path, []byte("file-salt-value\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("DIRECTOR_AUDIT_SALT_FILE", path)
	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(cfg.AuditSalt) != "file-salt-value" {
		t.Errorf("AuditSalt = %q", cfg.AuditSalt)
	}
}

func TestLoad_InvalidTimeoutRejected(t *testing.T) {
	isolateEnv(t)
	t.Setenv("DIRECTOR_UPSTREAM_TIMEOUT_SECONDS", "not-a-number")
	if _, err := Load(); err == nil {
		t.Fatal("expected error for invalid timeout")
	}
}

func TestLoad_HTTPUpstreamRequiresOverride(t *testing.T) {
	isolateEnv(t)
	t.Setenv("DIRECTOR_UPSTREAM_URL", "http://upstream.example")
	if _, err := Load(); err == nil {
		t.Fatal("expected error for non-HTTPS upstream without override")
	}
	t.Setenv("DIRECTOR_ALLOW_HTTP_UPSTREAM", "1")
	if _, err := Load(); err != nil {
		t.Errorf("expected success with override: %v", err)
	}
}

func TestLoad_InvalidRateLimitRejected(t *testing.T) {
	isolateEnv(t)
	t.Setenv("DIRECTOR_RATE_LIMIT_RPM", "-1")
	if _, err := Load(); err == nil {
		t.Fatal("expected error for negative RPM")
	}
}

func TestSplitKeys_Trims(t *testing.T) {
	got := splitKeys("  a  ,b,, c ,")
	want := []string{"a", "b", "c"}
	if len(got) != len(want) {
		t.Fatalf("got %v; want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] = %q; want %q", i, got[i], want[i])
		}
	}
}
