// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — gateway audit log sink (JSONL)

// Package audit serialises one AuditRecord per HTTP response to a
// JSONL file or stdout. Matches the shape of
// ``director_ai.core.safety.audit.AuditLogger`` on the Python side so
// downstream consumers see one unified log regardless of which
// front door handled the request.
package audit

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

// Record is the wire shape. Field order kept stable so line-by-line
// diffs stay readable.
type Record struct {
	Timestamp         string   `json:"timestamp"`
	RequestID         string   `json:"request_id"`
	APIKeyFingerprint string   `json:"api_key_fingerprint"`
	Method            string   `json:"method"`
	Path              string   `json:"path"`
	Status            int      `json:"status"`
	LatencyMS         int64    `json:"latency_ms"`
	BytesIn           int64    `json:"bytes_in"`
	BytesOut          int64    `json:"bytes_out"`
	Upstream          string   `json:"upstream,omitempty"`
	PolicyViolations  []string `json:"policy_violations,omitempty"`
}

// Logger writes AuditRecords to one io.Writer under a mutex.
type Logger struct {
	w  io.Writer
	mu sync.Mutex
}

// New returns a Logger writing to ``w``. If ``w`` is nil the logger
// writes to os.Stdout.
func New(w io.Writer) *Logger {
	if w == nil {
		w = os.Stdout
	}
	return &Logger{w: w}
}

// NewFile opens ``path`` in append mode and returns a Logger backed
// by it. The file is created with 0o600 — audit logs may contain
// tenant fingerprints and must not be world-readable.
func NewFile(path string) (*Logger, error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o600)
	if err != nil {
		return nil, fmt.Errorf("audit log open: %w", err)
	}
	return New(f), nil
}

// Log writes one record. Timestamp is filled automatically when empty.
func (l *Logger) Log(rec *Record) error {
	if rec.Timestamp == "" {
		rec.Timestamp = time.Now().UTC().Format(time.RFC3339Nano)
	}
	data, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("audit marshal: %w", err)
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, err := l.w.Write(data); err != nil {
		return err
	}
	_, err = l.w.Write([]byte("\n"))
	return err
}

// Close releases the underlying writer if it implements io.Closer.
func (l *Logger) Close() error {
	if c, ok := l.w.(io.Closer); ok {
		return c.Close()
	}
	return nil
}
