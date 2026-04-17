// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — audit sink tests

package audit

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestLog_WritesJSONL(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	rec := &Record{
		Timestamp:  "2026-04-17T04:00:00Z",
		RequestID:  "req-1",
		Method:     "POST",
		Path:       "/v1/chat/completions",
		Status:     200,
		LatencyMS:  42,
		BytesIn:    100,
		BytesOut:   500,
		Upstream:   "https://api.openai.com",
	}
	if err := l.Log(rec); err != nil {
		t.Fatalf("Log: %v", err)
	}
	lines := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected 1 line; got %d", len(lines))
	}
	var decoded Record
	if err := json.Unmarshal([]byte(lines[0]), &decoded); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if decoded.Status != 200 || decoded.LatencyMS != 42 {
		t.Errorf("round-trip mismatch: %+v", decoded)
	}
}

func TestLog_AutoTimestamp(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	if err := l.Log(&Record{RequestID: "r"}); err != nil {
		t.Fatalf("Log: %v", err)
	}
	var decoded Record
	if err := json.Unmarshal(bytes.TrimRight(buf.Bytes(), "\n"), &decoded); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if decoded.Timestamp == "" {
		t.Error("Timestamp not set automatically")
	}
	if _, err := time.Parse(time.RFC3339Nano, decoded.Timestamp); err != nil {
		t.Errorf("Timestamp not RFC3339: %q", decoded.Timestamp)
	}
}

func TestLog_ConcurrentSafe(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	var wg sync.WaitGroup
	const n = 100
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = l.Log(&Record{RequestID: "r", Status: 200})
		}()
	}
	wg.Wait()
	lines := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	if len(lines) != n {
		t.Errorf("expected %d lines; got %d", n, len(lines))
	}
}

func TestNewFile_CreatesFileWithPerms(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "audit.jsonl")
	l, err := NewFile(path)
	if err != nil {
		t.Fatalf("NewFile: %v", err)
	}
	defer l.Close()
	if err := l.Log(&Record{RequestID: "r"}); err != nil {
		t.Fatalf("Log: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("perm = %o; want 0o600", perm)
	}
}

func TestNewFile_AppendsAcrossReopen(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "audit.jsonl")
	l1, err := NewFile(path)
	if err != nil {
		t.Fatalf("NewFile 1: %v", err)
	}
	_ = l1.Log(&Record{RequestID: "first"})
	_ = l1.Close()
	l2, err := NewFile(path)
	if err != nil {
		t.Fatalf("NewFile 2: %v", err)
	}
	_ = l2.Log(&Record{RequestID: "second"})
	_ = l2.Close()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines; got %d (%q)", len(lines), data)
	}
}

func TestNewFile_ErrorOnUnwritable(t *testing.T) {
	if _, err := NewFile("/does/not/exist/audit.jsonl"); err == nil {
		t.Error("expected error on unwritable path")
	}
}

func TestNew_NilWriterFallsBackToStdout(t *testing.T) {
	l := New(nil)
	if l.w == nil {
		t.Error("writer should not be nil")
	}
}
