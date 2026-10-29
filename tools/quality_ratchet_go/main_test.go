// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — native command publication and refusal tests
// Package main_test exercises complete command output and real operating-system failures.
package main_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// nativeReport decodes the published command protocol independently of private Go types.
type nativeReport struct {
	// Schema identifies the native parser protocol.
	Schema string `json:"schema"`
	// Sources preserves all original command-input digests.
	Sources map[string]string `json:"sources"`
	// Findings retains original physical diagnostics.
	Findings []struct {
		// Path identifies the original input.
		Path string `json:"path"`
		// Line identifies the physical source line.
		Line int `json:"line"`
		// Column identifies the physical source byte column.
		Column int `json:"column"`
		// Message explains the actual source finding.
		Message string `json:"message"`
	} `json:"findings"`
}

// nativeCommand builds the actual command without intercepting any native tool.
func nativeCommand(t *testing.T) string {
	t.Helper()
	binary := filepath.Join(t.TempDir(), "documentation")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, "go", "build", "-buildvcs=false", "-cover", "-covermode=atomic", "-o", binary, ".")
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("native build failed: %v\n%s", err, output)
	}
	return binary
}

// executeNative observes an actual command status, bytes and optional real output file.
func executeNative(t *testing.T, binary string, args []string, sink *os.File) (int, []byte, string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, binary, args...)
	coverage := os.Getenv("DIRECTOR_QUALITY_GO_COVERDIR")
	if coverage == "" {
		coverage = t.TempDir()
	}
	command.Env = append(os.Environ(), "GOCOVERDIR="+coverage)
	var output, stderr bytes.Buffer
	command.Stdout = &output
	if sink != nil {
		command.Stdout = sink
	}
	command.Stderr = &stderr
	err := command.Run()
	var exited *exec.ExitError
	if err != nil && !errors.As(err, &exited) {
		t.Fatal(err)
	}
	return command.ProcessState.ExitCode(), output.Bytes(), stderr.String()
}

// TestRunPublishesCompleteOriginalSources checks native status and original source custody.
func TestRunPublishesCompleteOriginalSources(t *testing.T) {
	binary := nativeCommand(t)
	clean := sourceFile(t, "package fixture\n// Value returns one.\nfunc Value() int {return 1}\n")
	dirty := sourceFile(t, "package fixture\nfunc TestValue() {}\n")
	for _, test := range []struct {
		paths    []string
		status   int
		findings int
	}{
		{[]string{clean}, 0, 0},
		{[]string{"--", clean, dirty}, 1, 1},
	} {
		status, output, stderr := executeNative(t, binary, test.paths, nil)
		if status != test.status || stderr != "" {
			t.Fatalf("status=%d errors=%q", status, stderr)
		}
		var original nativeReport
		if err := json.Unmarshal(output, &original); err != nil {
			t.Fatal(err)
		}
		if original.Schema != "quality-ratchet.go-ast.v1" || len(original.Findings) != test.findings {
			t.Fatalf("incomplete native carrier: %+v", original)
		}
		if original.Sources[clean] == "" {
			t.Fatal("original clean source is absent")
		}
	}
}

// TestRunRefusesWholePublication prevents duplicates, failed input and broken sinks.
func TestRunRefusesWholePublication(t *testing.T) {
	binary := nativeCommand(t)
	clean := sourceFile(t, "package fixture\n")
	absent := filepath.Join(t.TempDir(), "absent.go")
	for _, paths := range [][]string{nil, {"--"}, {clean, clean}, {clean, absent}} {
		status, output, stderr := executeNative(t, binary, paths, nil)
		if status != 2 || len(output) != 0 || stderr == "" {
			t.Fatalf("incomplete analysis was published: status=%d output=%q errors=%q", status, output, stderr)
		}
	}
	sink, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatal(err)
	}
	status, _, stderr := executeNative(t, binary, []string{clean}, sink)
	if err := sink.Close(); err != nil {
		t.Fatal(err)
	}
	if status != 2 || stderr == "" {
		t.Fatal("broken output was qualified")
	}
	if err := os.Remove(clean); err != nil {
		t.Fatal(err)
	}
}
