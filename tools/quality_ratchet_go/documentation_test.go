// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — real native declaration and directive contract tests
// Package main_test verifies the compiled command against original source files.
package main_test

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// sourceFile writes an original Go source in the native runner's isolated workspace.
func sourceFile(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "original.go")
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestDeclarationDocumentation covers real Go declaration syntax and explanatory comments.
func TestDeclarationDocumentation(t *testing.T) {
	binary := nativeCommand(t)
	cases := []struct {
		name     string
		source   string
		findings int
	}{
		{"documented-function", "// Package fixture holds a value.\npackage fixture\n// value returns one.\nfunc value() int { return 1 }\n", 0},
		{"private-function", "package fixture\nfunc value() int { return 1 }\n", 1},
		{"test-entry", "package fixture\nfunc TestValue() {}\n", 1},
		{"main-entry", "package main\nfunc main() {}\n", 1},
		{"method", "package fixture\n// item stores a value.\ntype item int\nfunc (i item) Value() int {return int(i)}\n", 1},
		{"type", "package fixture\ntype Value int\n", 1},
		{"group-type-doc", "package fixture\ntype (\n// Value stores a count.\nValue int\n)\n", 0},
		{"group-doc", "package fixture\n// Values store counts.\ntype ( Value int; Other int )\n", 0},
		{"values", "package fixture\nvar a, b int\nconst C=1\n", 3},
		{"blank-value", "package fixture\nvar _ = 1\n", 0},
		{"documented-values", "package fixture\n// a stores the first count.\nvar a int\nconst (\n// C stores a constant count.\nC=1\n)\n", 0},
		{"imports", "package fixture\nimport \"fmt\"\n", 0},
		{"struct", "package fixture\n// Item stores named values.\ntype Item struct { Value int; hidden int; *Embedded }\n", 1},
		{"struct-doc", "package fixture\n// Item stores a count.\ntype Item struct {\n// Value is its count.\nValue int\nOther int // Other is another count.\n}\n", 0},
		{"interface", "package fixture\n// Reader reads values.\ntype Reader interface { Read(); close(); io.Reader }\n", 1},
		{"interface-doc", "package fixture\n// Reader reads values.\ntype Reader interface {\n// Read fetches a value.\nRead()\n}\n", 0},
		{"directive-only", "package fixture\n//go:noinline\nfunc value() {}\n", 1},
		{"directive-with-doc", "package fixture\n// value stores a count.\n//go:noinline\nfunc value() {}\n", 0},
		{"closure", "package fixture\n// callback returns the current count.\nvar callback=func() int{return 1}\n", 0},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			path := sourceFile(t, test.source)
			status, output, stderr := executeNative(t, binary, []string{path}, nil)
			expectedStatus := 0
			if test.findings > 0 {
				expectedStatus = 1
			}
			if status != expectedStatus || stderr != "" {
				t.Fatalf("status=%d errors=%q", status, stderr)
			}
			var original nativeReport
			if err := json.Unmarshal(output, &original); err != nil {
				t.Fatal(err)
			}
			if original.Sources[path] != fmt.Sprintf("%x", sha256.Sum256([]byte(test.source))) {
				t.Fatal("native digest differs from original source bytes")
			}
			if len(original.Findings) != test.findings {
				t.Fatalf("findings=%+v, expected count=%d", original.Findings, test.findings)
			}
			for _, finding := range original.Findings {
				if finding.Path != path || finding.Line < 1 || finding.Column < 1 {
					t.Fatalf("diagnostic lost physical source binding: %+v", finding)
				}
			}
		})
	}
}

// TestDirectivesRetainPhysicalLocations rejects masking while ignoring strings and prose.
func TestDirectivesRetainPhysicalLocations(t *testing.T) {
	binary := nativeCommand(t)
	source := "package fixture\n//revive:disable:exported\n//line invented.go:300\n/*line another.go:700*/\nfunc missing() {}\n// text stores literal directive content.\nvar text = \"//revive:disable\"\n// ordinary is harmless explanatory prose.\n// line this is a sentence\nvar ordinary = 1\n"
	path := sourceFile(t, source)
	status, output, stderr := executeNative(t, binary, []string{path}, nil)
	if status != 1 || stderr != "" {
		t.Fatalf("status=%d errors=%q", status, stderr)
	}
	var original nativeReport
	if err := json.Unmarshal(output, &original); err != nil {
		t.Fatal(err)
	}
	findings := original.Findings
	if len(findings) != 4 {
		t.Fatalf("unexpected real source diagnostics: %+v", findings)
	}
	for index, expected := range []int{2, 3, 4, 5} {
		if findings[index].Line != expected || findings[index].Path != path {
			t.Fatalf("adjusted or lost physical position: %+v", findings[index])
		}
	}
	if !strings.Contains(findings[3].Message, "missing") {
		t.Fatal("missing declaration was suppressed")
	}
}

// TestParserRefusesMissingAndMalformedOriginals keeps incomplete native analysis explicit.
func TestParserRefusesMissingAndMalformedOriginals(t *testing.T) {
	binary := nativeCommand(t)
	for _, path := range []string{
		filepath.Join(t.TempDir(), "absent.go"),
		sourceFile(t, "package fixture\nfunc invalid( {\n"),
	} {
		status, output, stderr := executeNative(t, binary, []string{path}, nil)
		if status != 2 || len(output) != 0 || stderr == "" {
			t.Fatalf("incomplete native source was accepted: status=%d output=%q errors=%q", status, output, stderr)
		}
	}
}
