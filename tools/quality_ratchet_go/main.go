// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — native Go documentation command
// Package main complements Revive with declaration coverage and suppression checks.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// report retains each original source digest and parser-derived diagnostic.
type report struct {
	// Schema identifies the complete parser report format.
	Schema string `json:"schema"`
	// Sources maps each explicit input to its original byte digest.
	Sources map[string]string `json:"sources"`
	// Findings retains all declaration and source-directive diagnostics.
	Findings []finding `json:"findings"`
}

// run analyses all explicit inputs before publishing one complete native report.
func run(args []string, output, errors io.Writer) int {
	if len(args) > 0 && args[0] == "--" {
		args = args[1:]
	}
	if len(args) == 0 {
		fmt.Fprintln(errors, "Native Go documentation requires explicit source inputs")
		return 2
	}
	result := report{"quality-ratchet.go-ast.v1", map[string]string{}, []finding{}}
	for _, path := range args {
		if _, duplicate := result.Sources[path]; duplicate {
			fmt.Fprintln(errors, "Repeated native Go source input")
			return 2
		}
		digest, findings, err := analyseFile(path)
		if err != nil {
			fmt.Fprintln(errors, "Native Go parser could not completely analyse source")
			return 2
		}
		result.Sources[path] = digest
		result.Findings = append(result.Findings, findings...)
	}
	if err := json.NewEncoder(output).Encode(result); err != nil {
		fmt.Fprintln(errors, "Native Go report could not be written")
		return 2
	}
	if len(result.Findings) > 0 {
		return 1
	}
	return 0
}

// main exposes the finite native command with its original process status.
func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}
