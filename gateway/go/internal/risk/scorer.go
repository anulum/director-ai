// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — Go-native prompt risk scorer

// Package risk mirrors the Python PromptRiskScorer heuristic so the
// gateway can refuse obvious attacks before any Python RPC is
// issued. The implementation is deliberately identical to
// ``director_ai/core/routing/scorer.py``'s length / structural /
// marker heuristic so traffic gets the same verdict regardless of
// which tier handles it.
//
// The gateway never sees the sanitiser or injection signals — those
// require the Python model to be loaded — so this Go path is
// heuristic-only. Callers that want the full blend must reach back
// into the Python ``PromptRiskScorer`` over gRPC; this package
// covers the cheap first line of defence.
package risk

import (
	"regexp"
	"strings"
)

// Components mirrors the Python ``RiskComponents`` dataclass. Only
// the heuristic channel is populated here; ``Sanitiser`` and
// ``Injection`` stay at zero.
type Components struct {
	Heuristic float64
	Sanitiser float64
	Injection float64
	Combined  float64
}

// Scorer produces a ``[0, 1]`` risk score from a prompt. Thread
// safe — the regex slices are never mutated after construction.
type Scorer struct {
	markers           []*regexp.Regexp
	structuralChars   string
	maxSafeLength     int
	weights           [3]float64
}

// systemStyleMarkers mirrors ``_SYSTEM_STYLE_MARKERS`` in the Python
// scorer.
var systemStyleMarkers = []string{
	`(?i)\bignore (?:all |the |your |previous )`,
	`(?i)\bSYSTEM\s*:`,
	`(?i)\[\s*system\s*\]`,
	`(?i)(?:you are|act as) (?:a|an) (?:admin|root|developer)`,
	`(?i)\bdelimiter\s+collision\b`,
}

const structuralChars = "[]{}<>|`"

// NewScorer builds a Scorer with the defaults that match the Python
// side: length saturates at 8000 characters, weights sum to 1.
func NewScorer() *Scorer {
	return newScorerWithParams(8000, [3]float64{0.15, 0.5, 0.35})
}

// NewScorerWithMaxLength lets callers tune the length-saturation
// threshold; the weights stay at the Python defaults.
func NewScorerWithMaxLength(maxSafeLength int) (*Scorer, error) {
	if maxSafeLength <= 0 {
		return nil, &invalidParamError{field: "maxSafeLength", value: maxSafeLength}
	}
	return newScorerWithParams(maxSafeLength, [3]float64{0.15, 0.5, 0.35}), nil
}

func newScorerWithParams(maxSafeLength int, weights [3]float64) *Scorer {
	regs := make([]*regexp.Regexp, 0, len(systemStyleMarkers))
	for _, pattern := range systemStyleMarkers {
		// patterns are literals compiled at init; a bad literal
		// would be a bug, not a runtime failure — MustCompile is
		// the right call.
		regs = append(regs, regexp.MustCompile(pattern))
	}
	return &Scorer{
		markers:         regs,
		structuralChars: structuralChars,
		maxSafeLength:   maxSafeLength,
		weights:         weights,
	}
}

// Score returns a ``Components`` record for ``prompt``. Mirrors
// :meth:`PromptRiskScorer.score` — empty or whitespace-only
// prompts return all zeros.
func (s *Scorer) Score(prompt string) Components {
	trimmed := strings.TrimSpace(prompt)
	if trimmed == "" {
		return Components{}
	}
	h := s.heuristic(prompt)
	wH, wS, wI := s.weights[0], s.weights[1], s.weights[2]
	linear := h*wH + 0*wS + 0*wI
	combined := h
	if linear > combined {
		combined = linear
	}
	if combined < 0.0 {
		combined = 0.0
	} else if combined > 1.0 {
		combined = 1.0
	}
	return Components{Heuristic: h, Combined: combined}
}

func (s *Scorer) heuristic(prompt string) float64 {
	length := len(prompt)
	lengthRatio := float64(length) / float64(s.maxSafeLength)
	if lengthRatio > 1.0 {
		lengthRatio = 1.0
	}
	structural := 0
	for _, ch := range prompt {
		if strings.ContainsRune(s.structuralChars, ch) {
			structural++
		}
	}
	density := float64(structural) / float64(maxInt(length, 1))
	structuralRisk := density * 40.0
	if structuralRisk > 1.0 {
		structuralRisk = 1.0
	}
	markerHits := 0
	for _, re := range s.markers {
		if re.MatchString(prompt) {
			markerHits++
		}
	}
	markerRisk := float64(markerHits) * 0.35
	if markerRisk > 1.0 {
		markerRisk = 1.0
	}
	risk := 0.4 * lengthRatio
	if v := 0.7 * structuralRisk; v > risk {
		risk = v
	}
	if markerRisk > risk {
		risk = markerRisk
	}
	if risk > 1.0 {
		risk = 1.0
	}
	return risk
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

type invalidParamError struct {
	field string
	value any
}

func (e *invalidParamError) Error() string {
	return "risk: invalid " + e.field
}
