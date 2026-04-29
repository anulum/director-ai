// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI - Go proto property contract tests

package directorv1

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"google.golang.org/protobuf/proto"
)

func TestGeneratedCoherenceVerdictProperties(t *testing.T) {
	rng := rand.New(rand.NewSource(12029))
	opts := proto.MarshalOptions{Deterministic: true}
	reasons := []HaltReason{
		HaltReason_HALT_REASON_NONE,
		HaltReason_HALT_REASON_COHERENCE_BELOW_THRESHOLD,
		HaltReason_HALT_REASON_INJECTION_DETECTED,
		HaltReason_HALT_REASON_POLICY_VIOLATION,
		HaltReason_HALT_REASON_TOKEN_TIMEOUT,
		HaltReason_HALT_REASON_TOTAL_TIMEOUT,
		HaltReason_HALT_REASON_CALLBACK_TIMEOUT,
	}

	for i := 0; i < 256; i++ {
		sourceCount := rng.Intn(5)
		sources := make([]*GroundingSource, 0, sourceCount)
		for j := 0; j < sourceCount; j++ {
			sources = append(sources, &GroundingSource{
				SourceId:   fmt.Sprintf("kb:%03d:%02d", i, j),
				Similarity: rng.Float32(),
				NliSupport: rng.Float32(),
			})
		}

		verdict := &CoherenceVerdict{
			Score:      rng.Float32(),
			Halted:     rng.Intn(2) == 0,
			HaltReason: reasons[rng.Intn(len(reasons))],
			HardLimit:  rng.Float32(),
			ScoreLower: rng.Float32(),
			ScoreUpper: rng.Float32(),
			Sources:    sources,
			Message:    fmt.Sprintf("case-%03d", i),
		}

		first, err := opts.Marshal(verdict)
		if err != nil {
			t.Fatalf("Marshal first case %d: %v", i, err)
		}
		second, err := opts.Marshal(verdict)
		if err != nil {
			t.Fatalf("Marshal second case %d: %v", i, err)
		}
		if !bytes.Equal(first, second) {
			t.Fatalf("case %d deterministic bytes differ", i)
		}

		restored := &CoherenceVerdict{}
		if err := proto.Unmarshal(first, restored); err != nil {
			t.Fatalf("Unmarshal case %d: %v", i, err)
		}
		assertClose32(t, restored.GetScore(), verdict.GetScore(), "Score", i)
		assertClose32(t, restored.GetHardLimit(), verdict.GetHardLimit(), "HardLimit", i)
		if restored.GetHalted() != verdict.GetHalted() {
			t.Fatalf("case %d Halted = %v; want %v", i, restored.GetHalted(), verdict.GetHalted())
		}
		if restored.GetHaltReason() != verdict.GetHaltReason() {
			t.Fatalf("case %d HaltReason = %v; want %v", i, restored.GetHaltReason(), verdict.GetHaltReason())
		}
		if len(restored.GetSources()) != len(verdict.GetSources()) {
			t.Fatalf("case %d source count = %d; want %d", i, len(restored.GetSources()), len(verdict.GetSources()))
		}
		for j := range restored.GetSources() {
			assertClose32(t, restored.GetSources()[j].GetSimilarity(), verdict.GetSources()[j].GetSimilarity(), "Similarity", i)
			assertClose32(t, restored.GetSources()[j].GetNliSupport(), verdict.GetSources()[j].GetNliSupport(), "NliSupport", i)
		}
	}
}

func TestGeneratedChatCompletionRequestProperties(t *testing.T) {
	rng := rand.New(rand.NewSource(42023))
	roles := []Role{
		Role_ROLE_SYSTEM,
		Role_ROLE_USER,
		Role_ROLE_ASSISTANT,
		Role_ROLE_TOOL,
	}

	for i := 0; i < 160; i++ {
		messageCount := 1 + rng.Intn(8)
		messages := make([]*ChatMessage, 0, messageCount)
		for j := 0; j < messageCount; j++ {
			messages = append(messages, &ChatMessage{
				Role:    roles[rng.Intn(len(roles))],
				Content: fmt.Sprintf("message-%03d-%02d", i, j),
				Name:    fmt.Sprintf("speaker-%02d", j),
			})
		}

		request := &ChatCompletionRequest{
			Model:       fmt.Sprintf("model-%02d", rng.Intn(9)),
			Messages:    messages,
			Temperature: rng.Float32(),
			MaxTokens:   int32(rng.Intn(4096)),
			Stream:      rng.Intn(2) == 0,
			TenantId:    fmt.Sprintf("tenant-%02d", rng.Intn(17)),
			RequestId:   fmt.Sprintf("req-%03d", i),
		}

		wire, err := proto.Marshal(request)
		if err != nil {
			t.Fatalf("Marshal case %d: %v", i, err)
		}
		restored := &ChatCompletionRequest{}
		if err := proto.Unmarshal(wire, restored); err != nil {
			t.Fatalf("Unmarshal case %d: %v", i, err)
		}
		if restored.GetRequestId() != request.GetRequestId() {
			t.Fatalf("case %d RequestId = %q; want %q", i, restored.GetRequestId(), request.GetRequestId())
		}
		if len(restored.GetMessages()) != len(request.GetMessages()) {
			t.Fatalf("case %d message count = %d; want %d", i, len(restored.GetMessages()), len(request.GetMessages()))
		}
		for j := range restored.GetMessages() {
			if restored.GetMessages()[j].GetRole() != request.GetMessages()[j].GetRole() {
				t.Fatalf("case %d message %d Role = %v; want %v", i, j, restored.GetMessages()[j].GetRole(), request.GetMessages()[j].GetRole())
			}
			if restored.GetMessages()[j].GetContent() != request.GetMessages()[j].GetContent() {
				t.Fatalf("case %d message %d Content = %q; want %q", i, j, restored.GetMessages()[j].GetContent(), request.GetMessages()[j].GetContent())
			}
		}
	}
}

func TestGeneratedSafetyEventProperties(t *testing.T) {
	rng := rand.New(rand.NewSource(62026))
	decisions := []PolicyDecision{
		PolicyDecision_POLICY_DECISION_ALLOW,
		PolicyDecision_POLICY_DECISION_WARN,
		PolicyDecision_POLICY_DECISION_HALT,
		PolicyDecision_POLICY_DECISION_BLOCK,
	}
	reasons := []HaltReason{
		HaltReason_HALT_REASON_NONE,
		HaltReason_HALT_REASON_COHERENCE_BELOW_THRESHOLD,
		HaltReason_HALT_REASON_POLICY_VIOLATION,
	}

	for i := 0; i < 160; i++ {
		event := &SafetyEvent{
			SchemaVersion:         "director.safety_event.v1",
			EventId:               fmt.Sprintf("sevt-%03d", i),
			Timestamp:             fmt.Sprintf("2026-04-29T22:%02d:00Z", i%60),
			RequestId:             fmt.Sprintf("req-%03d", i),
			TenantId:              fmt.Sprintf("tenant-%02d", rng.Intn(11)),
			HookId:                fmt.Sprintf("hook-%02d", rng.Intn(7)),
			HookScope:             "streaming",
			PolicyDecision:        decisions[rng.Intn(len(decisions))],
			HaltReason:            reasons[rng.Intn(len(reasons))],
			Threshold:             rng.Float32(),
			ObservedScore:         rng.Float32(),
			LatencyMs:             int64(rng.Intn(1_000_000)),
			EvidenceRefs:          []string{fmt.Sprintf("kb:%03d", i)},
			TenantSafeExplanation: fmt.Sprintf("event-%03d", i),
			Attributes:            map[string]string{"case": fmt.Sprintf("%03d", i)},
		}

		wire, err := proto.Marshal(event)
		if err != nil {
			t.Fatalf("Marshal case %d: %v", i, err)
		}
		restored := &SafetyEvent{}
		if err := proto.Unmarshal(wire, restored); err != nil {
			t.Fatalf("Unmarshal case %d: %v", i, err)
		}
		if restored.GetEventId() != event.GetEventId() {
			t.Fatalf("case %d EventId = %q; want %q", i, restored.GetEventId(), event.GetEventId())
		}
		if restored.GetPolicyDecision() != event.GetPolicyDecision() {
			t.Fatalf("case %d PolicyDecision = %v; want %v", i, restored.GetPolicyDecision(), event.GetPolicyDecision())
		}
		if restored.GetHaltReason() != event.GetHaltReason() {
			t.Fatalf("case %d HaltReason = %v; want %v", i, restored.GetHaltReason(), event.GetHaltReason())
		}
		assertClose32(t, restored.GetThreshold(), event.GetThreshold(), "Threshold", i)
		assertClose32(t, restored.GetObservedScore(), event.GetObservedScore(), "ObservedScore", i)
		if restored.GetAttributes()["case"] != event.GetAttributes()["case"] {
			t.Fatalf("case %d attributes mismatch", i)
		}
	}
}

func assertClose32(t *testing.T, got float32, want float32, field string, caseID int) {
	t.Helper()
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("case %d %s = %v; want %v", caseID, field, got, want)
	}
}
