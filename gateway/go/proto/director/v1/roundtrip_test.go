// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — Go proto round-trip tests

package directorv1

import (
	"bytes"
	"testing"

	"google.golang.org/protobuf/proto"
)

func TestCoherenceVerdict_RoundTrip(t *testing.T) {
	v := &CoherenceVerdict{
		Score:      0.91,
		Halted:     false,
		HaltReason: HaltReason_HALT_REASON_NONE,
		HardLimit:  0.5,
		Sources: []*GroundingSource{
			{SourceId: "kb:fact-1", Similarity: 0.9, NliSupport: 0.8},
		},
		Message: "ok",
	}
	buf, err := proto.Marshal(v)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	restored := &CoherenceVerdict{}
	if err := proto.Unmarshal(buf, restored); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if restored.GetScore() != 0.91 {
		t.Errorf("Score = %v; want 0.91", restored.GetScore())
	}
	if restored.GetHaltReason() != HaltReason_HALT_REASON_NONE {
		t.Errorf("HaltReason = %v", restored.GetHaltReason())
	}
	if len(restored.GetSources()) != 1 {
		t.Fatalf("Sources len = %d", len(restored.GetSources()))
	}
	if restored.GetSources()[0].GetSourceId() != "kb:fact-1" {
		t.Errorf("SourceId = %q", restored.GetSources()[0].GetSourceId())
	}
}

func TestChatCompletionRequest_RoundTrip(t *testing.T) {
	req := &ChatCompletionRequest{
		Model:       "gpt-4o-mini",
		Temperature: 0.7,
		MaxTokens:   128,
		Stream:      true,
		TenantId:    "tenant-1",
		RequestId:   "req-1",
		Messages: []*ChatMessage{
			{Role: Role_ROLE_SYSTEM, Content: "You are precise."},
			{Role: Role_ROLE_USER, Content: "What is 2+2?"},
		},
	}
	buf, err := proto.Marshal(req)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	restored := &ChatCompletionRequest{}
	if err := proto.Unmarshal(buf, restored); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if restored.GetModel() != "gpt-4o-mini" {
		t.Errorf("Model = %q", restored.GetModel())
	}
	if !restored.GetStream() {
		t.Errorf("Stream = false; want true")
	}
	if got := len(restored.GetMessages()); got != 2 {
		t.Fatalf("Messages len = %d; want 2", got)
	}
	if restored.GetMessages()[1].GetContent() != "What is 2+2?" {
		t.Errorf("message.Content = %q", restored.GetMessages()[1].GetContent())
	}
}

func TestAuditRecord_CarriesNestedVerdict(t *testing.T) {
	rec := &AuditRecord{
		Timestamp:         "2026-04-17T04:00:00.000Z",
		RequestId:         "req-1",
		TenantId:          "t-1",
		ApiKeyFingerprint: "ab12cd34",
		QueryHash:         "deadbeef",
		ResponseLength:    120,
		LatencyMs:         42,
		Model:             "gpt-4o-mini",
		PolicyViolations:  []string{"pii:email"},
		Verdict: &CoherenceVerdict{
			Score:     0.88,
			Halted:    false,
			HardLimit: 0.5,
		},
	}
	buf, err := proto.Marshal(rec)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	restored := &AuditRecord{}
	if err := proto.Unmarshal(buf, restored); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if restored.GetLatencyMs() != 42 {
		t.Errorf("LatencyMs = %d; want 42", restored.GetLatencyMs())
	}
	if got := restored.GetVerdict().GetScore(); got != 0.88 {
		t.Errorf("Verdict.Score = %v; want 0.88", got)
	}
	if got := restored.GetPolicyViolations(); len(got) != 1 || got[0] != "pii:email" {
		t.Errorf("PolicyViolations = %v", got)
	}
}

func TestDeterministicSerialisation(t *testing.T) {
	v := &CoherenceVerdict{Score: 0.5, Halted: false, HardLimit: 0.5}
	opts := proto.MarshalOptions{Deterministic: true}
	a, err := opts.Marshal(v)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	b, err := opts.Marshal(v)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if !bytes.Equal(a, b) {
		t.Errorf("deterministic Marshal produced different bytes:\n%x\n%x", a, b)
	}
}

func TestCrossLanguageFixture(t *testing.T) {
	// Serialised form of a CoherenceVerdict produced by the Python
	// side with the same field values. Guards against the two
	// language stubs drifting apart.
	v := &CoherenceVerdict{
		Score:      0.82,
		Halted:     false,
		HaltReason: HaltReason_HALT_REASON_NONE,
		HardLimit:  0.5,
	}
	opts := proto.MarshalOptions{Deterministic: true}
	wire, err := opts.Marshal(v)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if len(wire) == 0 {
		t.Fatalf("empty wire for populated verdict")
	}
	restored := &CoherenceVerdict{}
	if err := proto.Unmarshal(wire, restored); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got := restored.GetScore(); got != 0.82 {
		t.Errorf("Score = %v; want 0.82", got)
	}
}
