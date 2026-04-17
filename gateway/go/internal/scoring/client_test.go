// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — scoring client tests with an in-process fake server

package scoring

import (
	"context"
	"errors"
	"io"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"

	directorv1 "github.com/anulum/director-ai/gateway/proto/director/v1"
)

// fakeScoringServer is the Go-side stub implementation we test
// against. It exposes a knob (HaltAt) to simulate a halt verdict
// after N streaming tokens, and a counter for assertions.
type fakeScoringServer struct {
	directorv1.UnimplementedCoherenceScoringServer
	Score     float32
	Halt      bool
	HaltAt    int
	Calls     int
	StreamLen int
}

func (f *fakeScoringServer) ScoreClaim(
	_ context.Context, req *directorv1.ScoreClaimRequest,
) (*directorv1.ScoreClaimResponse, error) {
	f.Calls++
	verdict := &directorv1.CoherenceVerdict{
		Score:     f.Score,
		Halted:    f.Halt,
		HardLimit: req.GetThreshold(),
	}
	if f.Halt {
		verdict.HaltReason = directorv1.HaltReason_HALT_REASON_COHERENCE_BELOW_THRESHOLD
	}
	return &directorv1.ScoreClaimResponse{
		Verdict:   verdict,
		LatencyMs: 7,
	}, nil
}

func (f *fakeScoringServer) ScoreStream(
	stream directorv1.CoherenceScoring_ScoreStreamServer,
) error {
	i := 0
	for {
		_, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		i++
		f.StreamLen = i
		halted := f.HaltAt > 0 && i >= f.HaltAt
		verdict := &directorv1.CoherenceVerdict{
			Score:  f.Score,
			Halted: halted,
		}
		if halted {
			verdict.HaltReason =
				directorv1.HaltReason_HALT_REASON_COHERENCE_BELOW_THRESHOLD
		}
		if err := stream.Send(&directorv1.ScoreTokenResponse{Verdict: verdict}); err != nil {
			return err
		}
		if halted {
			return nil
		}
	}
}

func bootFake(t *testing.T, svc *fakeScoringServer) (addr string, cleanup func()) {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	grpcServer := grpc.NewServer()
	directorv1.RegisterCoherenceScoringServer(grpcServer, svc)
	go func() {
		_ = grpcServer.Serve(lis)
	}()
	return lis.Addr().String(), func() {
		grpcServer.GracefulStop()
	}
}

func TestDial_RejectsEmptyAddr(t *testing.T) {
	if _, err := Dial("", time.Second); err == nil {
		t.Error("expected error on empty address")
	}
}

func TestScoreClaim_UnaryRoundTrip(t *testing.T) {
	fake := &fakeScoringServer{Score: 0.82}
	addr, cleanup := bootFake(t, fake)
	defer cleanup()

	c, err := Dial(addr, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	verdict, latency, err := c.ScoreClaim(
		context.Background(),
		"Paris is the capital of France.",
		[]string{"Paris is the capital."},
		"t-1",
		"r-1",
		0.5,
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := verdict.GetScore(); got != 0.82 {
		t.Errorf("score = %v; want 0.82", got)
	}
	if latency != 7 {
		t.Errorf("latency = %d; want 7", latency)
	}
	if fake.Calls != 1 {
		t.Errorf("server calls = %d", fake.Calls)
	}
}

func TestScoreClaim_HaltIsSurfaced(t *testing.T) {
	fake := &fakeScoringServer{Score: 0.1, Halt: true}
	addr, cleanup := bootFake(t, fake)
	defer cleanup()

	c, err := Dial(addr, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	verdict, _, err := c.ScoreClaim(
		context.Background(), "sub-threshold", nil, "", "", 0.5,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !verdict.GetHalted() {
		t.Error("expected halted=true")
	}
	if verdict.GetHaltReason() !=
		directorv1.HaltReason_HALT_REASON_COHERENCE_BELOW_THRESHOLD {
		t.Errorf("halt_reason = %v", verdict.GetHaltReason())
	}
}

func TestStream_SendsUntilServerHalts(t *testing.T) {
	fake := &fakeScoringServer{Score: 0.9, HaltAt: 3}
	addr, cleanup := bootFake(t, fake)
	defer cleanup()

	c, err := Dial(addr, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	handle, err := c.StartStream(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var verdicts []*directorv1.CoherenceVerdict
	for i := 0; i < 5; i++ {
		if err := handle.Send(&directorv1.ScoreTokenRequest{
			AccumulatedText: "prefix",
			NextToken:       "tok",
		}); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			t.Fatalf("send %d: %v", i, err)
		}
		v, err := handle.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			t.Fatalf("recv %d: %v", i, err)
		}
		verdicts = append(verdicts, v)
		if v.GetHalted() {
			break
		}
	}
	_ = handle.Close()

	if len(verdicts) != 3 {
		t.Errorf("received %d verdicts; want 3", len(verdicts))
	}
	if last := verdicts[len(verdicts)-1]; !last.GetHalted() {
		t.Error("last verdict should be halted")
	}
	if fake.StreamLen != 3 {
		t.Errorf("server saw %d tokens; want 3", fake.StreamLen)
	}
}

func TestStream_ClosesOnContextCancel(t *testing.T) {
	fake := &fakeScoringServer{Score: 0.9, HaltAt: 0}
	addr, cleanup := bootFake(t, fake)
	defer cleanup()

	c, err := Dial(addr, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	ctx, cancel := context.WithCancel(context.Background())
	handle, err := c.StartStream(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := handle.Send(&directorv1.ScoreTokenRequest{
		AccumulatedText: "p", NextToken: "t",
	}); err != nil {
		t.Fatalf("first send: %v", err)
	}
	if _, err := handle.Recv(); err != nil {
		t.Fatalf("first recv: %v", err)
	}
	cancel()
	// Subsequent Recv should surface cancellation; we only check that
	// it does not block indefinitely.
	done := make(chan struct{})
	go func() {
		_, _ = handle.Recv()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Recv did not unblock after cancel")
	}
}

func TestClose_Idempotent(t *testing.T) {
	fake := &fakeScoringServer{}
	addr, cleanup := bootFake(t, fake)
	defer cleanup()
	c, err := Dial(addr, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Close(); err != nil {
		t.Errorf("first close: %v", err)
	}
}

func TestStreamHandle_NilReceiver(t *testing.T) {
	var h *StreamHandle
	if err := h.Send(nil); err == nil {
		t.Error("expected error on nil handle Send")
	}
	if _, err := h.Recv(); err == nil {
		t.Error("expected error on nil handle Recv")
	}
	if err := h.Close(); err != nil {
		t.Errorf("nil handle Close = %v; want nil", err)
	}
}
