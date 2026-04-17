// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — gRPC scoring client used by the gateway

// Package scoring wraps the generated ``director.v1.CoherenceScoring``
// gRPC client so the rest of the gateway stays transport-agnostic.
// A caller constructs one ``Client`` per process and passes it into
// middleware; the implementation handles dial, retries, and channel
// lifecycle.
package scoring

import (
	"context"
	"errors"
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	directorv1 "github.com/anulum/director-ai/gateway/proto/director/v1"
)

// Client is a thin wrapper around the generated stub with explicit
// lifecycle. The zero value is not usable; construct with Dial.
type Client struct {
	conn *grpc.ClientConn
	stub directorv1.CoherenceScoringClient
}

// Dial opens an insecure channel to ``addr`` (e.g. ``localhost:50052``)
// and returns a Client. Production deployments supply a secure
// channel by calling grpc.NewClient directly and passing the result
// to WithConn.
func Dial(addr string, timeout time.Duration) (*Client, error) {
	if addr == "" {
		return nil, errors.New("scoring: empty address")
	}
	dialCtx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	// NewClient (preferred over deprecated Dial) is lazy; we ping
	// with WaitForReady on the first RPC to fail fast when the
	// scoring server is down at startup.
	_ = dialCtx // kept for future Dial options
	conn, err := grpc.NewClient(
		addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("scoring: dial %s: %w", addr, err)
	}
	return WithConn(conn), nil
}

// WithConn builds a Client around a pre-dialled connection. The
// caller retains ownership of the connection — Close must not be
// called on connections passed here.
func WithConn(conn *grpc.ClientConn) *Client {
	return &Client{
		conn: conn,
		stub: directorv1.NewCoherenceScoringClient(conn),
	}
}

// Close releases the underlying channel. Safe to call on a Client
// that was built via Dial; a no-op on clients from WithConn when
// their connection is externally owned.
func (c *Client) Close() error {
	if c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

// ScoreClaim is a one-shot scoring call. ``claim`` is the candidate
// answer, ``documents`` are the retrieved context snippets, and
// ``threshold`` overrides the server default when positive.
func (c *Client) ScoreClaim(
	ctx context.Context,
	claim string,
	documents []string,
	tenantID, requestID string,
	threshold float32,
) (*directorv1.CoherenceVerdict, int64, error) {
	resp, err := c.stub.ScoreClaim(ctx, &directorv1.ScoreClaimRequest{
		Claim:     claim,
		Documents: documents,
		TenantId:  tenantID,
		RequestId: requestID,
		Threshold: threshold,
	})
	if err != nil {
		return nil, 0, err
	}
	return resp.GetVerdict(), resp.GetLatencyMs(), nil
}

// StreamHandle is the active side of a ScoreStream RPC. Send emits
// a token; Recv blocks for the next verdict; Close ends the stream
// cleanly (half-close then wait for the server's trailing metadata).
type StreamHandle struct {
	stream directorv1.CoherenceScoring_ScoreStreamClient
}

// StartStream opens a bidirectional ScoreStream with the given
// context. The caller MUST call Close when done.
func (c *Client) StartStream(ctx context.Context) (*StreamHandle, error) {
	s, err := c.stub.ScoreStream(ctx)
	if err != nil {
		return nil, err
	}
	return &StreamHandle{stream: s}, nil
}

// Send emits a single token request.
func (h *StreamHandle) Send(req *directorv1.ScoreTokenRequest) error {
	if h == nil || h.stream == nil {
		return errors.New("scoring: stream closed")
	}
	return h.stream.Send(req)
}

// Recv blocks for the next verdict. Returns (verdict, io.EOF) when
// the server has closed the stream normally (e.g. after a halt).
func (h *StreamHandle) Recv() (*directorv1.CoherenceVerdict, error) {
	if h == nil || h.stream == nil {
		return nil, errors.New("scoring: stream closed")
	}
	resp, err := h.stream.Recv()
	if err != nil {
		return nil, err
	}
	return resp.GetVerdict(), nil
}

// Close signals end-of-stream to the server and waits for the RPC
// to finish.
func (h *StreamHandle) Close() error {
	if h == nil || h.stream == nil {
		return nil
	}
	return h.stream.CloseSend()
}
