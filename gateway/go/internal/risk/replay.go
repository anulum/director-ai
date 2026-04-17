// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — in-memory body replay helper

package risk

import (
	"bytes"
	"io"
)

// _replayReader returns an ``io.ReadCloser`` backed by an in-memory
// byte slice. Unlike ``io.NopCloser(bytes.NewReader(...))`` this
// preserves compatibility with ``net/http``'s expectation that the
// body can be re-read if a middleware decides to rewind it.
func _replayReader(data []byte) io.ReadCloser {
	return &memoryBody{buf: bytes.NewReader(data)}
}

type memoryBody struct {
	buf *bytes.Reader
}

func (m *memoryBody) Read(p []byte) (int, error) {
	return m.buf.Read(p)
}

func (m *memoryBody) Close() error {
	// No underlying resource — ``net/http`` still calls ``Close``
	// for consistency, so the method must not return a spurious
	// error.
	return nil
}
