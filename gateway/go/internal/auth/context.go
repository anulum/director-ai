// SPDX-License-Identifier: Apache-2.0
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — auth context helpers

package auth

import "context"

func withFingerprint(ctx context.Context, fp string) context.Context {
	ctx = context.WithValue(ctx, KeyFingerprint, fp)
	ctx = context.WithValue(ctx, KeyAuthenticated, true)
	return ctx
}
