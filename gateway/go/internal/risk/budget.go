// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — per-tenant sliding-window risk budget

package risk

import (
	"container/list"
	"sync"
	"time"
)

// BudgetEntry mirrors the Python ``BudgetEntry`` dataclass.
// ``Accepted`` tells the caller whether the last reservation
// succeeded — without it, a non-zero ``Remaining`` could reflect
// either "reservation applied" or "reservation refused, ledger
// untouched".
type BudgetEntry struct {
	TenantID       string
	WindowSeconds  float64
	Allowance      float64
	Consumed       float64
	Remaining      float64
	Events         int
	Accepted       bool
}

// Exhausted reports whether the caller should reject the request.
func (e BudgetEntry) Exhausted() bool {
	return !e.Accepted || e.Remaining <= 0.0
}

// Budget is a sliding-window risk budget. Safe for concurrent use.
type Budget struct {
	allowance     float64
	window        time.Duration
	overrides     map[string]float64
	ledgers       map[string]*list.List
	totals        map[string]float64
	clock         func() time.Time
	mu            sync.Mutex
}

// NewBudget builds a budget with ``allowance`` per ``window``.
// Callers pass ``nil`` for ``clock`` to use ``time.Now``.
func NewBudget(allowance, windowSeconds float64, clock func() time.Time) (*Budget, error) {
	if allowance <= 0 {
		return nil, &invalidParamError{field: "allowance", value: allowance}
	}
	if windowSeconds <= 0 {
		return nil, &invalidParamError{field: "windowSeconds", value: windowSeconds}
	}
	if clock == nil {
		clock = time.Now
	}
	return &Budget{
		allowance: allowance,
		window:    time.Duration(windowSeconds * float64(time.Second)),
		overrides: make(map[string]float64),
		ledgers:   make(map[string]*list.List),
		totals:    make(map[string]float64),
		clock:     clock,
	}, nil
}

// SetAllowance overrides the allowance for a single tenant.
func (b *Budget) SetAllowance(tenantID string, allowance float64) error {
	if allowance <= 0 {
		return &invalidParamError{field: "allowance", value: allowance}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.overrides[tenantID] = allowance
	return nil
}

// AllowanceFor returns the effective allowance for ``tenantID``.
func (b *Budget) AllowanceFor(tenantID string) float64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	if a, ok := b.overrides[tenantID]; ok {
		return a
	}
	return b.allowance
}

// Reserve attempts to charge ``risk`` (clamped to [0, 1]) against
// ``tenantID``'s ledger. Pass zero to read the state without
// charging.
func (b *Budget) Reserve(tenantID string, risk float64) BudgetEntry {
	if risk < 0 {
		risk = 0
	} else if risk > 1 {
		risk = 1
	}
	now := b.clock()
	cutoff := now.Add(-b.window)
	b.mu.Lock()
	defer b.mu.Unlock()
	ledger, ok := b.ledgers[tenantID]
	if !ok {
		ledger = list.New()
		b.ledgers[tenantID] = ledger
	}
	// Prune expired entries.
	for e := ledger.Front(); e != nil; {
		entry := e.Value.(*ledgerEntry)
		if entry.ts.Before(cutoff) {
			next := e.Next()
			ledger.Remove(e)
			b.totals[tenantID] -= entry.risk
			e = next
			continue
		}
		break
	}
	allowance := b.allowance
	if a, ok := b.overrides[tenantID]; ok {
		allowance = a
	}
	tentative := b.totals[tenantID] + risk
	if tentative <= allowance+1e-9 {
		if risk > 0 {
			ledger.PushBack(&ledgerEntry{ts: now, risk: risk})
			b.totals[tenantID] += risk
		}
		return BudgetEntry{
			TenantID:      tenantID,
			WindowSeconds: b.window.Seconds(),
			Allowance:     allowance,
			Consumed:      b.totals[tenantID],
			Remaining:     allowance - b.totals[tenantID],
			Events:        ledger.Len(),
			Accepted:      true,
		}
	}
	return BudgetEntry{
		TenantID:      tenantID,
		WindowSeconds: b.window.Seconds(),
		Allowance:     allowance,
		Consumed:      b.totals[tenantID],
		Remaining:     allowance - b.totals[tenantID],
		Events:        ledger.Len(),
		Accepted:      false,
	}
}

// Snapshot reads the current ledger without charging.
func (b *Budget) Snapshot(tenantID string) BudgetEntry {
	return b.Reserve(tenantID, 0)
}

// Reset clears one tenant's ledger. Passing the empty string clears
// every tenant.
func (b *Budget) Reset(tenantID string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if tenantID == "" {
		b.ledgers = make(map[string]*list.List)
		b.totals = make(map[string]float64)
		return
	}
	delete(b.ledgers, tenantID)
	delete(b.totals, tenantID)
}

type ledgerEntry struct {
	ts   time.Time
	risk float64
}
