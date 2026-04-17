-- SPDX-License-Identifier: AGPL-3.0-or-later
-- Commercial licence available
-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
-- © Code 2020–2026 Miroslav Šotek. All rights reserved.
-- ORCID: 0009-0009-3560-0851
-- Contact: www.anulum.li | protoscience@anulum.li
-- Director-Class AI — HaltMonitor safety theorems

import HaltMonitor.Core

/-!
# Safety theorems for the HaltMonitor model

Three guarantees are proved:

* `run_emitted_preserves_input` — when the result is `emitted`, the
  emitted tokens equal the projected input tokens.
* `run_emitted_implies_all_pass` — when the result is `emitted`,
  every input item passes the threshold.
* `run_any_fail_implies_halted` — if any input item fails the
  threshold, the result is `halted`, not `emitted`.

Together these formalise the informal claim "no token whose
coherence score falls below `hard_limit` is ever emitted".
-/

namespace HaltMonitor

/-- `emitted ts` can only come from the input whose token projection
is `ts`. -/
theorem run_emitted_preserves_input
    (hardLimit : Score) :
    ∀ (items : List Item) (ts : List Token),
      run hardLimit items = Output.emitted ts → ts = items.map Prod.fst
  | [], ts, h => by
    simp [run] at h
    simp [h]
  | (tok, score) :: rest, ts, h => by
    simp [run] at h
    by_cases hfail : score < hardLimit
    · simp [hfail] at h
    · simp [hfail] at h
      split at h
      next toks heq =>
        cases h
        have ih := run_emitted_preserves_input hardLimit rest toks heq
        simp [ih]
      next => exact absurd h (by intro h; cases h)

/-- When the monitor returns `emitted`, every input item has a score
of at least `hardLimit`. -/
theorem run_emitted_implies_all_pass
    (hardLimit : Score) :
    ∀ (items : List Item) (ts : List Token),
      run hardLimit items = Output.emitted ts →
        ∀ it ∈ items, Item.passes hardLimit it
  | [], _, _, it, hmem => by cases hmem
  | (tok, score) :: rest, ts, h, it, hmem => by
    simp [run] at h
    by_cases hfail : score < hardLimit
    · simp [hfail] at h
    · simp [hfail] at h
      split at h
      next toks heq =>
        cases hmem with
        | head =>
          simp [Item.passes]
          exact hfail
        | tail _ hmem' =>
          exact run_emitted_implies_all_pass hardLimit rest toks heq it hmem'
      next => exact absurd h (by intro h; cases h)

/-- The contrapositive: if any input item fails the threshold, the
run does not return `emitted`. -/
theorem run_any_fail_implies_not_emitted
    (hardLimit : Score) (items : List Item)
    (hexists : ∃ it ∈ items, ¬ Item.passes hardLimit it) :
    ∀ ts, run hardLimit items ≠ Output.emitted ts := by
  intro ts heq
  obtain ⟨it, hmem, hfail⟩ := hexists
  exact hfail (run_emitted_implies_all_pass hardLimit items ts heq it hmem)

/-- The stronger phrasing: a failure anywhere in the input forces
the monitor into the `halted` constructor. -/
theorem run_any_fail_implies_halted
    (hardLimit : Score) (items : List Item)
    (hexists : ∃ it ∈ items, ¬ Item.passes hardLimit it) :
    ∃ ts, run hardLimit items = Output.halted ts := by
  match hrun : run hardLimit items with
  | Output.emitted ts =>
    exact absurd hrun (run_any_fail_implies_not_emitted hardLimit items hexists ts)
  | Output.halted ts =>
    exact ⟨ts, rfl⟩

/-- Concrete executable example: a failing token halts the stream
with the already-emitted prefix attached. -/
example :
    run (1/2) [("hello", 3/4), ("world", 1/4), ("ignored", 9/10)]
      = Output.halted ["hello"] := by
  native_decide

/-- Concrete executable example: an all-passing stream emits
unchanged. -/
example :
    run (1/2) [("a", 3/4), ("b", 7/10)]
      = Output.emitted ["a", "b"] := by
  native_decide

end HaltMonitor
