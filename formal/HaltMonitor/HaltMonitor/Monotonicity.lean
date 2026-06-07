-- SPDX-License-Identifier: AGPL-3.0-or-later
-- Commercial licence available
-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
-- © Code 2020–2026 Miroslav Šotek. All rights reserved.
-- ORCID: 0009-0009-3560-0851
-- Contact: www.anulum.li | protoscience@anulum.li
-- Director-Class AI — HaltMonitor threshold-monotonicity theorems

import HaltMonitor.Core
import HaltMonitor.Properties

/-!
# Threshold monotonicity for the HaltMonitor model

The safety theorems in `HaltMonitor.Properties` fix the threshold and reason
about a single run. This file proves how the monitor behaves as the threshold
*moves* — the formal counterpart of "`approved()` is monotone with respect to the
threshold" (Verified-AI FVGK programme, phase 1):

* `passes_antitone` — passing a stricter (higher) threshold implies passing any
  looser (lower) one.
* `run_all_pass_emitted` — the converse of `run_emitted_implies_all_pass`: if
  every item passes, the run emits the whole input.
* `run_threshold_monotone` — lowering the threshold can never turn an emitting
  stream into a halting one. Equivalently, raising the threshold is the only way
  to introduce a halt: tightening is safe-by-construction.
-/

namespace HaltMonitor

/-- Passing a stricter (higher) threshold implies passing any looser one. -/
theorem passes_antitone {h1 h2 : Score} (hle : h1 ≤ h2) {it : Item}
    (h : Item.passes h2 it) : Item.passes h1 it := by
  -- `Item.passes l it` is `¬ (it.2 < l)`, i.e. `l ≤ it.2` by `Rat.not_lt`.
  unfold Item.passes at *
  rw [Rat.not_lt] at h ⊢
  exact Rat.le_trans hle h

/-- Converse of `run_emitted_implies_all_pass`: when every item passes the
threshold, the monitor emits the whole input unchanged. -/
theorem run_all_pass_emitted (hardLimit : Score) :
    ∀ items : List Item,
      (∀ it ∈ items, Item.passes hardLimit it) →
        run hardLimit items = Output.emitted (items.map Prod.fst)
  | [], _ => by simp [run]
  | (tok, score) :: rest, hall => by
    have hhead : Item.passes hardLimit (tok, score) := hall (tok, score) (by simp)
    have hnotlt : ¬ (score < hardLimit) := hhead
    have hrest : ∀ it ∈ rest, Item.passes hardLimit it :=
      fun it hmem => hall it (by simp [hmem])
    have ih := run_all_pass_emitted hardLimit rest hrest
    simp [run, hnotlt, ih]

/-- Threshold monotonicity. If the stricter limit `h2` emits the whole stream,
then the looser limit `h1 ≤ h2` does too — lowering the threshold cannot
introduce a halt. -/
theorem run_threshold_monotone {h1 h2 : Score} (hle : h1 ≤ h2)
    (items : List Item) (hemit : (run h2 items).isEmitted = true) :
    (run h1 items).isEmitted = true := by
  match hrun : run h2 items with
  | Output.halted _ =>
    simp [hrun, Output.isEmitted] at hemit
  | Output.emitted ts =>
    have hall2 := run_emitted_implies_all_pass h2 items ts hrun
    have hall1 : ∀ it ∈ items, Item.passes h1 it :=
      fun it hmem => passes_antitone hle (hall2 it hmem)
    have hrun1 : run h1 items = Output.emitted (items.map Prod.fst) :=
      run_all_pass_emitted h1 items hall1
    simp [hrun1, Output.isEmitted]

/-- Concrete executable example: a stream that emits at the strict limit `3/4`
still emits at the looser limit `1/2`. -/
example :
    (run (1/2) [("a", 4/5), ("b", 9/10)]).isEmitted = true := by
  native_decide

end HaltMonitor
