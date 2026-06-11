-- SPDX-License-Identifier: Apache-2.0
-- Commercial licence available
-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
-- © Code 2020–2026 Miroslav Šotek. All rights reserved.
-- ORCID: 0009-0009-3560-0851
-- Contact: www.anulum.li | protoscience@anulum.li
-- Director-Class AI — CoherenceScore range model

/-!
# Coherence score range

The `HaltMonitor` model in `HaltMonitor.Core` reasons about an *abstract*
`Score` flowing through the streaming-halt loop. This file gives that score a
concrete model at the level of `CoherenceScorer.calculate_coherence` in
`src/director_ai/core/scoring/scorer.py`, and proves the property the rest of
the pipeline (thresholding, halting, monotonicity) silently assumes:

> **The composite coherence score always lies in the unit interval `[0,1]`.**

The Python computation modelled here is the core path of
`_score_and_evidence`:

```python
total_divergence = self.W_LOGIC * h_logic + self.W_FACT * h_fact
coherence = 1.0 - total_divergence
```

with `W_LOGIC + W_FACT == 1`, both weights non-negative, and each divergence
component already clamped to `[0,1]` upstream (`max(0.0, min(1.0, …))`). That
makes `total_divergence` a **convex combination** of two unit-interval values,
hence itself in `[0,1]`, so `coherence = 1 − total_divergence ∈ [0,1]`.

The rescaling branch in the Python code applies an explicit
`max(0.0, min(1.0, …))`; `clampUnit` models that clamp and is proved to land in
`[0,1]` unconditionally, and to be the identity on scores that are already in
range (so clamping never perturbs a valid score).

This is FVGK programme phase 1: the abstract `Score` of the halt model is now
backed by a scorer-level model whose output range is machine-checked, not
assumed. All proofs are core Lean 4 (no Mathlib).
-/

namespace HaltMonitor

/-- A value constrained to the closed unit interval `[0,1]`.

Mirrors a divergence/probability component after the upstream
`max(0.0, min(1.0, …))` clamp in the Python scorer: it cannot be negative and
cannot exceed one. Rational so the model stays decidable and executable. -/
structure UnitValue where
  /-- The underlying rational value. -/
  val : Rat
  /-- The value is non-negative. -/
  lo : 0 ≤ val
  /-- The value does not exceed one. -/
  hi : val ≤ 1

/-- The two-component coherence model.

`hLogical` and `hFactual` are the logical- and factual-divergence components
(`h_logic`, `h_fact` in the Python scorer), each a `UnitValue`. `wLogical` and
`wFactual` are the mixing weights (`W_LOGIC`, `W_FACT`): non-negative and summing
to one, i.e. a convex combination. -/
structure CoherenceScore where
  /-- Logical-divergence component (NLI contradiction probability). -/
  hLogical : UnitValue
  /-- Factual-divergence component (ground-truth deviation). -/
  hFactual : UnitValue
  /-- Weight on the logical-divergence component. -/
  wLogical : Rat
  /-- Weight on the factual-divergence component. -/
  wFactual : Rat
  /-- The logical weight is non-negative. -/
  wLogical_nonneg : 0 ≤ wLogical
  /-- The factual weight is non-negative. -/
  wFactual_nonneg : 0 ≤ wFactual
  /-- The weights form a convex combination (sum to one). -/
  weights_sum_one : wLogical + wFactual = 1

/-- Total divergence: the convex combination of the two component divergences.
Mirrors `total_divergence = W_LOGIC * h_logic + W_FACT * h_fact`. -/
def CoherenceScore.totalDivergence (c : CoherenceScore) : Rat :=
  c.wLogical * c.hLogical.val + c.wFactual * c.hFactual.val

/-- The composite coherence score. Mirrors `coherence = 1.0 - total_divergence`. -/
def CoherenceScore.score (c : CoherenceScore) : Rat :=
  1 - c.totalDivergence

/-- Total divergence is non-negative: each weighted component is a product of
two non-negatives. -/
theorem CoherenceScore.totalDivergence_nonneg (c : CoherenceScore) :
    0 ≤ c.totalDivergence := by
  unfold CoherenceScore.totalDivergence
  exact Rat.add_nonneg
    (Rat.mul_nonneg c.wLogical_nonneg c.hLogical.lo)
    (Rat.mul_nonneg c.wFactual_nonneg c.hFactual.lo)

/-- Total divergence does not exceed one: bounding each component by its weight
(`w * h ≤ w * 1 = w`) and summing gives `wLogical + wFactual = 1`. -/
theorem CoherenceScore.totalDivergence_le_one (c : CoherenceScore) :
    c.totalDivergence ≤ 1 := by
  unfold CoherenceScore.totalDivergence
  have a1 : c.wLogical * c.hLogical.val ≤ c.wLogical * 1 :=
    Rat.mul_le_mul_of_nonneg_left c.hLogical.hi c.wLogical_nonneg
  have a2 : c.wFactual * c.hFactual.val ≤ c.wFactual * 1 :=
    Rat.mul_le_mul_of_nonneg_left c.hFactual.hi c.wFactual_nonneg
  rw [Rat.mul_one] at a1 a2
  have step1 := (Rat.add_le_add_right (c := c.wFactual * c.hFactual.val)).mpr a1
  have step2 := (Rat.add_le_add_left (c := c.wLogical)).mpr a2
  have hsum := Rat.le_trans step1 step2
  rw [c.weights_sum_one] at hsum
  exact hsum

/-- **Lower bound.** The composite coherence score is non-negative — a token can
never be assigned a negative coherence. Follows from `totalDivergence ≤ 1`. -/
theorem CoherenceScore.score_nonneg (c : CoherenceScore) : 0 ≤ c.score := by
  unfold CoherenceScore.score
  have key := (Rat.add_le_add_right (c := -c.totalDivergence)).mpr
    c.totalDivergence_le_one
  rw [Rat.add_neg_cancel] at key
  rw [Rat.sub_eq_add_neg]
  exact key

/-- **Upper bound.** The composite coherence score never exceeds one. Follows
from `0 ≤ totalDivergence`. -/
theorem CoherenceScore.score_le_one (c : CoherenceScore) : c.score ≤ 1 := by
  unfold CoherenceScore.score
  have key := (Rat.add_le_add_left (c := 1)).mpr
    (Rat.neg_le_neg c.totalDivergence_nonneg)
  rw [Rat.neg_zero, Rat.add_zero] at key
  rw [Rat.sub_eq_add_neg]
  exact key

/-- **`score() ∈ [0,1]`.** The composite coherence score always lies in the unit
interval — the range invariant the halt monitor, thresholding, and monotonicity
proofs assume of the abstract `Score`, now machine-checked for the scorer model. -/
theorem CoherenceScore.score_mem_unit (c : CoherenceScore) :
    0 ≤ c.score ∧ c.score ≤ 1 :=
  ⟨c.score_nonneg, c.score_le_one⟩

/-- Concrete executable check with the production default weights
(`W_LOGIC = 3/5`, `W_FACT = 2/5`) and mid-range divergences: the score is the
expected `1 − (3/5·1/2 + 2/5·1/4) = 0.6`. -/
example :
    CoherenceScore.score
      { hLogical := ⟨1/2, by native_decide, by native_decide⟩
        hFactual := ⟨1/4, by native_decide, by native_decide⟩
        wLogical := 3/5, wFactual := 2/5
        wLogical_nonneg := by native_decide, wFactual_nonneg := by native_decide
        weights_sum_one := by native_decide } = 3/5 := by native_decide

/-- Clamp a rational into `[0,1]`, modelling the Python `max(0.0, min(1.0, x))`
applied on the rescaling branch of the scorer. -/
def clampUnit (x : Rat) : Rat :=
  if x < 0 then 0 else if 1 < x then 1 else x

/-- The clamp output is non-negative, regardless of input. -/
theorem clampUnit_nonneg (x : Rat) : 0 ≤ clampUnit x := by
  unfold clampUnit
  split
  · exact Rat.le_refl
  · split
    · decide
    · rename_i h _; exact Rat.not_lt.mp h

/-- The clamp output does not exceed one, regardless of input. -/
theorem clampUnit_le_one (x : Rat) : clampUnit x ≤ 1 := by
  unfold clampUnit
  split
  · decide
  · split
    · exact Rat.le_refl
    · rename_i _ h; exact Rat.not_lt.mp h

/-- **`clampUnit x ∈ [0,1]`.** The explicit clamp guarantees the unit interval
for any input — so the rescaling path cannot produce an out-of-range score. -/
theorem clampUnit_mem_unit (x : Rat) : 0 ≤ clampUnit x ∧ clampUnit x ≤ 1 :=
  ⟨clampUnit_nonneg x, clampUnit_le_one x⟩

/-- The clamp is the identity on values already in `[0,1]`: it never perturbs a
score that is already valid, only reins in out-of-range ones. -/
theorem clampUnit_id_of_mem {x : Rat} (h0 : 0 ≤ x) (h1 : x ≤ 1) :
    clampUnit x = x := by
  unfold clampUnit
  rw [if_neg (Rat.not_lt.mpr h0), if_neg (Rat.not_lt.mpr h1)]

/-- Clamping a coherence score is therefore a no-op: the model already proves the
score is in range, so the production clamp is a defensive belt-and-braces, never
a correction. -/
theorem CoherenceScore.clamp_score_eq (c : CoherenceScore) :
    clampUnit c.score = c.score :=
  clampUnit_id_of_mem c.score_nonneg c.score_le_one

end HaltMonitor
