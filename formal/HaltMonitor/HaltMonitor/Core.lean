-- SPDX-License-Identifier: AGPL-3.0-or-later
-- Commercial licence available
-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
-- © Code 2020–2026 Miroslav Šotek. All rights reserved.
-- ORCID: 0009-0009-3560-0851
-- Contact: www.anulum.li | protoscience@anulum.li
-- Director-Class AI — HaltMonitor core model

/-!
# HaltMonitor core model

A minimal, self-contained model of the `stream_output` loop in
`src/director_ai/core/runtime/kernel.py`. The Python loop consumes a
token generator, queries a coherence callback for each token, and
halts if the returned score drops below `hard_limit`.

We abstract that into a function on `List (Token × Score)` and prove
three safety properties in `HaltMonitor.Properties`:

1. **Halt is irrevocable.** Once a token fails the threshold, the
   output is the `halted` prefix — later tokens cannot recover.
2. **Emitted tokens always pass the threshold.** If a token appears
   in the emitted prefix, its score was `≥ hard_limit`.
3. **Soundness w.r.t. a failing score.** If any input token's score
   is `< hard_limit`, the output is `halted` (not `emitted`).

The model intentionally ignores timeouts, callbacks, and the mutable
`_active` flag from the Python class. Those are orchestration
details; the threshold check is the safety core and the only part
that benefits from a formal guarantee.
-/

namespace HaltMonitor

/-- Token type — a single unit of output. The HaltMonitor in Python
treats tokens as strings, but the model is polymorphic. -/
abbrev Token := String

/-- Coherence score. Rational so the model stays decidable and
runnable; the production Python implementation uses `Float` but that
is a boundary-precision concern, not a safety one. -/
abbrev Score := Rat

/-- Input item: a token with its scorer output. -/
abbrev Item := Token × Score

/-- The result of running the HaltMonitor on a stream.

`emitted ts` means the monitor walked the entire input and every
token's score was `≥ hard_limit`, so `ts` is exactly the input
token list.

`halted ts` means the monitor emitted the prefix `ts` and then hit
a token whose score was `< hard_limit`. The failing token is NOT
present in `ts`, matching the Python behaviour where the halt
message is returned instead of the offending token.
-/
inductive Output where
  | emitted : List Token → Output
  | halted  : List Token → Output
  deriving DecidableEq, Repr

/-- Run the HaltMonitor on a list of `(token, score)` items.

Mirrors `stream_output` line-by-line:
`current_score < self.hard_limit` → halt.
-/
def run (hardLimit : Score) : List Item → Output
  | [] => Output.emitted []
  | (tok, score) :: rest =>
    if score < hardLimit then
      Output.halted []
    else
      match run hardLimit rest with
      | Output.emitted toks => Output.emitted (tok :: toks)
      | Output.halted toks => Output.halted (tok :: toks)

/-- Project an `Output` to the emitted token list regardless of
outcome. Useful in proofs about the emitted prefix. -/
def Output.tokens : Output → List Token
  | Output.emitted ts => ts
  | Output.halted ts => ts

/-- Predicate: the output is an `emitted` result. -/
def Output.isEmitted : Output → Bool
  | Output.emitted _ => true
  | Output.halted _ => false

/-- Any input item's score passes the threshold. -/
def Item.passes (hardLimit : Score) (it : Item) : Prop :=
  ¬ (it.2 < hardLimit)

instance (hardLimit : Score) (it : Item) :
    Decidable (Item.passes hardLimit it) := by
  unfold Item.passes
  infer_instance

end HaltMonitor
