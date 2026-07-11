<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# factcg-cb — local fine-tune artefact directory

Local CB (contradiction/boundary) fine-tune of
`yaxili96/FactCG-DeBERTa-v3-Large`, driven by `tools/run_cb_training.py`
and `benchmarks/_cb_lowlr_train.py`. Small metadata (`config.json`,
`training_result.json`, tokeniser configs) is git-tracked; the binary
artefacts are not (WCH-6):

- `model.safetensors` — produced by the training run on this machine.
- `tokenizer.json`, `spm.model` — the tokeniser is inherited unchanged
  from the base model. Restore into this directory with:

  ```python
  from transformers import AutoTokenizer

  AutoTokenizer.from_pretrained(
      "yaxili96/FactCG-DeBERTa-v3-Large"
  ).save_pretrained("benchmarks/models/factcg-cb")
  ```

Nothing in `src/` or `tests/` reads this directory; it exists for the
GPU experiment scripts above.
