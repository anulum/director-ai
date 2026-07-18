// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — ffi::pii
//! PII regex multi-pattern scanner binding.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use backfire_core::PiiScanner;

/// Python wrapper around ``backfire_core::PiiScanner``.
///
/// Construction takes an iterable of ``(category, pattern)``
/// tuples; every pattern is compiled eagerly and a bad regex
/// raises ``ValueError`` so operator mistakes surface immediately.
/// ``scan(text)`` returns a list of ``(category, start, end)``
/// tuples with byte offsets — the Python
/// ``RegexPIIDetector`` wraps these into ``ModerationMatch`` records
/// when ``backfire_kernel`` is installed.
#[pyclass(name = "PiiScanner")]
struct PyPiiScanner {
    inner: PiiScanner,
}

#[pymethods]
impl PyPiiScanner {
    #[new]
    fn new(patterns: Vec<(String, String)>) -> PyResult<Self> {
        let refs: Vec<(&str, &str)> = patterns
            .iter()
            .map(|(c, p)| (c.as_str(), p.as_str()))
            .collect();
        let inner = PiiScanner::new(&refs).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Scan ``text`` and return a list of ``(category, start, end)``
    /// tuples. Byte offsets; empty list on empty input.
    fn scan(&self, text: &str) -> Vec<(String, usize, usize)> {
        self.inner
            .scan(text)
            .into_iter()
            .map(|m| (m.category, m.start, m.end))
            .collect()
    }

    /// Number of registered pattern/category pairs.
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Register the PII scanner on the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPiiScanner>()?;
    Ok(())
}
