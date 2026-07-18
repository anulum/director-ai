// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — ffi::retrieval
//! BM25 sparse-retrieval engine binding.

use pyo3::prelude::*;

/// Rust-accelerated BM25 sparse retrieval engine.
#[pyclass(name = "RustBM25")]
struct PyBM25 {
    inner: backfire_core::BM25Engine,
}

#[pymethods]
impl PyBM25 {
    #[new]
    #[pyo3(signature = (k1 = 1.2, b = 0.75))]
    fn new(k1: f64, b: f64) -> Self {
        Self {
            inner: backfire_core::BM25Engine::new(k1, b),
        }
    }

    /// Add a document to the BM25 index.
    fn add_document(&self, doc_id: &str, text: &str) {
        self.inner.add_document(doc_id, text);
    }

    /// Query the index, returning list of (doc_id, score) tuples.
    fn query(&self, query_text: &str, n_results: usize) -> Vec<(String, f64)> {
        self.inner
            .query(query_text, n_results)
            .into_iter()
            .map(|r| (r.doc_id, r.score))
            .collect()
    }

    /// Number of indexed documents.
    fn count(&self) -> usize {
        self.inner.count()
    }

    /// Clear all documents.
    fn clear(&self) {
        self.inner.clear();
    }
}

/// Register the BM25 retrieval engine on the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBM25>()?;
    Ok(())
}
