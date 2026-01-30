//! Python bindings for PocketWiki Rust components

use pocketwiki_core::bm25::{BM25Index as CoreBM25Index, BM25Params, SearchResult as CoreSearchResult};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Python-exposed search result
#[pyclass]
#[derive(Clone)]
pub struct SearchResult {
    #[pyo3(get)]
    pub chunk_id: String,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub rank: usize,
}

impl From<CoreSearchResult> for SearchResult {
    fn from(result: CoreSearchResult) -> Self {
        Self {
            chunk_id: result.chunk_id,
            score: result.score,
            rank: result.rank,
        }
    }
}

#[pymethods]
impl SearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(chunk_id='{}', score={:.4}, rank={})",
            self.chunk_id, self.score, self.rank
        )
    }

    fn to_dict(&self) -> HashMap<String, PyObject> {
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("chunk_id".to_string(), self.chunk_id.to_object(py));
            map.insert("score".to_string(), self.score.to_object(py));
            map.insert("rank".to_string(), self.rank.to_object(py));
            map
        })
    }
}

/// Python-exposed BM25 index
#[pyclass]
pub struct BM25Index {
    index: CoreBM25Index,
}

#[pymethods]
impl BM25Index {
    /// Create a new BM25 index
    #[new]
    #[pyo3(signature = (k1=1.5, b=0.75))]
    fn new(k1: f32, b: f32) -> Self {
        let params = BM25Params { k1, b };
        Self {
            index: CoreBM25Index::with_params(params),
        }
    }

    /// Add a document to the index
    ///
    /// Args:
    ///     doc_id: Document identifier (integer)
    ///     text: Document text content
    fn add_document(&mut self, doc_id: u32, text: &str) {
        self.index.add_document(doc_id, text);
    }

    /// Build the index (must call after adding all documents)
    fn build(&mut self) {
        self.index.build();
    }

    /// Search the index
    ///
    /// Args:
    ///     query: Search query text
    ///     k: Number of results to return (default: 10)
    ///
    /// Returns:
    ///     List of SearchResult objects
    #[pyo3(signature = (query, k=10))]
    fn search(&self, query: &str, k: usize) -> Vec<SearchResult> {
        self.index
            .search(query, k)
            .into_iter()
            .map(SearchResult::from)
            .collect()
    }

    /// Get index statistics
    ///
    /// Returns:
    ///     Dictionary with num_docs, num_terms, avg_doc_len
    fn stats(&self) -> HashMap<String, PyObject> {
        let stats = self.index.stats();
        Python::with_gil(|py| {
            let mut map = HashMap::new();
            map.insert("num_docs".to_string(), stats.num_docs.to_object(py));
            map.insert("num_terms".to_string(), stats.num_terms.to_object(py));
            map.insert("avg_doc_len".to_string(), stats.avg_doc_len.to_object(py));
            map
        })
    }

    fn __repr__(&self) -> String {
        let stats = self.index.stats();
        format!(
            "BM25Index(num_docs={}, num_terms={}, avg_doc_len={:.2})",
            stats.num_docs, stats.num_terms, stats.avg_doc_len
        )
    }
}

/// Python module
#[pymodule]
fn pocketwiki_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BM25Index>()?;
    m.add_class::<SearchResult>()?;
    Ok(())
}
