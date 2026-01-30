//! BM25 scoring and inverted index implementation

use ahash::{AHashMap, AHashSet};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::tokenizer::Tokenizer;
use crate::varint::encode_postings;

/// BM25 parameters
#[derive(Debug, Clone, Copy)]
pub struct BM25Params {
    pub k1: f32,
    pub b: f32,
}

impl Default for BM25Params {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }
}

/// BM25 scorer for calculating relevance scores
pub struct BM25Scorer {
    params: BM25Params,
    avg_doc_len: f32,
    doc_count: usize,
}

impl BM25Scorer {
    /// Create a new BM25 scorer
    pub fn new(params: BM25Params, avg_doc_len: f32, doc_count: usize) -> Self {
        Self {
            params,
            avg_doc_len,
            doc_count,
        }
    }

    /// Calculate BM25 score for a single term
    pub fn score_term(
        &self,
        term_freq: f32,
        doc_len: f32,
        doc_freq: usize,
    ) -> f32 {
        let idf = self.idf(doc_freq);
        let tf_component = (term_freq * (self.params.k1 + 1.0))
            / (term_freq + self.params.k1 * (1.0 - self.params.b + self.params.b * (doc_len / self.avg_doc_len)));
        idf * tf_component
    }

    /// Calculate IDF (inverse document frequency)
    fn idf(&self, doc_freq: usize) -> f32 {
        let n = self.doc_count as f32;
        let df = doc_freq as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }
}

/// Document metadata for BM25
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocMeta {
    pub doc_id: u32,
    pub doc_len: u32,
}

/// Search result
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub chunk_id: String,
    pub score: f32,
    pub rank: usize,
}

/// BM25 inverted index
pub struct BM25Index {
    tokenizer: Tokenizer,
    params: BM25Params,

    // Core index structures
    postings: AHashMap<String, Vec<u8>>, // term -> compressed doc_ids
    term_freqs: AHashMap<String, HashMap<u32, u32>>, // term -> {doc_id: freq}
    doc_metas: Vec<DocMeta>,

    // Statistics
    total_doc_len: u64,
}

impl BM25Index {
    /// Create a new empty index
    pub fn new() -> Self {
        Self::with_params(BM25Params::default())
    }

    /// Create index with custom parameters
    pub fn with_params(params: BM25Params) -> Self {
        Self {
            tokenizer: Tokenizer::default(),
            params,
            postings: AHashMap::new(),
            term_freqs: AHashMap::new(),
            doc_metas: Vec::new(),
            total_doc_len: 0,
        }
    }

    /// Add a document to the index
    pub fn add_document(&mut self, doc_id: u32, text: &str) {
        let tokens = self.tokenizer.tokenize(text);
        let doc_len = tokens.len() as u32;

        // Track document metadata
        self.doc_metas.push(DocMeta { doc_id, doc_len });
        self.total_doc_len += doc_len as u64;

        // Count term frequencies
        let mut term_counts: HashMap<String, u32> = HashMap::new();
        for term in tokens {
            *term_counts.entry(term).or_insert(0) += 1;
        }

        // Update inverted index
        for (term, count) in term_counts {
            self.term_freqs
                .entry(term)
                .or_insert_with(HashMap::new)
                .insert(doc_id, count);
        }
    }

    /// Build compressed postings lists (call after adding all documents)
    pub fn build(&mut self) {
        for (term, doc_freqs) in &self.term_freqs {
            let mut doc_ids: Vec<u32> = doc_freqs.keys().copied().collect();
            doc_ids.sort_unstable();
            let compressed = encode_postings(&doc_ids);
            self.postings.insert(term.clone(), compressed);
        }
    }

    /// Search the index
    pub fn search(&self, query: &str, k: usize) -> Vec<SearchResult> {
        let query_tokens = self.tokenizer.tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        // Create scorer
        let avg_doc_len = if self.doc_metas.is_empty() {
            1.0
        } else {
            self.total_doc_len as f32 / self.doc_metas.len() as f32
        };
        let scorer = BM25Scorer::new(self.params, avg_doc_len, self.doc_metas.len());

        // Collect candidate documents
        let mut candidates = AHashSet::new();
        for token in &query_tokens {
            if let Some(term_docs) = self.term_freqs.get(token) {
                candidates.extend(term_docs.keys());
            }
        }

        // Score each candidate
        let mut scores: Vec<(u32, f32)> = candidates
            .iter()
            .map(|&&doc_id| {
                let score = self.score_document(doc_id, &query_tokens, &scorer);
                (doc_id, score)
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k results
        scores
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(rank, (doc_id, score))| SearchResult {
                chunk_id: format!("chunk_{}", doc_id),
                score,
                rank,
            })
            .collect()
    }

    /// Score a single document for a query
    fn score_document(&self, doc_id: u32, query_tokens: &[String], scorer: &BM25Scorer) -> f32 {
        let doc_len = self.doc_metas
            .iter()
            .find(|m| m.doc_id == doc_id)
            .map(|m| m.doc_len as f32)
            .unwrap_or(1.0);

        let mut score = 0.0;
        for token in query_tokens {
            if let Some(term_docs) = self.term_freqs.get(token) {
                if let Some(&term_freq) = term_docs.get(&doc_id) {
                    let doc_freq = term_docs.len();
                    score += scorer.score_term(term_freq as f32, doc_len, doc_freq);
                }
            }
        }
        score
    }

    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            num_docs: self.doc_metas.len(),
            num_terms: self.postings.len(),
            avg_doc_len: if self.doc_metas.is_empty() {
                0.0
            } else {
                self.total_doc_len as f32 / self.doc_metas.len() as f32
            },
        }
    }
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::new()
    }
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub num_docs: usize,
    pub num_terms: usize,
    pub avg_doc_len: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_scorer() {
        let scorer = BM25Scorer::new(BM25Params::default(), 10.0, 100);
        let score = scorer.score_term(2.0, 10.0, 50);
        assert!(score > 0.0);
    }

    #[test]
    fn test_index_build() {
        let mut index = BM25Index::new();
        index.add_document(1, "the quick brown fox");
        index.add_document(2, "the lazy dog");
        index.add_document(3, "quick brown dog");
        index.build();

        let stats = index.stats();
        assert_eq!(stats.num_docs, 3);
        assert!(stats.num_terms > 0);
    }

    #[test]
    fn test_search() {
        let mut index = BM25Index::new();
        index.add_document(1, "Python programming language");
        index.add_document(2, "Rust systems programming");
        index.add_document(3, "Python data science");
        index.build();

        let results = index.search("Python programming", 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk_id, "chunk_1"); // Best match
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_empty_query() {
        let mut index = BM25Index::new();
        index.add_document(1, "test document");
        index.build();

        let results = index.search("", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_no_results() {
        let mut index = BM25Index::new();
        index.add_document(1, "Python programming");
        index.build();

        let results = index.search("JavaScript", 10);
        assert!(results.is_empty());
    }
}
