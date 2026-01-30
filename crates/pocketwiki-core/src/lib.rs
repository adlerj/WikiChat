//! PocketWiki Core - High-performance BM25 retrieval in Rust
//!
//! This library provides efficient sparse retrieval using BM25 algorithm
//! with compressed inverted index storage.

pub mod bm25;
pub mod tokenizer;
pub mod varint;

pub use bm25::{BM25Index, BM25Scorer, SearchResult};
pub use tokenizer::Tokenizer;
