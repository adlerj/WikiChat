//! Simple tokenizer for BM25
//!
//! Uses Unicode word boundaries and lowercase normalization.

use unicode_segmentation::UnicodeSegmentation;

/// Simple tokenizer that splits on whitespace and punctuation
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Minimum token length (default: 2)
    pub min_length: usize,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self { min_length: 2 }
    }
}

impl Tokenizer {
    /// Create a new tokenizer with custom minimum length
    pub fn new(min_length: usize) -> Self {
        Self { min_length }
    }

    /// Tokenize text into terms
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.unicode_words()
            .map(|word| word.to_lowercase())
            .filter(|word| word.len() >= self.min_length)
            .collect()
    }

    /// Tokenize text into unique terms (for indexing)
    pub fn tokenize_unique(&self, text: &str) -> Vec<String> {
        let mut terms = self.tokenize(text);
        terms.sort_unstable();
        terms.dedup();
        terms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let tokenizer = Tokenizer::default();
        let tokens = tokenizer.tokenize("Hello World! This is a test.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "test"]);
    }

    #[test]
    fn test_min_length() {
        let tokenizer = Tokenizer::new(3);
        let tokens = tokenizer.tokenize("a bb ccc dddd");
        assert_eq!(tokens, vec!["ccc", "dddd"]);
    }

    #[test]
    fn test_unicode() {
        let tokenizer = Tokenizer::default();
        let tokens = tokenizer.tokenize("Café résumé naïve");
        assert_eq!(tokens, vec!["café", "résumé", "naïve"]);
    }

    #[test]
    fn test_unique() {
        let tokenizer = Tokenizer::default();
        let tokens = tokenizer.tokenize_unique("the quick brown fox jumps over the lazy dog");
        assert_eq!(tokens, vec!["brown", "dog", "fox", "jumps", "lazy", "over", "quick", "the"]);
    }
}
