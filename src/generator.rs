//! Token and content generators for mock responses.

use fastrand::Rng;

/// Lorem ipsum style word list for generating fake content.
const WORDS: &[&str] = &[
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "I",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
    "AI",
    "model",
    "neural",
    "network",
    "learning",
    "data",
    "training",
    "inference",
    "token",
    "embedding",
    "transformer",
    "attention",
    "layer",
    "output",
    "input",
    "parameter",
    "weight",
    "gradient",
    "optimization",
    "loss",
    "accuracy",
    "batch",
];

/// Generator for fake content with configurable behavior.
pub struct ContentGenerator {
    rng: Rng,
    /// Average tokens per chunk for streaming.
    pub tokens_per_chunk: usize,
    /// Delay between chunks in ms (for realistic streaming).
    #[allow(dead_code)]
    pub chunk_delay_ms: u64,
}

impl ContentGenerator {
    pub fn new() -> Self {
        Self {
            rng: Rng::new(),
            tokens_per_chunk: 3,
            chunk_delay_ms: 20,
        }
    }

    #[cfg(test)]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: Rng::with_seed(seed),
            tokens_per_chunk: 3,
            chunk_delay_ms: 20,
        }
    }

    /// Generate a random word.
    pub fn word(&mut self) -> &'static str {
        WORDS[self.rng.usize(..WORDS.len())]
    }

    /// Generate N random words joined by spaces.
    pub fn words(&mut self, count: usize) -> String {
        (0..count)
            .map(|_| self.word())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Generate a sentence (5-15 words).
    pub fn sentence(&mut self) -> String {
        let count = self.rng.usize(5..15);
        let mut s = self.words(count);
        // Capitalize first letter
        if let Some(first) = s.get_mut(0..1) {
            first.make_ascii_uppercase();
        }
        s.push('.');
        s
    }

    /// Generate a paragraph (2-5 sentences).
    pub fn paragraph(&mut self) -> String {
        let count = self.rng.usize(2..5);
        (0..count)
            .map(|_| self.sentence())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Generate content chunks for streaming.
    /// Returns an iterator of (content, is_last) pairs.
    pub fn stream_chunks(&mut self, total_tokens: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut remaining = total_tokens;

        while remaining > 0 {
            let chunk_size = remaining.min(self.rng.usize(1..=self.tokens_per_chunk.max(1)));
            chunks.push(self.words(chunk_size));
            remaining = remaining.saturating_sub(chunk_size);
        }

        // Add punctuation to last chunk
        if let Some(last) = chunks.last_mut() {
            last.push('.');
        }

        chunks
    }

    /// Generate a random tool call ID.
    pub fn tool_call_id(&mut self) -> String {
        format!("{:011x}", self.rng.u64(..))
    }

    /// Generate a random chat completion ID.
    pub fn completion_id(&mut self) -> String {
        format!("chatcmpl-{}", uuid::Uuid::new_v4())
    }

    /// Generate a system fingerprint.
    pub fn fingerprint(&mut self) -> String {
        format!("fp_{:016x}", self.rng.u64(..))
    }

    /// Estimate token count from text (rough: ~4 chars per token).
    pub fn estimate_tokens(text: &str) -> u32 {
        ((text.len() as f32) / 4.0).ceil() as u32
    }
}

impl Default for ContentGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_generation() {
        let mut gen = ContentGenerator::new();
        let word = gen.word();
        assert!(!word.is_empty());
        assert!(WORDS.contains(&word));
    }

    #[test]
    fn test_sentence_generation() {
        let mut gen = ContentGenerator::new();
        let sentence = gen.sentence();
        assert!(sentence.ends_with('.'));
        assert!(sentence.chars().next().unwrap().is_uppercase());
    }

    #[test]
    fn test_stream_chunks() {
        let mut gen = ContentGenerator::new();
        let chunks = gen.stream_chunks(10);
        assert!(!chunks.is_empty());
        // Last chunk should end with punctuation
        assert!(chunks.last().unwrap().ends_with('.'));
    }

    #[test]
    fn test_deterministic_with_seed() {
        let mut gen1 = ContentGenerator::with_seed(42);
        let mut gen2 = ContentGenerator::with_seed(42);

        let words1: Vec<_> = (0..10).map(|_| gen1.word()).collect();
        let words2: Vec<_> = (0..10).map(|_| gen2.word()).collect();

        assert_eq!(words1, words2);
    }
}
