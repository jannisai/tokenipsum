//! Configuration management for TokenIpsum.

use serde::Deserialize;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Main configuration structure.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub rate_limit: RateLimitConfig,
    pub errors: ErrorConfig,
    pub auth: AuthConfig,
    pub providers: ProviderConfig,
    pub content: ContentConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub port: u16,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RateLimitConfig {
    pub enabled: bool,
    pub requests_per_minute: u32,
    pub fail_after_requests: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ErrorConfig {
    pub error_rate: f32,
    pub force_error: ForceError,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ForceError {
    #[default]
    None,
    Unauthorized,
    RateLimit,
    ServerError,
    Timeout,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AuthConfig {
    pub require_auth: bool,
    pub valid_keys: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ProviderConfig {
    pub cerebras: bool,
    pub gemini: bool,
    pub claude: bool,
    pub openai: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ContentConfig {
    pub deterministic: bool,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            rate_limit: RateLimitConfig::default(),
            errors: ErrorConfig::default(),
            auth: AuthConfig::default(),
            providers: ProviderConfig::default(),
            content: ContentConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8787,
            latency_ms: 0,
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_minute: 60,
            fail_after_requests: 0,
        }
    }
}

impl Default for ErrorConfig {
    fn default() -> Self {
        Self {
            error_rate: 0.0,
            force_error: ForceError::None,
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            require_auth: false,
            valid_keys: vec![],
        }
    }
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            cerebras: true,
            gemini: true,
            claude: true,
            openai: true,
        }
    }
}

impl Default for ContentConfig {
    fn default() -> Self {
        Self {
            deterministic: false,
            seed: 42,
        }
    }
}

impl Config {
    /// Load config from file, falling back to defaults.
    #[allow(dead_code)]
    pub fn load() -> Self {
        Self::load_from("config.toml")
    }

    /// Load config from a specific path.
    pub fn load_from<P: AsRef<Path>>(path: P) -> Self {
        match std::fs::read_to_string(path.as_ref()) {
            Ok(content) => toml::from_str(&content).unwrap_or_else(|e| {
                tracing::warn!("Failed to parse config: {}, using defaults", e);
                Self::default()
            }),
            Err(_) => {
                tracing::info!("No config file found, using defaults");
                Self::default()
            }
        }
    }
}

/// Runtime state for tracking requests and errors.
#[derive(Debug)]
pub struct RuntimeState {
    pub config: Config,
    pub request_count: AtomicU64,
    rng: std::sync::Mutex<fastrand::Rng>,
}

impl RuntimeState {
    pub fn new(config: Config) -> Arc<Self> {
        let seed = if config.content.deterministic {
            config.content.seed
        } else {
            fastrand::u64(..)
        };

        Arc::new(Self {
            config,
            request_count: AtomicU64::new(0),
            rng: std::sync::Mutex::new(fastrand::Rng::with_seed(seed)),
        })
    }

    /// Increment request count and return current count.
    pub fn increment_requests(&self) -> u64 {
        self.request_count.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Check if we should return an error based on config.
    pub fn should_error(&self) -> Option<ErrorType> {
        // Check forced error
        match &self.config.errors.force_error {
            ForceError::None => {}
            ForceError::Unauthorized => return Some(ErrorType::Unauthorized),
            ForceError::RateLimit => return Some(ErrorType::RateLimit),
            ForceError::ServerError => return Some(ErrorType::ServerError),
            ForceError::Timeout => return Some(ErrorType::Timeout),
        }

        // Check rate limit
        if self.config.rate_limit.fail_after_requests > 0 {
            let count = self.request_count.load(Ordering::SeqCst);
            if count >= self.config.rate_limit.fail_after_requests {
                return Some(ErrorType::RateLimit);
            }
        }

        // Check random error rate
        if self.config.errors.error_rate > 0.0 {
            let mut rng = self.rng.lock().unwrap();
            if rng.f32() < self.config.errors.error_rate {
                // Random error type
                return Some(match rng.u8(0..3) {
                    0 => ErrorType::Unauthorized,
                    1 => ErrorType::RateLimit,
                    _ => ErrorType::ServerError,
                });
            }
        }

        None
    }

    /// Check if API key is valid.
    pub fn is_valid_key(&self, key: Option<&str>) -> bool {
        if !self.config.auth.require_auth {
            return true;
        }

        match key {
            Some(k) => self.config.auth.valid_keys.iter().any(|valid| valid == k),
            None => false,
        }
    }

    /// Get latency to add (in ms).
    pub fn latency_ms(&self) -> u64 {
        self.config.server.latency_ms
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ErrorType {
    Unauthorized,
    RateLimit,
    ServerError,
    Timeout,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.server.port, 8787);
        assert!(!config.rate_limit.enabled);
        assert!(config.providers.cerebras);
    }

    #[test]
    fn test_parse_config() {
        let toml = r#"
            [server]
            port = 9000
            latency_ms = 100

            [rate_limit]
            enabled = true
            fail_after_requests = 10

            [errors]
            force_error = "rate_limit"
        "#;

        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.server.latency_ms, 100);
        assert!(config.rate_limit.enabled);
        assert_eq!(config.rate_limit.fail_after_requests, 10);
        assert_eq!(config.errors.force_error, ForceError::RateLimit);
    }

    #[test]
    fn test_runtime_state() {
        let config = Config::default();
        let state = RuntimeState::new(config);

        assert_eq!(state.increment_requests(), 1);
        assert_eq!(state.increment_requests(), 2);
    }

    #[test]
    fn test_force_error() {
        let mut config = Config::default();
        config.errors.force_error = ForceError::RateLimit;

        let state = RuntimeState::new(config);
        assert!(matches!(state.should_error(), Some(ErrorType::RateLimit)));
    }
}
