//! TokenIpsum - Mock LLM API Server
//!
//! Generates fake but structurally accurate responses for testing LLM clients.
//!
//! Usage:
//!   tokenipsum                    # Use defaults
//!   CONFIG=config.toml tokenipsum # Use config file

use std::net::SocketAddr;

use tokenipsum::{create_router, Config, RuntimeState};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "tokenipsum=info,tower_http=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config_path = std::env::var("CONFIG").unwrap_or_else(|_| "config.toml".to_string());
    let config = Config::load_from(&config_path);

    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(config.server.port);

    let state = RuntimeState::new(config.clone());

    // Log enabled providers
    if config.providers.cerebras {
        tracing::info!("Cerebras endpoint: POST /v1/chat/completions");
    }
    if config.providers.gemini {
        tracing::info!("Gemini endpoint: POST /v1beta/models/{{model}}:generateContent");
        tracing::info!("Gemini endpoint: POST /v1beta/models/{{model}}:streamGenerateContent");
    }
    if config.providers.claude {
        tracing::info!("Claude endpoint: POST /v1/messages");
    }
    if config.providers.openai {
        tracing::info!("OpenAI endpoint: POST /v1/responses");
    }

    let app = create_router(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("TokenIpsum listening on http://{}", addr);

    if config.rate_limit.fail_after_requests > 0 {
        tracing::info!(
            "Rate limit: will return 429 after {} requests",
            config.rate_limit.fail_after_requests
        );
    }
    if config.errors.error_rate > 0.0 {
        tracing::info!(
            "Random errors enabled: {:.0}% chance",
            config.errors.error_rate * 100.0
        );
    }

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
