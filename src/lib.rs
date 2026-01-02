//! TokenIpsum - Mock LLM API Server
//!
//! A testing utility that generates fake but structurally accurate API responses
//! matching real LLM providers (OpenAI, Anthropic Claude, Google Gemini, Cerebras).
//!
//! # Usage as a Library
//!
//! ```rust,no_run
//! use tokenipsum::{Config, RuntimeState, create_router};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = Config::default();
//!     let state = RuntimeState::new(config);
//!     let app = create_router(state);
//!
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:8787").await.unwrap();
//!     axum::serve(listener, app).await.unwrap();
//! }
//! ```
//!
//! # Supported Providers
//!
//! - **Cerebras**: `/v1/chat/completions` - OpenAI-compatible chat completions
//! - **Claude**: `/v1/messages` - Anthropic Messages API
//! - **Gemini**: `/v1beta/models/{model}:generateContent` - Google Gemini API
//! - **OpenAI**: `/v1/responses` - OpenAI Responses API

pub mod cerebras;
pub mod claude;
pub mod config;
pub mod errors;
pub mod gemini;
pub mod generator;
pub mod openai;

pub use config::{Config, RuntimeState};
pub use errors::Provider;
pub use generator::ContentGenerator;

use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::State,
    http::{header::AUTHORIZATION, Request},
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
    Json, Router,
};
use tokio::time::sleep;
use tower_http::cors::CorsLayer;

type AppState = Arc<RuntimeState>;

/// Creates the configured Axum router with all enabled providers.
///
/// # Example
///
/// ```rust,no_run
/// use tokenipsum::{Config, RuntimeState, create_router};
///
/// let config = Config::default();
/// let state = RuntimeState::new(config);
/// let app = create_router(state);
/// ```
pub fn create_router(state: Arc<RuntimeState>) -> Router {
    let config = &state.config;
    let mut app = Router::new().route("/health", get(health));

    if config.providers.cerebras {
        app = app.route("/v1/chat/completions", post(cerebras_handler));
    }

    if config.providers.gemini {
        app = app.route("/v1beta/models/{model_action}", post(gemini_handler));
    }

    if config.providers.claude {
        app = app.route("/v1/messages", post(claude_handler));
    }

    if config.providers.openai {
        app = app.route("/v1/responses", post(openai_handler));
    }

    app.layer(middleware::from_fn_with_state(
        state.clone(),
        error_middleware,
    ))
    .layer(middleware::from_fn_with_state(
        state.clone(),
        latency_middleware,
    ))
    .layer(CorsLayer::permissive())
    .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn latency_middleware(
    State(state): State<AppState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let latency = state.latency_ms();
    if latency > 0 {
        sleep(Duration::from_millis(latency)).await;
    }
    next.run(request).await
}

async fn error_middleware(
    State(state): State<AppState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    state.increment_requests();

    if state.config.auth.require_auth {
        let auth = request
            .headers()
            .get(AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .map(|s| s.trim_start_matches("Bearer ").trim());

        if !state.is_valid_key(auth) {
            let provider = provider_from_path(request.uri().path());
            return errors::error_response(config::ErrorType::Unauthorized, provider);
        }
    }

    if let Some(error) = state.should_error() {
        let provider = provider_from_path(request.uri().path());
        return errors::error_response(error, provider);
    }

    next.run(request).await
}

fn provider_from_path(path: &str) -> Provider {
    if path.contains("/v1beta/models") {
        Provider::Gemini
    } else if path.contains("/v1/messages") {
        Provider::Claude
    } else if path.contains("/v1/responses") {
        Provider::OpenAI
    } else {
        Provider::Cerebras
    }
}

async fn cerebras_handler(
    State(_state): State<AppState>,
    body: Json<cerebras::ChatCompletionRequest>,
) -> Response {
    cerebras::chat_completions(body).await
}

async fn gemini_handler(
    State(_state): State<AppState>,
    path: axum::extract::Path<String>,
    body: Json<gemini::GenerateContentRequest>,
) -> Response {
    gemini::handle_model_action(path, body).await
}

async fn claude_handler(
    State(_state): State<AppState>,
    body: Json<claude::MessagesRequest>,
) -> Response {
    claude::messages(body).await
}

async fn openai_handler(
    State(_state): State<AppState>,
    body: Json<openai::ResponsesRequest>,
) -> Response {
    openai::responses(body).await
}
