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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[test]
    fn test_provider_from_path() {
        assert!(matches!(
            provider_from_path("/v1beta/models/gemini:generateContent"),
            Provider::Gemini
        ));
        assert!(matches!(
            provider_from_path("/v1/messages"),
            Provider::Claude
        ));
        assert!(matches!(
            provider_from_path("/v1/responses"),
            Provider::OpenAI
        ));
        assert!(matches!(
            provider_from_path("/v1/chat/completions"),
            Provider::Cerebras
        ));
        assert!(matches!(provider_from_path("/health"), Provider::Cerebras));
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let config = Config::default();
        let state = RuntimeState::new(config);
        let app = create_router(state);

        let response = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"ok");
    }

    #[tokio::test]
    async fn test_cerebras_endpoint() {
        let config = Config::default();
        let state = RuntimeState::new(config);
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "llama-3.3-70b",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_claude_endpoint() {
        let config = Config::default();
        let state = RuntimeState::new(config);
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "claude-3-haiku",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let response = app
            .oneshot(
                Request::post("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_gemini_endpoint() {
        let config = Config::default();
        let state = RuntimeState::new(config);
        let app = create_router(state);

        let body = serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
        });

        let response = app
            .oneshot(
                Request::post("/v1beta/models/gemini-pro:generateContent")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_openai_endpoint() {
        let config = Config::default();
        let state = RuntimeState::new(config);
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "gpt-4o",
            "input": "Hello"
        });

        let response = app
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_required_no_key() {
        let mut config = Config::default();
        config.auth.require_auth = true;
        config.auth.valid_keys = vec!["test-key".to_string()];
        let state = RuntimeState::new(config);
        let app = create_router(state);

        let response = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_required_valid_key() {
        let mut config = Config::default();
        config.auth.require_auth = true;
        config.auth.valid_keys = vec!["test-key".to_string()];
        let state = RuntimeState::new(config);
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::get("/health")
                    .header("authorization", "Bearer test-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_required_invalid_key() {
        let mut config = Config::default();
        config.auth.require_auth = true;
        config.auth.valid_keys = vec!["test-key".to_string()];
        let state = RuntimeState::new(config);
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::get("/health")
                    .header("authorization", "Bearer wrong-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_rate_limit_after_requests() {
        let mut config = Config::default();
        config.rate_limit.fail_after_requests = 2;
        let state = RuntimeState::new(config);

        // First request - should succeed
        let app = create_router(state.clone());
        let response = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Second request - should succeed (this is request #2, equals limit)
        let app = create_router(state.clone());
        let response = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_disabled_provider() {
        let mut config = Config::default();
        config.providers.cerebras = false;
        let state = RuntimeState::new(config);
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "llama",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_forced_error() {
        let mut config = Config::default();
        config.errors.force_error = config::ForceError::ServerError;
        let state = RuntimeState::new(config);
        let app = create_router(state);

        let response = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_forced_timeout_error() {
        let mut config = Config::default();
        config.errors.force_error = config::ForceError::Timeout;
        let state = RuntimeState::new(config);
        let app = create_router(state);

        let response = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    }
}
