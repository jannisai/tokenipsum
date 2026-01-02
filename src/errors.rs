//! Error response generators for different providers.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::config::ErrorType;

/// Generate an error response for a specific provider.
pub fn error_response(error: ErrorType, provider: Provider) -> Response {
    match error {
        ErrorType::Unauthorized => unauthorized(provider),
        ErrorType::RateLimit => rate_limit(provider),
        ErrorType::ServerError => server_error(provider),
        ErrorType::Timeout => timeout(provider),
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Provider {
    Cerebras,
    Gemini,
    Claude,
    OpenAI,
}

fn unauthorized(provider: Provider) -> Response {
    let (status, body) = match provider {
        Provider::Cerebras | Provider::OpenAI => (
            StatusCode::UNAUTHORIZED,
            json!({
                "error": {
                    "message": "Invalid API key provided. You can find your API key at https://platform.example.com/account/api-keys.",
                    "type": "invalid_request_error",
                    "param": null,
                    "code": "invalid_api_key"
                }
            }),
        ),
        Provider::Gemini => (
            StatusCode::UNAUTHORIZED,
            json!({
                "error": {
                    "code": 401,
                    "message": "API key not valid. Please pass a valid API key.",
                    "status": "UNAUTHENTICATED",
                    "details": [{
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "API_KEY_INVALID",
                        "domain": "googleapis.com"
                    }]
                }
            }),
        ),
        Provider::Claude => (
            StatusCode::UNAUTHORIZED,
            json!({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid API key provided."
                }
            }),
        ),
    };

    (status, Json(body)).into_response()
}

fn rate_limit(provider: Provider) -> Response {
    let (status, body, headers) = match provider {
        Provider::Cerebras | Provider::OpenAI => (
            StatusCode::TOO_MANY_REQUESTS,
            json!({
                "error": {
                    "message": "Rate limit reached for requests. Please slow down.",
                    "type": "rate_limit_error",
                    "param": null,
                    "code": "rate_limit_exceeded"
                }
            }),
            vec![
                ("x-ratelimit-limit-requests", "60"),
                ("x-ratelimit-remaining-requests", "0"),
                ("x-ratelimit-reset-requests", "1s"),
                ("retry-after", "1"),
            ],
        ),
        Provider::Gemini => (
            StatusCode::TOO_MANY_REQUESTS,
            json!({
                "error": {
                    "code": 429,
                    "message": "Resource has been exhausted (e.g. check quota).",
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [{
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{
                            "subject": "GenerateContentRequest",
                            "description": "Quota exceeded"
                        }]
                    }]
                }
            }),
            vec![("retry-after", "60")],
        ),
        Provider::Claude => (
            StatusCode::TOO_MANY_REQUESTS,
            json!({
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "Rate limit exceeded. Please retry after 60 seconds."
                }
            }),
            vec![
                ("retry-after", "60"),
                ("x-ratelimit-limit-requests", "60"),
                ("x-ratelimit-remaining-requests", "0"),
            ],
        ),
    };

    let mut response = (status, Json(body)).into_response();
    for (key, value) in headers {
        response
            .headers_mut()
            .insert(key, value.parse().unwrap());
    }
    response
}

fn server_error(provider: Provider) -> Response {
    let (status, body) = match provider {
        Provider::Cerebras | Provider::OpenAI => (
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({
                "error": {
                    "message": "The server had an error while processing your request. Sorry about that!",
                    "type": "server_error",
                    "param": null,
                    "code": "internal_error"
                }
            }),
        ),
        Provider::Gemini => (
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({
                "error": {
                    "code": 500,
                    "message": "An internal error has occurred. Please retry or report in https://developers.generativeai.google/guide/troubleshooting",
                    "status": "INTERNAL"
                }
            }),
        ),
        Provider::Claude => (
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "An unexpected error occurred. Please try again later."
                }
            }),
        ),
    };

    (status, Json(body)).into_response()
}

fn timeout(provider: Provider) -> Response {
    let (status, body) = match provider {
        Provider::Cerebras | Provider::OpenAI => (
            StatusCode::GATEWAY_TIMEOUT,
            json!({
                "error": {
                    "message": "Request timed out. Please try again.",
                    "type": "timeout_error",
                    "param": null,
                    "code": "timeout"
                }
            }),
        ),
        Provider::Gemini => (
            StatusCode::GATEWAY_TIMEOUT,
            json!({
                "error": {
                    "code": 504,
                    "message": "Deadline exceeded while waiting for response.",
                    "status": "DEADLINE_EXCEEDED"
                }
            }),
        ),
        Provider::Claude => (
            StatusCode::GATEWAY_TIMEOUT,
            json!({
                "type": "error",
                "error": {
                    "type": "timeout_error",
                    "message": "Request timed out."
                }
            }),
        ),
    };

    (status, Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unauthorized_responses() {
        let _ = unauthorized(Provider::Cerebras);
        let _ = unauthorized(Provider::Gemini);
        let _ = unauthorized(Provider::Claude);
        let _ = unauthorized(Provider::OpenAI);
    }

    #[test]
    fn test_rate_limit_responses() {
        let resp = rate_limit(Provider::Claude);
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }
}
