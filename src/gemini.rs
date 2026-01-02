#![allow(dead_code)]
//! Google Gemini API mock implementation.
//!
//! Generates responses matching the exact structure of the real Gemini API.
//!
//! Endpoints:
//! - POST /v1beta/models/{model}:generateContent - Non-streaming
//! - POST /v1beta/models/{model}:streamGenerateContent?alt=sse - Streaming

use crate::generator::ContentGenerator;
use axum::{
    body::Body,
    extract::Path,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time::sleep;

/// Request body for generateContent.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    pub contents: Vec<Content>,
    #[serde(default)]
    pub system_instruction: Option<Content>,
    #[serde(default)]
    pub generation_config: Option<GenerationConfig>,
    #[serde(default)]
    pub tools: Option<Vec<ToolDeclaration>>,
    #[serde(default)]
    pub tool_config: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    pub role: Option<String>,
    pub parts: Vec<Part>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    pub text: Option<String>,
    pub function_call: Option<Value>,
    pub function_response: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDeclaration {
    pub function_declarations: Option<Vec<FunctionDeclaration>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

/// Non-streaming response.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    pub candidates: Vec<Candidate>,
    pub usage_metadata: UsageMetadata,
    pub model_version: String,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub content: ResponseContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    pub index: u32,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ResponseContent {
    pub parts: Vec<ResponsePart>,
    pub role: String,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ResponsePart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCall {
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    pub prompt_token_count: u32,
    pub candidates_token_count: u32,
    pub total_token_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content_token_count: Option<u32>,
}

/// Unified handler for /v1beta/models/{model_action}
/// Parses model:action format and dispatches accordingly.
pub async fn handle_model_action(
    Path(model_action): Path<String>,
    Json(req): Json<GenerateContentRequest>,
) -> Response {
    // Parse "model:action" format
    let (model, action) = match model_action.rsplit_once(':') {
        Some((m, a)) => (m.to_string(), a),
        None => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from("Invalid path: expected model:action format"))
                .unwrap();
        }
    };

    let gen = ContentGenerator::new();
    let wants_tools = should_call_tool(&req);

    match action {
        "generateContent" => non_stream_response(model, req, gen, wants_tools),
        "streamGenerateContent" => stream_response(model, req, gen, wants_tools).await,
        _ => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from(format!("Unknown action: {action}")))
            .unwrap(),
    }
}

/// Decide if we should generate a tool call response.
fn should_call_tool(req: &GenerateContentRequest) -> bool {
    if req.tools.is_none() {
        return false;
    }

    // Check last content for tool-triggering keywords
    if let Some(last) = req.contents.last() {
        for part in &last.parts {
            if let Some(text) = &part.text {
                let lower = text.to_lowercase();
                return lower.contains("weather")
                    || lower.contains("search")
                    || lower.contains("calculate")
                    || lower.contains("what is")
                    || lower.contains("find");
            }
        }
    }
    false
}

/// Generate non-streaming response.
fn non_stream_response(
    model: String,
    req: GenerateContentRequest,
    mut gen: ContentGenerator,
    wants_tools: bool,
) -> Response {
    let max_tokens = req
        .generation_config
        .as_ref()
        .and_then(|c| c.max_output_tokens)
        .unwrap_or(100);

    let prompt_tokens: u32 = req
        .contents
        .iter()
        .flat_map(|c| &c.parts)
        .filter_map(|p| p.text.as_ref())
        .map(|t| ContentGenerator::estimate_tokens(t))
        .sum();

    let (parts, finish_reason, completion_tokens) = if wants_tools {
        let func_name = get_first_function_name(&req);
        let arg_value = extract_argument(&req);

        (
            vec![ResponsePart {
                text: None,
                function_call: Some(FunctionCall {
                    name: func_name,
                    args: json!({ "location": arg_value }),
                }),
            }],
            "STOP",
            12u32,
        )
    } else {
        let content = gen.paragraph();
        let tokens = ContentGenerator::estimate_tokens(&content).min(max_tokens);
        (
            vec![ResponsePart {
                text: Some(content),
                function_call: None,
            }],
            "STOP",
            tokens,
        )
    };

    let response = GenerateContentResponse {
        candidates: vec![Candidate {
            content: ResponseContent {
                parts,
                role: "model".to_string(),
            },
            finish_reason: Some(finish_reason.to_string()),
            index: 0,
        }],
        usage_metadata: UsageMetadata {
            prompt_token_count: prompt_tokens,
            candidates_token_count: completion_tokens,
            total_token_count: prompt_tokens + completion_tokens,
            cached_content_token_count: None,
        },
        model_version: model,
    };

    Json(response).into_response()
}

/// Generate streaming SSE response.
async fn stream_response(
    model: String,
    req: GenerateContentRequest,
    mut gen: ContentGenerator,
    wants_tools: bool,
) -> Response {
    let max_tokens = req
        .generation_config
        .as_ref()
        .and_then(|c| c.max_output_tokens)
        .unwrap_or(50) as usize;

    let prompt_tokens: u32 = req
        .contents
        .iter()
        .flat_map(|c| &c.parts)
        .filter_map(|p| p.text.as_ref())
        .map(|t| ContentGenerator::estimate_tokens(t))
        .sum();

    // Generate chunks
    let chunks: Vec<Value> = if wants_tools {
        let func_name = get_first_function_name(&req);
        let arg_value = extract_argument(&req);

        vec![
            // Single chunk with function call
            json!({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "functionCall": {
                                "name": func_name,
                                "args": { "location": arg_value }
                            }
                        }],
                        "role": "model"
                    },
                    "index": 0
                }],
                "usageMetadata": {
                    "promptTokenCount": prompt_tokens,
                    "candidatesTokenCount": 12,
                    "totalTokenCount": prompt_tokens + 12
                },
                "modelVersion": &model
            }),
            // Final chunk with finish reason
            json!({
                "candidates": [{
                    "content": {
                        "parts": [],
                        "role": "model"
                    },
                    "finishReason": "STOP",
                    "index": 0
                }],
                "usageMetadata": {
                    "promptTokenCount": prompt_tokens,
                    "candidatesTokenCount": 12,
                    "totalTokenCount": prompt_tokens + 12
                },
                "modelVersion": &model
            }),
        ]
    } else {
        let content_parts = gen.stream_chunks(max_tokens);
        let mut total_tokens = 0u32;

        let mut result: Vec<Value> = content_parts
            .into_iter()
            .enumerate()
            .map(|(i, content)| {
                let prefix = if i > 0 { " " } else { "" };
                let text = format!("{}{}", prefix, content);
                total_tokens += ContentGenerator::estimate_tokens(&text);

                json!({
                    "candidates": [{
                        "content": {
                            "parts": [{ "text": text }],
                            "role": "model"
                        },
                        "index": 0
                    }],
                    "usageMetadata": {
                        "promptTokenCount": prompt_tokens,
                        "candidatesTokenCount": total_tokens,
                        "totalTokenCount": prompt_tokens + total_tokens
                    },
                    "modelVersion": &model
                })
            })
            .collect();

        // Add final chunk with finish reason
        result.push(json!({
            "candidates": [{
                "content": {
                    "parts": [],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": prompt_tokens,
                "candidatesTokenCount": total_tokens,
                "totalTokenCount": prompt_tokens + total_tokens
            },
            "modelVersion": &model
        }));

        result
    };

    // Build the SSE stream
    let stream = stream::iter(chunks)
        .then(|chunk| async move {
            sleep(Duration::from_millis(15)).await;
            format!("data: {}\n\n", chunk)
        })
        .map(Ok::<_, std::convert::Infallible>);

    let body = Body::from_stream(stream);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap()
}

/// Get the first function name from tools.
fn get_first_function_name(req: &GenerateContentRequest) -> String {
    req.tools
        .as_ref()
        .and_then(|tools| tools.first())
        .and_then(|t| t.function_declarations.as_ref())
        .and_then(|fns| fns.first())
        .map(|f| f.name.clone())
        .unwrap_or_else(|| "unknown_function".to_string())
}

/// Extract an argument value from the user message.
fn extract_argument(req: &GenerateContentRequest) -> String {
    req.contents
        .last()
        .and_then(|c| c.parts.first())
        .and_then(|p| p.text.as_ref())
        .map(|text| {
            text.split_whitespace()
                .filter(|w| w.len() > 2)
                .last()
                .unwrap_or("unknown")
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_request() {
        let json = r#"{
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Hello"}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 100
            }
        }"#;

        let req: GenerateContentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.contents.len(), 1);
        assert_eq!(
            req.generation_config.as_ref().unwrap().max_output_tokens,
            Some(100)
        );
    }

    #[test]
    fn test_serialize_response() {
        let response = GenerateContentResponse {
            candidates: vec![Candidate {
                content: ResponseContent {
                    parts: vec![ResponsePart {
                        text: Some("Hello!".to_string()),
                        function_call: None,
                    }],
                    role: "model".to_string(),
                },
                finish_reason: Some("STOP".to_string()),
                index: 0,
            }],
            usage_metadata: UsageMetadata {
                prompt_token_count: 10,
                candidates_token_count: 5,
                total_token_count: 15,
                cached_content_token_count: None,
            },
            model_version: "gemini-2.0-flash".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("Hello!"));
        assert!(json.contains("promptTokenCount"));
        assert!(json.contains("candidatesTokenCount"));
    }

    #[test]
    fn test_should_call_tool() {
        let req = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![Part {
                    text: Some("What is the weather in Tokyo?".to_string()),
                    function_call: None,
                    function_response: None,
                }],
            }],
            system_instruction: None,
            generation_config: None,
            tools: Some(vec![ToolDeclaration {
                function_declarations: Some(vec![FunctionDeclaration {
                    name: "get_weather".to_string(),
                    description: None,
                    parameters: None,
                }]),
            }]),
            tool_config: None,
        };

        assert!(should_call_tool(&req));
    }

    #[test]
    fn test_function_call_response() {
        let response = GenerateContentResponse {
            candidates: vec![Candidate {
                content: ResponseContent {
                    parts: vec![ResponsePart {
                        text: None,
                        function_call: Some(FunctionCall {
                            name: "get_weather".to_string(),
                            args: json!({"location": "Tokyo"}),
                        }),
                    }],
                    role: "model".to_string(),
                },
                finish_reason: Some("STOP".to_string()),
                index: 0,
            }],
            usage_metadata: UsageMetadata {
                prompt_token_count: 10,
                candidates_token_count: 12,
                total_token_count: 22,
                cached_content_token_count: None,
            },
            model_version: "gemini-2.0-flash".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("functionCall"));
        assert!(json.contains("get_weather"));
        assert!(json.contains("Tokyo"));
    }
}
