#![allow(dead_code)]
//! Cerebras API mock implementation.
//!
//! Generates responses matching the exact structure of the real Cerebras API.

use crate::generator::ContentGenerator;
use axum::{
    body::Body,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::sleep;

/// Request body for chat completions.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    pub max_tokens: Option<u32>,

    pub temperature: Option<f32>,

    pub top_p: Option<f32>,
    pub tools: Option<Vec<Tool>>,

    pub tool_choice: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Deserialize)]
pub struct ToolFunction {
    pub name: String,

    pub description: Option<String>,

    pub parameters: Option<Value>,
}

/// Non-streaming response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    pub time_info: TimeInfo,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: &'static str,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Debug, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct TimeInfo {
    pub queue_time: f64,
    pub prompt_time: f64,
    pub completion_time: f64,
    pub total_time: f64,
    pub created: f64,
}

/// Main handler for /v1/chat/completions
pub async fn chat_completions(Json(req): Json<ChatCompletionRequest>) -> Response {
    let gen = ContentGenerator::new();

    // Check if tool calling is requested
    let wants_tools = req.tools.is_some() && should_call_tool(&req);

    if req.stream {
        stream_response(req, gen, wants_tools).await
    } else {
        non_stream_response(req, gen, wants_tools)
    }
}

/// Decide if we should generate a tool call response.
fn should_call_tool(req: &ChatCompletionRequest) -> bool {
    // Simple heuristic: if the last message mentions something tool-like
    if let Some(last) = req.messages.last() {
        if let Some(content) = &last.content {
            let lower = content.to_lowercase();
            return lower.contains("weather")
                || lower.contains("search")
                || lower.contains("calculate")
                || lower.contains("what is")
                || lower.contains("find");
        }
    }
    false
}

/// Generate non-streaming response.
fn non_stream_response(
    req: ChatCompletionRequest,
    mut gen: ContentGenerator,
    wants_tools: bool,
) -> Response {
    let id = gen.completion_id();
    let created = now_unix();
    let fingerprint = gen.fingerprint();

    let max_tokens = req.max_tokens.unwrap_or(100);
    let content = gen.paragraph();
    let completion_tokens = ContentGenerator::estimate_tokens(&content).min(max_tokens);

    let prompt_tokens: u32 = req
        .messages
        .iter()
        .filter_map(|m| m.content.as_ref())
        .map(|c| ContentGenerator::estimate_tokens(c))
        .sum();

    let (message, finish_reason) = if wants_tools {
        let tool = req.tools.as_ref().and_then(|t| t.first());
        let tool_name = tool.map(|t| t.function.name.clone()).unwrap_or_default();

        // Extract a location or query from the message
        let arg_value = extract_argument(&req);

        (
            ResponseMessage {
                role: "assistant",
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: gen.tool_call_id(),
                    call_type: "function",
                    function: FunctionCall {
                        name: tool_name,
                        arguments: json!({ "location": arg_value }).to_string(),
                    },
                }]),
            },
            "tool_calls",
        )
    } else {
        (
            ResponseMessage {
                role: "assistant",
                content: Some(content),
                tool_calls: None,
            },
            "stop",
        )
    };

    let response = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model: req.model,
        system_fingerprint: fingerprint,
        choices: vec![Choice {
            index: 0,
            message,
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
        },
        time_info: TimeInfo {
            queue_time: 0.025,
            prompt_time: 0.003,
            completion_time: 0.005,
            total_time: 0.035,
            created: now_unix() as f64,
        },
    };

    Json(response).into_response()
}

/// Generate streaming SSE response.
async fn stream_response(
    req: ChatCompletionRequest,
    mut gen: ContentGenerator,
    wants_tools: bool,
) -> Response {
    let id = gen.completion_id();
    let model = req.model.clone();
    let fingerprint = gen.fingerprint();
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
    let max_tokens = req.max_tokens.unwrap_or(50) as usize;

    let prompt_tokens: u32 = req
        .messages
        .iter()
        .filter_map(|m| m.content.as_ref())
        .map(|c| ContentGenerator::estimate_tokens(c))
        .sum();

    // Generate chunks
    let chunks = if wants_tools {
        generate_tool_chunks(&req, &mut gen, &id, &model, &fingerprint)
    } else {
        generate_content_chunks(&mut gen, max_tokens, &id, &model, &fingerprint)
    };

    let finish_reason = if wants_tools { "tool_calls" } else { "stop" };

    // Calculate total completion tokens
    let completion_tokens: u32 = chunks
        .iter()
        .filter_map(|c| {
            c.get("choices")
                .and_then(|ch| ch.get(0))
                .and_then(|ch| ch.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|c| c.as_str())
                .map(ContentGenerator::estimate_tokens)
        })
        .sum::<u32>()
        .max(1);

    // Build the stream
    let created = now_unix();
    let stream = stream::iter(chunks)
        .chain(stream::once(async move {
            // Final chunk with finish_reason and usage
            let mut final_chunk = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "system_fingerprint": fingerprint,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason
                }]
            });

            if include_usage {
                final_chunk["usage"] = json!({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": { "cached_tokens": 0 }
                });
                final_chunk["time_info"] = json!({
                    "queue_time": 0.025,
                    "prompt_time": 0.003,
                    "completion_time": 0.005,
                    "total_time": 0.035,
                    "created": now_unix() as f64
                });
            }

            final_chunk
        }))
        .then(|chunk| async move {
            // Add small delay for realistic streaming
            sleep(Duration::from_millis(15)).await;
            format!("data: {chunk}\n\n")
        })
        .chain(stream::once(async { "data: [DONE]\n\n".to_string() }))
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

/// Generate content chunks for streaming.
fn generate_content_chunks(
    gen: &mut ContentGenerator,
    max_tokens: usize,
    id: &str,
    model: &str,
    fingerprint: &str,
) -> Vec<Value> {
    let created = now_unix();

    // First chunk: role
    let mut chunks = vec![json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": fingerprint,
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant" }
        }]
    })];

    // Content chunks
    let content_parts = gen.stream_chunks(max_tokens);
    for (i, content) in content_parts.into_iter().enumerate() {
        let prefix = if i > 0 { " " } else { "" };
        chunks.push(json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": fingerprint,
            "choices": [{
                "index": 0,
                "delta": { "content": format!("{}{}", prefix, content) }
            }]
        }));
    }

    chunks
}

/// Generate tool call chunks for streaming.
fn generate_tool_chunks(
    req: &ChatCompletionRequest,
    gen: &mut ContentGenerator,
    id: &str,
    model: &str,
    fingerprint: &str,
) -> Vec<Value> {
    let created = now_unix();

    let tool = req.tools.as_ref().and_then(|t| t.first());
    let tool_name = tool.map(|t| t.function.name.clone()).unwrap_or_default();
    let arg_value = extract_argument(req);
    let tool_call_id = gen.tool_call_id();

    vec![
        // First chunk: role
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": fingerprint,
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant" }
            }]
        }),
        // Tool call chunk
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": fingerprint,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json!({ "location": arg_value }).to_string()
                        }
                    }]
                }
            }]
        }),
    ]
}

/// Extract an argument value from the user message.
fn extract_argument(req: &ChatCompletionRequest) -> String {
    req.messages
        .last()
        .and_then(|m| m.content.as_ref())
        .map_or_else(
            || "unknown".to_string(),
            |c| {
                // Try to extract a location or query
                // Simple: take last word that might be a proper noun
                c.split_whitespace()
                    .filter(|w| w.len() > 2)
                    .next_back()
                    .unwrap_or("unknown")
                    .trim_matches(|ch: char| !ch.is_alphanumeric())
                    .to_string()
            },
        )
}

/// Get current unix timestamp.
fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_request() {
        let json = r#"{
            "model": "llama-3.3-70b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "max_tokens": 100
        }"#;

        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "llama-3.3-70b");
        assert!(req.stream);
        assert_eq!(req.max_tokens, Some(100));
    }

    #[test]
    fn test_serialize_response() {
        let response = ChatCompletionResponse {
            id: "test-id".to_string(),
            object: "chat.completion",
            created: 1234567890,
            model: "test-model".to_string(),
            system_fingerprint: "fp_test".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant",
                    content: Some("Hello!".to_string()),
                    tool_calls: None,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
            },
            time_info: TimeInfo {
                queue_time: 0.01,
                prompt_time: 0.02,
                completion_time: 0.03,
                total_time: 0.06,
                created: 1234567890.0,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("chat.completion"));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_should_call_tool() {
        let req = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: Some("What is the weather in Tokyo?".to_string()),
            }],
            stream: false,
            stream_options: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: Some(vec![]),
            tool_choice: None,
        };

        assert!(should_call_tool(&req));
    }
}
