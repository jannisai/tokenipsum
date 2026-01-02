#![allow(dead_code)]
//! Anthropic Claude API mock implementation.
//!
//! Generates responses matching the exact structure of the real Claude Messages API.
//!
//! Endpoints:
//! - POST /v1/messages - Non-streaming and streaming

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
use std::time::Duration;
use tokio::time::sleep;

/// Request body for messages endpoint.
#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    #[allow(dead_code)]
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
}

#[derive(Debug, Deserialize)]
pub struct Tool {
    pub name: String,
    #[allow(dead_code)]
    pub description: Option<String>,
    #[allow(dead_code)]
    pub input_schema: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: String,
    pub budget_tokens: u32,
}

/// Non-streaming response.
#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ResponseContent>,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ResponseContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
}

#[derive(Debug, Serialize, Clone)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

/// Main handler for POST /v1/messages
pub async fn messages(Json(req): Json<MessagesRequest>) -> Response {
    let gen = ContentGenerator::new();
    let wants_tools = req.tools.is_some() && should_call_tool(&req);
    let wants_thinking = req.thinking.is_some();

    if req.stream {
        stream_response(req, gen, wants_tools, wants_thinking).await
    } else {
        non_stream_response(req, gen, wants_tools, wants_thinking)
    }
}

/// Decide if we should generate a tool call response.
fn should_call_tool(req: &MessagesRequest) -> bool {
    if let Some(last) = req.messages.last() {
        let text = match &last.content {
            MessageContent::Text(t) => Some(t.as_str()),
            MessageContent::Blocks(blocks) => blocks.iter().find_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            }),
        };
        if let Some(text) = text {
            let lower = text.to_lowercase();
            return lower.contains("weather")
                || lower.contains("search")
                || lower.contains("calculate")
                || lower.contains("what is")
                || lower.contains("find");
        }
    }
    false
}

/// Generate a fake message ID.
fn generate_message_id(gen: &mut ContentGenerator) -> String {
    format!("msg_{}", gen.tool_call_id())
}

/// Generate a fake tool use ID.
fn generate_tool_use_id(gen: &mut ContentGenerator) -> String {
    format!("toolu_{}", gen.tool_call_id())
}

/// Generate a fake thinking signature.
fn generate_signature(gen: &mut ContentGenerator) -> String {
    // Fake base64-encoded signature
    let mut sig = String::with_capacity(200);
    sig.push_str("EtUB");
    for _ in 0..40 {
        sig.push_str(&format!(
            "{:02x}",
            gen.tool_call_id().chars().next().unwrap_or('0') as u8
        ));
    }
    sig.push_str("==");
    sig
}

/// Extract input tokens from messages.
fn count_input_tokens(req: &MessagesRequest) -> u32 {
    let system_tokens = req
        .system
        .as_ref()
        .map(|s| ContentGenerator::estimate_tokens(s))
        .unwrap_or(0);

    let message_tokens: u32 = req
        .messages
        .iter()
        .map(|m| match &m.content {
            MessageContent::Text(t) => ContentGenerator::estimate_tokens(t),
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .map(|b| match b {
                    ContentBlock::Text { text } => ContentGenerator::estimate_tokens(text),
                    ContentBlock::Thinking { thinking, .. } => {
                        ContentGenerator::estimate_tokens(thinking)
                    }
                    _ => 10,
                })
                .sum(),
        })
        .sum();

    system_tokens + message_tokens
}

/// Generate non-streaming response.
fn non_stream_response(
    req: MessagesRequest,
    mut gen: ContentGenerator,
    wants_tools: bool,
    wants_thinking: bool,
) -> Response {
    let id = generate_message_id(&mut gen);
    let input_tokens = count_input_tokens(&req);

    let mut content = Vec::new();
    let mut output_tokens = 0u32;

    // Add thinking block if requested
    if wants_thinking {
        let thinking_text = gen.paragraph();
        output_tokens += ContentGenerator::estimate_tokens(&thinking_text);
        content.push(ResponseContent::Thinking {
            thinking: thinking_text,
            signature: generate_signature(&mut gen),
        });
    }

    let stop_reason = if wants_tools {
        let tool = req.tools.as_ref().and_then(|t| t.first());
        let tool_name = tool
            .map(|t| t.name.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let arg_value = extract_argument(&req);

        output_tokens += 50; // Approximate tool call tokens
        content.push(ResponseContent::ToolUse {
            id: generate_tool_use_id(&mut gen),
            name: tool_name,
            input: json!({ "location": arg_value }),
        });
        "tool_use"
    } else {
        let text = gen.paragraph();
        output_tokens += ContentGenerator::estimate_tokens(&text);
        content.push(ResponseContent::Text { text });
        "end_turn"
    };

    let response = MessagesResponse {
        id,
        response_type: "message",
        role: "assistant",
        model: req.model,
        content,
        stop_reason: stop_reason.to_string(),
        stop_sequence: None,
        usage: Usage {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens: Some(0),
            cache_read_input_tokens: Some(0),
        },
    };

    Json(response).into_response()
}

/// Generate streaming SSE response.
async fn stream_response(
    req: MessagesRequest,
    mut gen: ContentGenerator,
    wants_tools: bool,
    wants_thinking: bool,
) -> Response {
    let id = generate_message_id(&mut gen);
    let model = req.model.clone();
    let input_tokens = count_input_tokens(&req);

    let mut events: Vec<String> = Vec::new();
    let mut output_tokens = 0u32;
    let mut content_index = 0u32;

    // message_start
    events.push(format!(
        "event: message_start\ndata: {}\n\n",
        json!({
            "type": "message_start",
            "message": {
                "id": &id,
                "type": "message",
                "role": "assistant",
                "model": &model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": input_tokens,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 1
                }
            }
        })
    ));

    // Thinking block if requested
    if wants_thinking {
        let thinking_text = gen.paragraph();
        let signature = generate_signature(&mut gen);
        output_tokens += ContentGenerator::estimate_tokens(&thinking_text);

        // content_block_start for thinking
        events.push(format!(
            "event: content_block_start\ndata: {}\n\n",
            json!({
                "type": "content_block_start",
                "index": content_index,
                "content_block": { "type": "thinking", "thinking": "", "signature": "" }
            })
        ));

        // Stream thinking in chunks
        let words: Vec<&str> = thinking_text.split_whitespace().collect();
        for chunk in words.chunks(3) {
            events.push(format!(
                "event: content_block_delta\ndata: {}\n\n",
                json!({
                    "type": "content_block_delta",
                    "index": content_index,
                    "delta": { "type": "thinking_delta", "thinking": chunk.join(" ") + " " }
                })
            ));
        }

        // Signature delta
        events.push(format!(
            "event: content_block_delta\ndata: {}\n\n",
            json!({
                "type": "content_block_delta",
                "index": content_index,
                "delta": { "type": "signature_delta", "signature": signature }
            })
        ));

        events.push(format!(
            "event: content_block_stop\ndata: {}\n\n",
            json!({ "type": "content_block_stop", "index": content_index })
        ));

        content_index += 1;
    }

    let stop_reason = if wants_tools {
        let tool = req.tools.as_ref().and_then(|t| t.first());
        let tool_name = tool
            .map(|t| t.name.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let arg_value = extract_argument(&req);
        let tool_id = generate_tool_use_id(&mut gen);
        output_tokens += 50;

        // content_block_start for tool_use
        events.push(format!(
            "event: content_block_start\ndata: {}\n\n",
            json!({
                "type": "content_block_start",
                "index": content_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": {}
                }
            })
        ));

        // input_json_delta
        events.push(format!(
            "event: content_block_delta\ndata: {}\n\n",
            json!({
                "type": "content_block_delta",
                "index": content_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json!({ "location": arg_value }).to_string()
                }
            })
        ));

        events.push(format!(
            "event: content_block_stop\ndata: {}\n\n",
            json!({ "type": "content_block_stop", "index": content_index })
        ));

        "tool_use"
    } else {
        // Text content
        events.push(format!(
            "event: content_block_start\ndata: {}\n\n",
            json!({
                "type": "content_block_start",
                "index": content_index,
                "content_block": { "type": "text", "text": "" }
            })
        ));

        let max_tokens = req.max_tokens.min(100) as usize;
        let chunks = gen.stream_chunks(max_tokens);
        for (i, chunk) in chunks.iter().enumerate() {
            let prefix = if i > 0 { " " } else { "" };
            let text = format!("{prefix}{chunk}");
            output_tokens += ContentGenerator::estimate_tokens(&text);

            events.push(format!(
                "event: content_block_delta\ndata: {}\n\n",
                json!({
                    "type": "content_block_delta",
                    "index": content_index,
                    "delta": { "type": "text_delta", "text": text }
                })
            ));
        }

        events.push(format!(
            "event: content_block_stop\ndata: {}\n\n",
            json!({ "type": "content_block_stop", "index": content_index })
        ));

        "end_turn"
    };

    // message_delta with stop_reason and usage
    events.push(format!(
        "event: message_delta\ndata: {}\n\n",
        json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason, "stop_sequence": null },
            "usage": {
                "input_tokens": input_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": output_tokens
            }
        })
    ));

    // message_stop
    events.push("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string());

    // Build the stream with delays
    let stream = stream::iter(events)
        .then(|event| async move {
            sleep(Duration::from_millis(15)).await;
            event
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

/// Extract an argument value from the user message.
fn extract_argument(req: &MessagesRequest) -> String {
    req.messages
        .last()
        .and_then(|m| match &m.content {
            MessageContent::Text(t) => Some(t.as_str()),
            MessageContent::Blocks(blocks) => blocks.iter().find_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            }),
        })
        .map_or_else(
            || "unknown".to_string(),
            |text| {
                text.split_whitespace()
                    .rfind(|w| w.len() > 2)
                    .unwrap_or("unknown")
                    .trim_matches(|ch: char| !ch.is_alphanumeric())
                    .to_string()
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_text_message() {
        let json = r#"{
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}]
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "claude-haiku-4-5-20251001");
        assert_eq!(req.max_tokens, 100);
    }

    #[test]
    fn test_deserialize_block_message() {
        let json = r#"{
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            }]
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(
            &req.messages[0].content,
            MessageContent::Blocks(_)
        ));
    }

    #[test]
    fn test_deserialize_with_thinking() {
        let json = r#"{
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 2000,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "What is 2+2?"}]
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert!(req.thinking.is_some());
        assert_eq!(req.thinking.unwrap().budget_tokens, 1024);
    }

    #[test]
    fn test_serialize_response() {
        let response = MessagesResponse {
            id: "msg_test".to_string(),
            response_type: "message",
            role: "assistant",
            model: "claude-haiku-4-5-20251001".to_string(),
            content: vec![ResponseContent::Text {
                text: "Hello!".to_string(),
            }],
            stop_reason: "end_turn".to_string(),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: Some(0),
                cache_read_input_tokens: Some(0),
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("Hello!"));
        assert!(json.contains("end_turn"));
    }

    #[test]
    fn test_serialize_tool_use() {
        let response = MessagesResponse {
            id: "msg_test".to_string(),
            response_type: "message",
            role: "assistant",
            model: "claude-haiku-4-5-20251001".to_string(),
            content: vec![ResponseContent::ToolUse {
                id: "toolu_123".to_string(),
                name: "get_weather".to_string(),
                input: json!({"location": "Tokyo"}),
            }],
            stop_reason: "tool_use".to_string(),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 50,
                output_tokens: 30,
                cache_creation_input_tokens: Some(0),
                cache_read_input_tokens: Some(0),
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("tool_use"));
        assert!(json.contains("get_weather"));
        assert!(json.contains("Tokyo"));
    }

    #[test]
    fn test_serialize_thinking() {
        let response = MessagesResponse {
            id: "msg_test".to_string(),
            response_type: "message",
            role: "assistant",
            model: "claude-haiku-4-5-20251001".to_string(),
            content: vec![
                ResponseContent::Thinking {
                    thinking: "Let me think...".to_string(),
                    signature: "EtUB...==".to_string(),
                },
                ResponseContent::Text {
                    text: "The answer is 4.".to_string(),
                },
            ],
            stop_reason: "end_turn".to_string(),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 50,
                output_tokens: 100,
                cache_creation_input_tokens: Some(0),
                cache_read_input_tokens: Some(0),
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("thinking"));
        assert!(json.contains("signature"));
    }

    #[test]
    fn test_should_call_tool() {
        let req = MessagesRequest {
            model: "test".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("What is the weather in Tokyo?".to_string()),
            }],
            max_tokens: 100,
            stream: false,
            system: None,
            temperature: None,
            tools: Some(vec![Tool {
                name: "get_weather".to_string(),
                description: None,
                input_schema: None,
            }]),
            thinking: None,
        };

        assert!(should_call_tool(&req));
    }

    #[test]
    fn test_should_not_call_tool() {
        let req = MessagesRequest {
            model: "test".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("Hello there!".to_string()),
            }],
            max_tokens: 100,
            stream: false,
            system: None,
            temperature: None,
            tools: Some(vec![]),
            thinking: None,
        };

        assert!(!should_call_tool(&req));
    }

    #[test]
    fn test_thinking_config_deserialize() {
        let json = r#"{
            "model": "claude",
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "Hello"}]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert!(req.thinking.is_some());
        assert_eq!(req.thinking.unwrap().budget_tokens, 1024);
    }

    #[test]
    fn test_extract_argument() {
        let req = MessagesRequest {
            model: "test".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("What is the weather in Paris?".to_string()),
            }],
            max_tokens: 100,
            stream: false,
            system: None,
            temperature: None,
            tools: None,
            thinking: None,
        };

        let arg = extract_argument(&req);
        assert_eq!(arg, "Paris");
    }

    #[test]
    fn test_count_input_tokens() {
        let req = MessagesRequest {
            model: "test".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("Hello world".to_string()),
            }],
            max_tokens: 100,
            stream: false,
            system: Some("You are helpful.".to_string()),
            temperature: None,
            tools: None,
            thinking: None,
        };

        let tokens = count_input_tokens(&req);
        assert!(tokens > 0);
    }

    #[test]
    fn test_generate_message_id() {
        let mut gen = ContentGenerator::with_seed(42);
        let id = generate_message_id(&mut gen);
        assert!(id.starts_with("msg_"));
    }

    #[test]
    fn test_generate_tool_use_id() {
        let mut gen = ContentGenerator::with_seed(42);
        let id = generate_tool_use_id(&mut gen);
        assert!(id.starts_with("toolu_"));
    }

    #[test]
    fn test_generate_signature() {
        let mut gen = ContentGenerator::with_seed(42);
        let sig = generate_signature(&mut gen);
        assert!(!sig.is_empty());
    }

    #[tokio::test]
    async fn test_messages_non_streaming() {
        let req = MessagesRequest {
            model: "claude-haiku".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("Hello".to_string()),
            }],
            max_tokens: 100,
            stream: false,
            system: None,
            temperature: None,
            tools: None,
            thinking: None,
        };

        let response = messages(Json(req)).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_messages_streaming() {
        let req = MessagesRequest {
            model: "claude-haiku".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("Hello".to_string()),
            }],
            max_tokens: 100,
            stream: true,
            system: None,
            temperature: None,
            tools: None,
            thinking: None,
        };

        let response = messages(Json(req)).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
    }

    #[tokio::test]
    async fn test_messages_with_thinking() {
        let req = MessagesRequest {
            model: "claude-haiku".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("What is 2+2?".to_string()),
            }],
            max_tokens: 2000,
            stream: false,
            system: None,
            temperature: None,
            tools: None,
            thinking: Some(ThinkingConfig {
                thinking_type: "enabled".to_string(),
                budget_tokens: 1024,
            }),
        };

        let response = messages(Json(req)).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_messages_with_tools() {
        let req = MessagesRequest {
            model: "claude-haiku".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: MessageContent::Text("What is the weather in Tokyo?".to_string()),
            }],
            max_tokens: 100,
            stream: false,
            system: None,
            temperature: None,
            tools: Some(vec![Tool {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                input_schema: None,
            }]),
            thinking: None,
        };

        let response = messages(Json(req)).await;
        assert_eq!(response.status(), StatusCode::OK);
    }
}
