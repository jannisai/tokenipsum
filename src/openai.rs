#![allow(dead_code)]
//! OpenAI Responses API mock implementation.
//!
//! Generates responses matching the exact structure of the new OpenAI Responses API.
//!
//! Endpoints:
//! - POST /v1/responses - Non-streaming and streaming

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

/// Request body for responses endpoint.
#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: InputType,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub text: Option<TextConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InputType {
    Text(String),
    Messages(Vec<InputMessage>),
}

#[derive(Debug, Deserialize)]
pub struct InputMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct ReasoningConfig {
    pub effort: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TextConfig {
    pub format: Option<TextFormat>,
    pub verbosity: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TextFormat {
    #[serde(rename = "type")]
    pub format_type: String,
}

/// Non-streaming response.
#[derive(Debug, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub status: &'static str,
    pub background: bool,
    pub model: String,
    pub output: Vec<OutputItem>,
    pub usage: Usage,
    // Optional fields
    pub billing: Option<Billing>,
    pub completed_at: Option<u64>,
    pub error: Option<Value>,
    pub incomplete_details: Option<Value>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub max_tool_calls: Option<u32>,
    pub parallel_tool_calls: bool,
    pub previous_response_id: Option<String>,
    pub reasoning: ReasoningOutput,
    pub service_tier: &'static str,
    pub store: bool,
    pub temperature: f32,
    pub text: TextOutput,
    pub tool_choice: &'static str,
    pub tools: Vec<Value>,
    pub top_p: f32,
    pub truncation: &'static str,
    pub user: Option<String>,
    pub metadata: Value,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        status: &'static str,
        content: Vec<OutputContent>,
        role: &'static str,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        status: &'static str,
        name: String,
        arguments: String,
        call_id: String,
    },
}

#[derive(Debug, Serialize)]
pub struct OutputContent {
    #[serde(rename = "type")]
    pub content_type: &'static str,
    pub annotations: Vec<Value>,
    pub logprobs: Vec<Value>,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub input_tokens_details: TokenDetails,
    pub output_tokens: u32,
    pub output_tokens_details: OutputTokenDetails,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct TokenDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct OutputTokenDetails {
    pub reasoning_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct Billing {
    pub payer: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ReasoningOutput {
    pub effort: Option<String>,
    pub summary: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TextOutput {
    pub format: TextFormatOutput,
    pub verbosity: &'static str,
}

#[derive(Debug, Serialize)]
pub struct TextFormatOutput {
    #[serde(rename = "type")]
    pub format_type: &'static str,
}

/// Main handler for POST /v1/responses
pub async fn responses(Json(req): Json<ResponsesRequest>) -> Response {
    let gen = ContentGenerator::new();
    let wants_tools = req.tools.is_some() && should_call_tool(&req);

    if req.stream {
        stream_response(req, gen, wants_tools).await
    } else {
        non_stream_response(req, gen, wants_tools)
    }
}

/// Decide if we should generate a tool call response.
fn should_call_tool(req: &ResponsesRequest) -> bool {
    let text = extract_input_text(&req.input);
    if let Some(text) = text {
        let lower = text.to_lowercase();
        return lower.contains("weather")
            || lower.contains("search")
            || lower.contains("calculate")
            || lower.contains("what is")
            || lower.contains("find");
    }
    false
}

fn extract_input_text(input: &InputType) -> Option<&str> {
    match input {
        InputType::Text(t) => Some(t.as_str()),
        InputType::Messages(msgs) => msgs.last().and_then(|m| match &m.content {
            MessageContent::Text(t) => Some(t.as_str()),
            MessageContent::Parts(parts) => parts.iter().find_map(|p| p.text.as_deref()),
        }),
    }
}

fn generate_response_id(gen: &mut ContentGenerator) -> String {
    format!("resp_{}", gen.tool_call_id())
}

fn generate_message_id(gen: &mut ContentGenerator) -> String {
    format!("msg_{}", gen.tool_call_id())
}

fn generate_call_id(gen: &mut ContentGenerator) -> String {
    format!("call_{}", gen.tool_call_id())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn count_input_tokens(input: &InputType) -> u32 {
    match input {
        InputType::Text(t) => ContentGenerator::estimate_tokens(t),
        InputType::Messages(msgs) => msgs
            .iter()
            .map(|m| match &m.content {
                MessageContent::Text(t) => ContentGenerator::estimate_tokens(t),
                MessageContent::Parts(parts) => parts
                    .iter()
                    .filter_map(|p| p.text.as_ref())
                    .map(|t| ContentGenerator::estimate_tokens(t))
                    .sum(),
            })
            .sum(),
    }
}

/// Generate non-streaming response.
fn non_stream_response(
    req: ResponsesRequest,
    mut gen: ContentGenerator,
    wants_tools: bool,
) -> Response {
    let id = generate_response_id(&mut gen);
    let created_at = now_unix();
    let input_tokens = count_input_tokens(&req.input);

    let (output, output_tokens) = if wants_tools {
        let tool = req.tools.as_ref().and_then(|t| t.first());
        let tool_name = tool.map(|t| t.name.clone()).unwrap_or_else(|| "unknown".to_string());
        let arg_value = extract_input_text(&req.input)
            .map(|t| {
                t.split_whitespace()
                    .filter(|w| w.len() > 2)
                    .last()
                    .unwrap_or("unknown")
                    .to_string()
            })
            .unwrap_or_else(|| "unknown".to_string());

        (
            vec![OutputItem::FunctionCall {
                id: format!("fc_{}", gen.tool_call_id()),
                status: "completed",
                name: tool_name,
                arguments: json!({"location": arg_value}).to_string(),
                call_id: generate_call_id(&mut gen),
            }],
            15u32,
        )
    } else {
        let content = gen.paragraph();
        let tokens = ContentGenerator::estimate_tokens(&content);
        (
            vec![OutputItem::Message {
                id: generate_message_id(&mut gen),
                status: "completed",
                content: vec![OutputContent {
                    content_type: "output_text",
                    annotations: vec![],
                    logprobs: vec![],
                    text: content,
                }],
                role: "assistant",
            }],
            tokens,
        )
    };

    let response = ResponsesResponse {
        id,
        object: "response",
        created_at,
        status: "completed",
        background: false,
        model: req.model,
        output,
        usage: Usage {
            input_tokens,
            input_tokens_details: TokenDetails { cached_tokens: 0 },
            output_tokens,
            output_tokens_details: OutputTokenDetails { reasoning_tokens: 0 },
            total_tokens: input_tokens + output_tokens,
        },
        billing: Some(Billing { payer: "openai" }),
        completed_at: Some(now_unix()),
        error: None,
        incomplete_details: None,
        instructions: req.instructions,
        max_output_tokens: req.max_output_tokens,
        max_tool_calls: None,
        parallel_tool_calls: true,
        previous_response_id: None,
        reasoning: ReasoningOutput {
            effort: req.reasoning.as_ref().and_then(|r| r.effort.clone()),
            summary: None,
        },
        service_tier: "default",
        store: req.store.unwrap_or(true),
        temperature: req.temperature.unwrap_or(1.0),
        text: TextOutput {
            format: TextFormatOutput { format_type: "text" },
            verbosity: "medium",
        },
        tool_choice: "auto",
        tools: vec![],
        top_p: req.top_p.unwrap_or(1.0),
        truncation: "disabled",
        user: None,
        metadata: json!({}),
    };

    Json(response).into_response()
}

/// Generate streaming SSE response.
async fn stream_response(
    req: ResponsesRequest,
    mut gen: ContentGenerator,
    wants_tools: bool,
) -> Response {
    let id = generate_response_id(&mut gen);
    let model = req.model.clone();
    let created_at = now_unix();
    let input_tokens = count_input_tokens(&req.input);

    let mut events: Vec<String> = Vec::new();
    let mut seq = 0u32;

    // Helper to create event
    let event = |name: &str, data: Value| -> String {
        format!("event: {}\ndata: {}\n\n", name, data)
    };

    // response.created
    events.push(event(
        "response.created",
        json!({
            "type": "response.created",
            "sequence_number": seq,
            "response": {
                "id": &id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "model": &model,
                "output": [],
                "usage": null
            }
        }),
    ));
    seq += 1;

    // response.in_progress
    events.push(event(
        "response.in_progress",
        json!({
            "type": "response.in_progress",
            "sequence_number": seq,
            "response": {
                "id": &id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "model": &model,
                "output": []
            }
        }),
    ));
    seq += 1;

    let output_tokens;

    if wants_tools {
        let tool = req.tools.as_ref().and_then(|t| t.first());
        let tool_name = tool.map(|t| t.name.clone()).unwrap_or_else(|| "unknown".to_string());
        let arg_value = extract_input_text(&req.input)
            .map(|t| {
                t.split_whitespace()
                    .filter(|w| w.len() > 2)
                    .last()
                    .unwrap_or("unknown")
                    .to_string()
            })
            .unwrap_or_else(|| "unknown".to_string());
        let fc_id = format!("fc_{}", gen.tool_call_id());
        let call_id = generate_call_id(&mut gen);

        output_tokens = 15u32;

        // output_item.added for function_call
        events.push(event(
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "sequence_number": seq,
                "output_index": 0,
                "item": {
                    "id": &fc_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "name": &tool_name,
                    "arguments": "",
                    "call_id": &call_id
                }
            }),
        ));
        seq += 1;

        // function_call_arguments.delta
        let args = json!({"location": arg_value}).to_string();
        events.push(event(
            "response.function_call_arguments.delta",
            json!({
                "type": "response.function_call_arguments.delta",
                "sequence_number": seq,
                "item_id": &fc_id,
                "output_index": 0,
                "delta": &args
            }),
        ));
        seq += 1;

        // function_call_arguments.done
        events.push(event(
            "response.function_call_arguments.done",
            json!({
                "type": "response.function_call_arguments.done",
                "sequence_number": seq,
                "item_id": &fc_id,
                "output_index": 0,
                "arguments": &args
            }),
        ));
        seq += 1;

        // output_item.done
        events.push(event(
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "sequence_number": seq,
                "output_index": 0,
                "item": {
                    "id": &fc_id,
                    "type": "function_call",
                    "status": "completed",
                    "name": &tool_name,
                    "arguments": &args,
                    "call_id": &call_id
                }
            }),
        ));
        seq += 1;
    } else {
        let msg_id = generate_message_id(&mut gen);
        let content_parts = gen.stream_chunks(req.max_output_tokens.unwrap_or(50) as usize);

        output_tokens = content_parts
            .iter()
            .map(|c| ContentGenerator::estimate_tokens(c))
            .sum::<u32>()
            .max(1);

        // output_item.added
        events.push(event(
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "sequence_number": seq,
                "output_index": 0,
                "item": {
                    "id": &msg_id,
                    "type": "message",
                    "status": "in_progress",
                    "content": [],
                    "role": "assistant"
                }
            }),
        ));
        seq += 1;

        // content_part.added
        events.push(event(
            "response.content_part.added",
            json!({
                "type": "response.content_part.added",
                "sequence_number": seq,
                "item_id": &msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": ""
                }
            }),
        ));
        seq += 1;

        // output_text.delta for each chunk
        let mut full_text = String::new();
        for (i, chunk) in content_parts.iter().enumerate() {
            let delta = if i > 0 {
                format!(" {}", chunk)
            } else {
                chunk.clone()
            };
            full_text.push_str(&delta);

            events.push(event(
                "response.output_text.delta",
                json!({
                    "type": "response.output_text.delta",
                    "sequence_number": seq,
                    "item_id": &msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": &delta,
                    "logprobs": []
                }),
            ));
            seq += 1;
        }

        // output_text.done
        events.push(event(
            "response.output_text.done",
            json!({
                "type": "response.output_text.done",
                "sequence_number": seq,
                "item_id": &msg_id,
                "output_index": 0,
                "content_index": 0,
                "text": &full_text,
                "logprobs": []
            }),
        ));
        seq += 1;

        // content_part.done
        events.push(event(
            "response.content_part.done",
            json!({
                "type": "response.content_part.done",
                "sequence_number": seq,
                "item_id": &msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": &full_text
                }
            }),
        ));
        seq += 1;

        // output_item.done
        events.push(event(
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "sequence_number": seq,
                "output_index": 0,
                "item": {
                    "id": &msg_id,
                    "type": "message",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": &full_text
                    }],
                    "role": "assistant"
                }
            }),
        ));
        seq += 1;
    }

    // response.completed
    events.push(event(
        "response.completed",
        json!({
            "type": "response.completed",
            "sequence_number": seq,
            "response": {
                "id": &id,
                "object": "response",
                "created_at": created_at,
                "status": "completed",
                "completed_at": now_unix(),
                "model": &model,
                "output": [],
                "usage": {
                    "input_tokens": input_tokens,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": output_tokens,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": input_tokens + output_tokens
                }
            }
        }),
    ));

    // Build the stream
    let stream = stream::iter(events)
        .then(|event| async move {
            sleep(Duration::from_millis(10)).await;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_simple_request() {
        let json = r#"{"model": "gpt-4o-mini", "input": "Hello"}"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "gpt-4o-mini");
        assert!(matches!(req.input, InputType::Text(_)));
    }

    #[test]
    fn test_deserialize_messages_request() {
        let json = r#"{
            "model": "gpt-4o-mini",
            "input": [
                {"role": "user", "content": "Hello"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.input, InputType::Messages(_)));
    }

    #[test]
    fn test_deserialize_with_tools() {
        let json = r#"{
            "model": "gpt-4o-mini",
            "input": "What is the weather?",
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "parameters": {}
            }]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        assert!(should_call_tool(&req));
    }

    #[test]
    fn test_serialize_response() {
        let response = ResponsesResponse {
            id: "resp_test".to_string(),
            object: "response",
            created_at: 12345,
            status: "completed",
            background: false,
            model: "gpt-4o-mini".to_string(),
            output: vec![OutputItem::Message {
                id: "msg_test".to_string(),
                status: "completed",
                content: vec![OutputContent {
                    content_type: "output_text",
                    annotations: vec![],
                    logprobs: vec![],
                    text: "Hello!".to_string(),
                }],
                role: "assistant",
            }],
            usage: Usage {
                input_tokens: 10,
                input_tokens_details: TokenDetails { cached_tokens: 0 },
                output_tokens: 5,
                output_tokens_details: OutputTokenDetails { reasoning_tokens: 0 },
                total_tokens: 15,
            },
            billing: Some(Billing { payer: "openai" }),
            completed_at: Some(12345),
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            parallel_tool_calls: true,
            previous_response_id: None,
            reasoning: ReasoningOutput {
                effort: None,
                summary: None,
            },
            service_tier: "default",
            store: true,
            temperature: 1.0,
            text: TextOutput {
                format: TextFormatOutput { format_type: "text" },
                verbosity: "medium",
            },
            tool_choice: "auto",
            tools: vec![],
            top_p: 1.0,
            truncation: "disabled",
            user: None,
            metadata: json!({}),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("Hello!"));
        assert!(json.contains("output_text"));
    }
}
