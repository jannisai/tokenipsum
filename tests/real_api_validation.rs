//! Real API validation tests.
//!
//! These tests compare the structure of mock responses against real API responses.
//! They require API keys set in environment variables.
//!
//! Run with: cargo test --test real_api_validation -- --ignored
//!
//! Set these env vars (or use .env file):
//! - CEREBRAS_API_KEY
//! - GEMINI_API_KEY
//! - ANTHROPIC_API_KEY

use serde_json::Value;
use std::collections::HashSet;

/// Extract all keys from a JSON value recursively, with path prefixes.
fn extract_keys(value: &Value, prefix: &str) -> HashSet<String> {
    let mut keys = HashSet::new();

    match value {
        Value::Object(map) => {
            for (k, v) in map {
                let path = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{}.{}", prefix, k)
                };
                keys.insert(path.clone());
                keys.extend(extract_keys(v, &path));
            }
        }
        Value::Array(arr) => {
            if let Some(first) = arr.first() {
                // Use [*] to denote array items
                let path = format!("{}[*]", prefix);
                keys.extend(extract_keys(first, &path));
            }
        }
        _ => {}
    }

    keys
}

/// Compare two JSON structures and report differences.
fn compare_structure(real: &Value, mock: &Value) -> (HashSet<String>, HashSet<String>) {
    let real_keys = extract_keys(real, "");
    let mock_keys = extract_keys(mock, "");

    let missing_in_mock: HashSet<_> = real_keys.difference(&mock_keys).cloned().collect();
    let extra_in_mock: HashSet<_> = mock_keys.difference(&real_keys).cloned().collect();

    (missing_in_mock, extra_in_mock)
}

/// Print comparison results.
fn print_comparison(name: &str, real: &Value, mock: &Value) {
    let (missing, extra) = compare_structure(real, mock);

    println!("\n=== {} Structure Comparison ===", name);
    println!("Real API response keys: {:?}", extract_keys(real, ""));
    println!("Mock API response keys: {:?}", extract_keys(mock, ""));

    if missing.is_empty() && extra.is_empty() {
        println!("✓ Structures match!");
    } else {
        if !missing.is_empty() {
            println!("⚠ Missing in mock: {:?}", missing);
        }
        if !extra.is_empty() {
            println!("ℹ Extra in mock: {:?}", extra);
        }
    }
}

mod cerebras {
    use super::*;

    async fn call_real_api(api_key: &str) -> Result<Value, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let resp = client
            .post("https://api.cerebras.ai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": "llama-3.3-70b",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 10
            }))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp)
    }

    async fn call_mock_api(base_url: &str) -> Result<Value, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/v1/chat/completions", base_url))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": "llama-3.3-70b",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 10
            }))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp)
    }

    #[tokio::test]
    #[ignore = "requires CEREBRAS_API_KEY"]
    async fn validate_non_streaming_structure() {
        let api_key = std::env::var("CEREBRAS_API_KEY").expect("CEREBRAS_API_KEY not set");
        let mock_url =
            std::env::var("MOCK_URL").unwrap_or_else(|_| "http://localhost:8787".to_string());

        let real = call_real_api(&api_key).await.expect("Real API call failed");
        let mock = call_mock_api(&mock_url)
            .await
            .expect("Mock API call failed");

        print_comparison("Cerebras Non-Streaming", &real, &mock);

        // Core fields that must exist
        let _real_keys = extract_keys(&real, "");
        let mock_keys = extract_keys(&mock, "");

        assert!(mock_keys.contains("id"), "Mock missing 'id'");
        assert!(mock_keys.contains("model"), "Mock missing 'model'");
        assert!(mock_keys.contains("choices"), "Mock missing 'choices'");
        assert!(mock_keys.contains("usage"), "Mock missing 'usage'");

        // Check critical nested fields
        let critical = [
            "choices[*].message",
            "choices[*].finish_reason",
            "usage.total_tokens",
        ];
        for key in critical {
            assert!(
                mock_keys.iter().any(|k| k.contains(key)),
                "Mock missing critical key: {}",
                key
            );
        }

        println!("✓ Cerebras validation passed");
    }
}

mod gemini {
    use super::*;

    async fn call_real_api(api_key: &str) -> Result<Value, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={}",
            api_key
        );
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "contents": [{"role": "user", "parts": [{"text": "Say hi"}]}],
                "generationConfig": {"maxOutputTokens": 10}
            }))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp)
    }

    async fn call_mock_api(base_url: &str) -> Result<Value, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!(
                "{}/v1beta/models/gemini-2.0-flash:generateContent",
                base_url
            ))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "contents": [{"role": "user", "parts": [{"text": "Say hi"}]}],
                "generationConfig": {"maxOutputTokens": 10}
            }))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp)
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY"]
    async fn validate_non_streaming_structure() {
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not set");
        let mock_url =
            std::env::var("MOCK_URL").unwrap_or_else(|_| "http://localhost:8787".to_string());

        let real = call_real_api(&api_key).await.expect("Real API call failed");
        let mock = call_mock_api(&mock_url)
            .await
            .expect("Mock API call failed");

        print_comparison("Gemini Non-Streaming", &real, &mock);

        let mock_keys = extract_keys(&mock, "");

        assert!(
            mock_keys.contains("candidates"),
            "Mock missing 'candidates'"
        );
        assert!(
            mock_keys.contains("usageMetadata"),
            "Mock missing 'usageMetadata'"
        );

        let critical = [
            "candidates[*].content",
            "candidates[*].content.parts",
            "usageMetadata.totalTokenCount",
        ];
        for key in critical {
            assert!(
                mock_keys.iter().any(|k| k.contains(key)),
                "Mock missing critical key: {}",
                key
            );
        }

        println!("✓ Gemini validation passed");
    }
}

mod claude {
    use super::*;

    async fn call_real_api(api_key: &str) -> Result<Value, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let resp = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Say hi"}]
            }))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp)
    }

    async fn call_mock_api(base_url: &str) -> Result<Value, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/v1/messages", base_url))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Say hi"}]
            }))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp)
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY"]
    async fn validate_non_streaming_structure() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
        let mock_url =
            std::env::var("MOCK_URL").unwrap_or_else(|_| "http://localhost:8787".to_string());

        let real = call_real_api(&api_key).await.expect("Real API call failed");
        let mock = call_mock_api(&mock_url)
            .await
            .expect("Mock API call failed");

        print_comparison("Claude Non-Streaming", &real, &mock);

        let mock_keys = extract_keys(&mock, "");

        assert!(mock_keys.contains("id"), "Mock missing 'id'");
        assert!(mock_keys.contains("type"), "Mock missing 'type'");
        assert!(mock_keys.contains("role"), "Mock missing 'role'");
        assert!(mock_keys.contains("model"), "Mock missing 'model'");
        assert!(mock_keys.contains("content"), "Mock missing 'content'");
        assert!(
            mock_keys.contains("stop_reason"),
            "Mock missing 'stop_reason'"
        );
        assert!(mock_keys.contains("usage"), "Mock missing 'usage'");

        let critical = [
            "content[*].type",
            "content[*].text",
            "usage.input_tokens",
            "usage.output_tokens",
        ];
        for key in critical {
            assert!(
                mock_keys.iter().any(|k| k.contains(key)),
                "Mock missing critical key: {}",
                key
            );
        }

        println!("✓ Claude validation passed");
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY"]
    async fn validate_thinking_structure() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
        let mock_url =
            std::env::var("MOCK_URL").unwrap_or_else(|_| "http://localhost:8787".to_string());

        let client = reqwest::Client::new();

        // Real API with thinking
        let real: Value = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 2000,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [{"role": "user", "content": "What is 2+2?"}]
            }))
            .send()
            .await
            .expect("Real API call failed")
            .json()
            .await
            .expect("Failed to parse real response");

        // Mock API with thinking
        let mock: Value = client
            .post(format!("{}/v1/messages", mock_url))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 2000,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [{"role": "user", "content": "What is 2+2?"}]
            }))
            .send()
            .await
            .expect("Mock API call failed")
            .json()
            .await
            .expect("Failed to parse mock response");

        print_comparison("Claude Thinking", &real, &mock);

        // Check thinking block exists
        let content = mock.get("content").expect("No content in mock");
        let has_thinking = content
            .as_array()
            .map(|arr| {
                arr.iter()
                    .any(|c| c.get("type") == Some(&Value::String("thinking".to_string())))
            })
            .unwrap_or(false);

        assert!(has_thinking, "Mock missing thinking block");

        // Check signature exists in thinking block
        let thinking_block = content.as_array().and_then(|arr| {
            arr.iter()
                .find(|c| c.get("type") == Some(&Value::String("thinking".to_string())))
        });

        if let Some(block) = thinking_block {
            assert!(
                block.get("signature").is_some(),
                "Thinking block missing signature"
            );
        }

        println!("✓ Claude thinking validation passed");
    }
}

mod openai {
    use super::*;

    async fn call_real_api(api_key: &str) -> Result<Value, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let resp = client
            .post("https://api.openai.com/v1/responses")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": "gpt-4o-mini",
                "input": "Say hi"
            }))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp)
    }

    async fn call_mock_api(base_url: &str) -> Result<Value, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/v1/responses", base_url))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": "gpt-4o-mini",
                "input": "Say hi"
            }))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp)
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn validate_responses_structure() {
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
        let mock_url =
            std::env::var("MOCK_URL").unwrap_or_else(|_| "http://localhost:8787".to_string());

        let real = call_real_api(&api_key).await.expect("Real API call failed");
        let mock = call_mock_api(&mock_url)
            .await
            .expect("Mock API call failed");

        print_comparison("OpenAI Responses", &real, &mock);

        let mock_keys = extract_keys(&mock, "");

        assert!(mock_keys.contains("id"), "Mock missing 'id'");
        assert!(mock_keys.contains("object"), "Mock missing 'object'");
        assert!(mock_keys.contains("status"), "Mock missing 'status'");
        assert!(mock_keys.contains("model"), "Mock missing 'model'");
        assert!(mock_keys.contains("output"), "Mock missing 'output'");
        assert!(mock_keys.contains("usage"), "Mock missing 'usage'");

        let critical = [
            "output[*].type",
            "output[*].content",
            "usage.input_tokens",
            "usage.output_tokens",
        ];
        for key in critical {
            assert!(
                mock_keys.iter().any(|k| k.contains(key)),
                "Mock missing critical key: {}",
                key
            );
        }

        println!("✓ OpenAI Responses validation passed");
    }
}

mod tool_calling {
    use super::*;

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY"]
    async fn validate_claude_tool_use() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
        let mock_url =
            std::env::var("MOCK_URL").unwrap_or_else(|_| "http://localhost:8787".to_string());

        let client = reqwest::Client::new();
        let body = serde_json::json!({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 200,
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}
            }],
            "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]
        });

        let real: Value = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .expect("Real API failed")
            .json()
            .await
            .expect("Parse failed");

        let mock: Value = client
            .post(format!("{}/v1/messages", mock_url))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .expect("Mock API failed")
            .json()
            .await
            .expect("Parse failed");

        print_comparison("Claude Tool Use", &real, &mock);

        // Check tool_use block
        let mock_content = mock.get("content").and_then(|c| c.as_array());
        let has_tool_use = mock_content
            .map(|arr| {
                arr.iter()
                    .any(|c| c.get("type") == Some(&Value::String("tool_use".to_string())))
            })
            .unwrap_or(false);

        assert!(has_tool_use, "Mock missing tool_use block");
        assert_eq!(
            mock.get("stop_reason"),
            Some(&Value::String("tool_use".to_string())),
            "Mock should have stop_reason: tool_use"
        );

        println!("✓ Claude tool use validation passed");
    }
}
