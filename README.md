# TokenIpsum

Mock LLM API server for testing. Generates fake but structurally accurate responses matching real provider APIs.

## Features

- **Multi-provider support** - Cerebras, Gemini, Claude, OpenAI (new Responses API)
- **Error simulation** - 401, 429, 500 errors with provider-specific formats
- **Rate limiting** - Configurable request limits
- **Latency simulation** - Add artificial delays
- **Auth validation** - Optional API key checking
- **Configurable** - TOML config file or environment variables

## Quick Start

### Docker (Recommended)

```bash
# Just run it
docker compose up

# Detached mode
docker compose up -d

# Rebuild after changes
docker compose up --build
```

### Cargo

```bash
# Run with defaults
cargo run

# Run with config file
CONFIG=config.toml cargo run

# Run on different port
PORT=9000 cargo run
```

## Configuration

Copy `config.example.toml` to `config.toml`:

```toml
[server]
port = 8787
latency_ms = 0           # Artificial delay in ms

[rate_limit]
enabled = false
requests_per_minute = 60
fail_after_requests = 0  # Return 429 after N requests (0 = never)

[errors]
error_rate = 0.0         # Random error probability (0.0-1.0)
force_error = "none"     # Force: none, unauthorized, rate_limit, server_error, timeout

[auth]
require_auth = false
valid_keys = ["test-key-123"]

[providers]
cerebras = true
gemini = true
claude = true
openai = true
```

## Endpoints

### Health Check
```bash
curl http://localhost:8787/health
```

### Cerebras (OpenAI-compatible)
```bash
# POST /v1/chat/completions
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.3-70b","messages":[{"role":"user","content":"Hello"}]}'
```

### Gemini
```bash
# POST /v1beta/models/{model}:generateContent
curl http://localhost:8787/v1beta/models/gemini-2.0-flash:generateContent \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}'

# POST /v1beta/models/{model}:streamGenerateContent
curl http://localhost:8787/v1beta/models/gemini-2.0-flash:streamGenerateContent \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}'
```

### Claude (Anthropic)
```bash
# POST /v1/messages
curl http://localhost:8787/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-haiku","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}'

# With thinking
curl http://localhost:8787/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model":"claude-haiku",
    "max_tokens":2000,
    "thinking":{"type":"enabled","budget_tokens":1024},
    "messages":[{"role":"user","content":"What is 2+2?"}]
  }'
```

### OpenAI (new Responses API)
```bash
# POST /v1/responses
curl http://localhost:8787/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","input":"Hello"}'

# With messages format
curl http://localhost:8787/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model":"gpt-4o-mini",
    "input":[{"role":"user","content":"Hello"}]
  }'

# Streaming
curl http://localhost:8787/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","input":"Hello","stream":true}'
```

## Tool Calling

All providers support tool/function calling. Triggers on keywords: "weather", "search", "calculate", "find", "what is".

```bash
# Cerebras
curl http://localhost:8787/v1/chat/completions \
  -d '{"model":"llama","messages":[{"role":"user","content":"Weather in Tokyo?"}],
       "tools":[{"type":"function","function":{"name":"get_weather"}}]}'

# Claude
curl http://localhost:8787/v1/messages \
  -d '{"model":"claude","max_tokens":200,"messages":[{"role":"user","content":"Weather in Tokyo?"}],
       "tools":[{"name":"get_weather","input_schema":{}}]}'

# OpenAI
curl http://localhost:8787/v1/responses \
  -d '{"model":"gpt-4o","input":"Weather in Tokyo?",
       "tools":[{"type":"function","name":"get_weather"}]}'
```

## Error Simulation

### Rate Limiting
```toml
# config.toml - return 429 after 10 requests
[rate_limit]
fail_after_requests = 10
```

```bash
# Response after limit exceeded (Claude format):
{"type":"error","error":{"type":"rate_limit_error","message":"Rate limit exceeded..."}}
```

### Force Errors
```toml
# config.toml
[errors]
force_error = "unauthorized"  # All requests return 401
```

### Random Errors
```toml
# config.toml - 10% of requests fail randomly
[errors]
error_rate = 0.1
```

### Auth Validation
```toml
# config.toml
[auth]
require_auth = true
valid_keys = ["sk-test-123", "sk-prod-456"]
```

```bash
# Without valid key:
curl http://localhost:8787/v1/messages -H "Authorization: Bearer invalid"
# {"type":"error","error":{"type":"authentication_error","message":"Invalid API key provided."}}
```

## Error Response Formats

Each provider returns errors in its native format:

**Cerebras/OpenAI:**
```json
{"error":{"message":"...","type":"rate_limit_error","code":"rate_limit_exceeded"}}
```

**Gemini:**
```json
{"error":{"code":429,"message":"...","status":"RESOURCE_EXHAUSTED"}}
```

**Claude:**
```json
{"type":"error","error":{"type":"rate_limit_error","message":"..."}}
```

## Testing

```bash
# Unit tests
cargo test

# Real API validation (requires API keys)
export CEREBRAS_API_KEY="..."
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."

cargo run &
cargo test --test real_api_validation -- --ignored --nocapture
```

## Library Usage

Use TokenIpsum as a dependency in your Rust project:

```toml
[dependencies]
tokenipsum = { git = "https://github.com/youruser/tokenipsum" }
# or local path
tokenipsum = { path = "../tokenipsum" }
```

### Basic Example

```rust
use tokenipsum::{Config, RuntimeState, create_router};

#[tokio::main]
async fn main() {
    let config = Config::default();
    let state = RuntimeState::new(config);
    let app = create_router(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8787").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

### Custom Configuration

```rust
use tokenipsum::{Config, RuntimeState, create_router};

#[tokio::main]
async fn main() {
    let mut config = Config::default();

    // Only enable Claude
    config.providers.cerebras = false;
    config.providers.gemini = false;
    config.providers.openai = false;
    config.providers.claude = true;

    // Simulate 100ms latency
    config.server.latency_ms = 100;

    // 10% random error rate
    config.errors.error_rate = 0.1;

    let state = RuntimeState::new(config);
    let app = create_router(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8787").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

### Exported Items

- `Config`, `RuntimeState` - Configuration and state management
- `create_router(state)` - Creates the Axum router with all enabled providers
- `Provider` - Provider enum (Cerebras, Claude, Gemini, OpenAI)
- `ContentGenerator` - Lorem ipsum content generator
- `cerebras`, `claude`, `gemini`, `openai` - Provider modules with request/response types

## Project Structure

```
tokenipsum/
├── src/
│   ├── main.rs        # CLI entrypoint
│   ├── lib.rs         # Library exports
│   ├── config.rs      # TOML config and runtime state
│   ├── errors.rs      # Error response generators
│   ├── generator.rs   # Lorem ipsum content generator
│   ├── cerebras.rs    # Cerebras/OpenAI chat completions
│   ├── gemini.rs      # Google Gemini
│   ├── claude.rs      # Anthropic Claude
│   └── openai.rs      # OpenAI Responses API
├── tests/
│   └── real_api_validation.rs
├── Dockerfile         # Multi-stage build
├── docker-compose.yml
├── config.example.toml
└── README.md
```

## License

MIT OR Apache-2.0
