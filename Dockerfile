# Stage 1: Build
FROM rust:1.83-alpine AS builder

RUN apk add --no-cache musl-dev

WORKDIR /app

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./

# Create dummy src to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src

# Copy actual source
COPY src ./src

# Touch main.rs to invalidate the dummy build
RUN touch src/main.rs

# Build the actual binary
RUN cargo build --release --locked

# Stage 2: Runtime
FROM alpine:3.21

RUN apk add --no-cache ca-certificates

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/target/release/tokenipsum /usr/local/bin/tokenipsum

# Copy example config (optional, can be overridden via mount)
COPY config.example.toml /app/config.example.toml

ENV RUST_LOG=tokenipsum=info
ENV PORT=8787

EXPOSE 8787

ENTRYPOINT ["tokenipsum"]
