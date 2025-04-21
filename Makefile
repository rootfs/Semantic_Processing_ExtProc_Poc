.PHONY: all build clean test docker-build podman-build docker-run podman-run

# Default target
all: build

# vLLM env var
VLLM_ENDPOINT ?= http://192.168.12.175:11434

# Container settings
USE_CONTAINER ?= false
CONTAINER_ENGINE ?= podman
IMAGE_NAME ?= llm-semantic-router
CONTAINER_NAME ?= llm-semantic-router-container
CONTAINER_VOLUMES = -v ${PWD}:/app

# USE_CONTAINER and CONTAINER_ENGINE to determine the container command or not use container
ifeq ($(USE_CONTAINER),true)
  ifeq ($(CONTAINER_ENGINE),docker)
    CONTAINER_CMD = docker
  else ifeq ($(CONTAINER_ENGINE),podman)
    CONTAINER_CMD = podman
  else
    $(error CONTAINER_ENGINE must be either docker or podman)
  endif
  EXEC_PREFIX = $(CONTAINER_CMD) exec $(CONTAINER_NAME)
  RUN_PREFIX = $(CONTAINER_CMD) run --rm $(CONTAINER_VOLUMES) --network=host --name $(CONTAINER_NAME)
else
  EXEC_PREFIX =
  RUN_PREFIX =
endif

# Build the Rust library and Golang binding
build: rust build-router

# Build the Rust library
rust:
	@echo "Building Rust library..."
ifeq ($(USE_CONTAINER),true)
	$(CONTAINER_CMD) build -t $(IMAGE_NAME) .
	$(RUN_PREFIX) -d $(IMAGE_NAME) sleep infinity
	$(EXEC_PREFIX) bash -c "cd candle-binding && cargo build --release"
	$(CONTAINER_CMD) stop $(CONTAINER_NAME)
else
	cd candle-binding && cargo build --release
endif

# Build router
build-router: rust
	@echo "Building router..."
ifeq ($(USE_CONTAINER),true)
	$(RUN_PREFIX) -d $(IMAGE_NAME) sleep infinity
	$(EXEC_PREFIX) bash -c "mkdir -p bin && cd semantic_router && go build -o ../bin/router cmd/main.go"
	$(CONTAINER_CMD) stop $(CONTAINER_NAME)
else
	@mkdir -p bin
	@cd semantic_router && go build -o ../bin/router cmd/main.go
endif

# Run the router
run-router: build-router
	@echo "Running router..."
ifeq ($(USE_CONTAINER),true)
	$(RUN_PREFIX) $(IMAGE_NAME) bash -c "export LD_LIBRARY_PATH=/app/candle-binding/target/release && ./bin/router -config=config/config.yaml"
else
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		./bin/router -config=config/config.yaml
endif

# Run Envoy proxy
run-envoy:
	@echo "Starting Envoy..."
ifeq ($(USE_CONTAINER),true)
	$(RUN_PREFIX) $(IMAGE_NAME) envoy --config-path config/envoy.yaml --component-log-level ext_proc:debug
else
	envoy --config-path config/envoy.yaml --component-log-level "ext_proc:trace,router:trace,http:trace"
endif

# Test the Rust library
test-binding:
	@echo "Running Go tests with static library..."
ifeq ($(USE_CONTAINER),true)
	$(RUN_PREFIX) -d $(IMAGE_NAME) sleep infinity
	$(EXEC_PREFIX) bash -c "cd candle-binding && CGO_ENABLED=1 go test -v"
	$(CONTAINER_CMD) stop $(CONTAINER_NAME)
else
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		cd candle-binding && CGO_ENABLED=1 go test -v
endif

# Test the Rust library and the Go binding
test: test-binding

# Clean built artifacts
clean:
	@echo "Cleaning build artifacts..."
ifeq ($(USE_CONTAINER),true)
	$(RUN_PREFIX) -d $(IMAGE_NAME) sleep infinity
	$(EXEC_PREFIX) bash -c "cd candle-binding && cargo clean && rm -f ../bin/router"
	$(CONTAINER_CMD) stop $(CONTAINER_NAME)
	$(CONTAINER_CMD) rmi $(IMAGE_NAME)
else
	cd candle-binding && cargo clean
	rm -f bin/router
endif

# Build container image
container-build:
	@echo "Building container image..."
	$(CONTAINER_CMD) build -t $(IMAGE_NAME) .

# Start an interactive shell in the container
container-shell: container-build
	@echo "Starting interactive container shell..."
	$(RUN_PREFIX) -it $(IMAGE_NAME) bash

# Test the Envoy extproc
test-prompt:
	@echo "Testing Envoy extproc with curl (Math)..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "qwen2.5:32b", "messages": [{"role": "assistant", "content": "You are a professional math teacher. Explain math concepts clearly and show step-by-step solutions to problems."}, {"role": "user", "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?"}], "temperature": 0.7}'
	@echo "Testing Envoy extproc with curl (Creative Writing)..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "qwen2.5:32b", "messages": [{"role": "assistant", "content": "You are a story writer. Create interesting stories with good characters and settings."}, {"role": "user", "content": "Write a short story about a space cat."}], "temperature": 0.7}'
	@echo "Testing Envoy extproc with curl (Default/General)..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "qwen2.5:32b", "messages": [{"role": "assistant", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}], "temperature": 0.7}'

test-vllm:
	curl -X POST $(VLLM_ENDPOINT)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "qwen2.5:32b", "messages": [{"role": "assistant", "content": "You are a professional math teacher. Explain math concepts clearly and show step-by-step solutions to problems."}, {"role": "user", "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?"}], "temperature": 0.7}' | jq
