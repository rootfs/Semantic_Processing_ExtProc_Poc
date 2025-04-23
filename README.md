# LLM Semantic Router

An Envoy External Processor (ExtProc) that acts as an external **Mixture-of-Models (MoM)** router. It intelligently directs OpenAI API requests to the most suitable backend model from a defined pool based on semantic understanding of the request's intent. This is achieved using BERT similarity matching (current implementation) and classification (future implementation). Conceptually similar to Mixture-of-Experts (MoE) which lives *within* a model, this system selects the best *entire model* for the nature of the task.

The detailed design doc can be found [here](https://docs.google.com/document/d/1BwwRxdf74GuCdG1veSApzMRMJhXeUWcw0wH9YRAmgGw/edit?usp=sharing).

The screenshot below shows the LLM Router dashboard in Grafana.

![LLM Router Dashboard](./docs/grafana_screenshot.png)

The router is implemented in two ways: Golang (with Rust FFI based on Candle) and Python. Benchmarking will be conducted to determine the best implementation.

## Usage

### Run the Envoy Proxy

This listens for incoming requests and uses the ExtProc filter.
```bash
make run-envoy
```

### Run the Semantic Router (Go Implementation)

This builds the Rust binding and the Go router, then starts the ExtProc gRPC server that Envoy communicates with.
```bash
make run-router
```

Once both Envoy and the router are running, you can test the routing logic using predefined prompts:

```bash
make test-prompt
```

This will send curl requests simulating different types of user prompts (Math, Creative Writing, General) to the Envoy endpoint (`http://localhost:8801`). The router should direct these to the appropriate backend model configured in `config/config.yaml`.

