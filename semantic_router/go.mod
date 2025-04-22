module github.com/neuralmagic/semantic_router_poc/semantic_router

go 1.24.1

replace (
	github.com/neuralmagic/semantic_router_poc/candle-binding => ../candle-binding
	github.com/neuralmagic/semantic_router_poc/semantic_router/pkg/config => ./pkg/config
	github.com/neuralmagic/semantic_router_poc/semantic_router/pkg/extproc => ./pkg/extproc
)

require (
	github.com/envoyproxy/go-control-plane/envoy v1.32.4
	github.com/neuralmagic/semantic_router_poc/candle-binding v0.0.0-00010101000000-000000000000
	github.com/prometheus/client_golang v1.18.0
	google.golang.org/grpc v1.71.1
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/cncf/xds/go v0.0.0-20241223141626-cff3c89139a3 // indirect
	github.com/envoyproxy/protoc-gen-validate v1.2.1 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/planetscale/vtprotobuf v0.6.1-0.20240319094008-0393e58bdf10 // indirect
	github.com/prometheus/client_model v0.6.1 // indirect
	github.com/prometheus/common v0.46.0 // indirect
	github.com/prometheus/procfs v0.12.0 // indirect
	github.com/rogpeppe/go-internal v1.12.0 // indirect
	golang.org/x/net v0.34.0 // indirect
	golang.org/x/sys v0.29.0 // indirect
	golang.org/x/text v0.21.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250115164207-1a7da9e5054f // indirect
	google.golang.org/protobuf v1.36.4 // indirect
)
