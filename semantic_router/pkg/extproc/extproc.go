package extproc

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	candle_binding "github.com/neuralmagic/semantic_router_poc/candle-binding"
	"github.com/neuralmagic/semantic_router_poc/semantic_router/pkg/config"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var (
	initialized bool
	initMutex   sync.Mutex
)

// OpenAIRouter is an Envoy ExtProc server that routes OpenAI API requests
type OpenAIRouter struct {
	Config           *config.RouterConfig
	TaskDescriptions []string
}

// Ensure OpenAIRouter implements the ext_proc calls
var _ ext_proc.ExternalProcessorServer = &OpenAIRouter{}

// NewOpenAIRouter creates a new OpenAI API router instance
func NewOpenAIRouter(configPath string) (*OpenAIRouter, error) {
	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	initMutex.Lock()
	defer initMutex.Unlock()

	if !initialized {
		// Initialize the BERT model
		err = candle_binding.InitModel(cfg.BertModel.ModelID, cfg.BertModel.UseCPU)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize BERT model: %w", err)
		}
		initialized = true
	}

	taskDescriptions := cfg.GetTaskDescriptions()
	log.Printf("Task descriptions: %v", taskDescriptions)

	return &OpenAIRouter{Config: cfg, TaskDescriptions: taskDescriptions}, nil
}

// Send a response with proper error handling and logging
func sendResponse(stream ext_proc.ExternalProcessor_ProcessServer, response *ext_proc.ProcessingResponse, msgType string) error {
	log.Printf("Sending %s response: %+v", msgType, response)
	if err := stream.Send(response); err != nil {
		log.Printf("Error sending %s response: %v", msgType, err)
		return err
	}
	log.Printf("Successfully sent %s response", msgType)
	return nil
}

// Process implements the ext_proc calls
func (r *OpenAIRouter) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	log.Println("Started processing a new request")
	requestHeaders := make(map[string]string)

	for {
		req, err := stream.Recv()
		if err != nil {
			log.Printf("Error receiving request: %v", err)
			return err
		}

		log.Printf("Processing message type: %T", req.Request)

		switch v := req.Request.(type) {
		case *ext_proc.ProcessingRequest_RequestHeaders:
			log.Println("Received request headers")

			// Store headers for later use (since we need to modify the request length)
			headers := v.RequestHeaders.Headers
			for _, h := range headers.Headers {
				requestHeaders[h.Key] = h.Value
			}

			// Allow the request to continue
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestHeaders{
					RequestHeaders: &ext_proc.HeadersResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}

			if err := sendResponse(stream, response, "header"); err != nil {
				return err
			}

		case *ext_proc.ProcessingRequest_RequestBody:
			log.Println("Received request body")

			// Parse the OpenAI request
			body := v.RequestBody.Body
			openAIRequest, err := parseOpenAIRequest(body)
			if err != nil {
				log.Printf("Error parsing OpenAI request: %v", err)
				return status.Errorf(codes.InvalidArgument, "invalid request body: %v", err)
			}

			log.Printf("Original model: %s", openAIRequest.Model)

			// Get content from messages
			var userContent string
			var nonUserMessages []string

			for _, msg := range openAIRequest.Messages {
				if msg.Role == "user" {
					userContent = msg.Content
				} else if msg.Role != "" {
					nonUserMessages = append(nonUserMessages, msg.Content)
				}
			}

			// Create default response with CONTINUE status
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestBody{
					RequestBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}

			// The user content could be very long and not relevant to the task,
			// so we only use non-user messages (aka system, assistant, etc)
			// If there are non-user messages, use BERT to find the best model
			if len(nonUserMessages) > 0 && userContent != "" {
				// Add all non-user messages to get context
				nonUserContent := strings.Join(nonUserMessages, " ")

				// Find the most similar task description
				bestMatch := r.findBestModelMatch(nonUserContent)
				if bestMatch != openAIRequest.Model && bestMatch != "" {
					log.Printf("Routing to model: %s", bestMatch)

					// Modify the model in the request
					openAIRequest.Model = bestMatch

					// Serialize the modified request
					modifiedBody, err := json.Marshal(openAIRequest)
					if err != nil {
						log.Printf("Error serializing modified request: %v", err)
						return status.Errorf(codes.Internal, "error serializing modified request: %v", err)
					}

					// Create body mutation with the modified body
					bodyMutation := &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_Body{
							Body: modifiedBody,
						},
					}

					// Also create a header mutation to remove the original content-length
					headerMutation := &ext_proc.HeaderMutation{
						RemoveHeaders: []string{"content-length"},
					}

					// Set the response with both mutations
					response = &ext_proc.ProcessingResponse{
						Response: &ext_proc.ProcessingResponse_RequestBody{
							RequestBody: &ext_proc.BodyResponse{
								Response: &ext_proc.CommonResponse{
									Status:         ext_proc.CommonResponse_CONTINUE,
									HeaderMutation: headerMutation,
									BodyMutation:   bodyMutation,
								},
							},
						},
					}

					log.Printf("Use new model: %s", bestMatch)
				}
			}

			if err := sendResponse(stream, response, "body"); err != nil {
				return err
			}

		case *ext_proc.ProcessingRequest_ResponseHeaders:
			log.Println("Received response headers")

			// Allow the response to continue without modification
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_ResponseHeaders{
					ResponseHeaders: &ext_proc.HeadersResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}

			if err := sendResponse(stream, response, "response header"); err != nil {
				return err
			}

		case *ext_proc.ProcessingRequest_ResponseBody:
			log.Println("Received response body")

			// Allow the response to continue without modification
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_ResponseBody{
					ResponseBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}

			if err := sendResponse(stream, response, "response body"); err != nil {
				return err
			}

		default:
			log.Printf("Unknown request type: %v", v)

			// For unknown message types, create a body response with CONTINUE status
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestBody{
					RequestBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}

			if err := sendResponse(stream, response, "unknown"); err != nil {
				return err
			}
		}
	}
}

// Find the best model match using similarity search
func (r *OpenAIRouter) findBestModelMatch(query string) string {
	if len(r.TaskDescriptions) == 0 {
		return r.Config.DefaultModel
	}

	// Use BERT to find the most similar task description
	result := candle_binding.FindMostSimilar(query, r.TaskDescriptions)
	log.Printf("Similarity search result: index=%d, score=%.4f", result.Index, result.Score)

	if result.Index < 0 || result.Score < r.Config.BertModel.Threshold {
		log.Printf("Using default model: %s", r.Config.DefaultModel)
		return r.Config.DefaultModel
	}

	// Get the model for the matched task
	model := r.Config.GetModelForTaskIndex(result.Index)
	log.Printf("Found matching model: %s", model)
	return model
}

// OpenAIRequest represents an OpenAI API request
type OpenAIRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
}

// ChatMessage represents a message in the OpenAI chat format
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Parse the OpenAI request JSON
func parseOpenAIRequest(data []byte) (*OpenAIRequest, error) {
	var req OpenAIRequest
	if err := json.Unmarshal(data, &req); err != nil {
		return nil, err
	}
	return &req, nil
}

// Server represents a gRPC server for the Envoy ExtProc
type Server struct {
	router *OpenAIRouter
	server *grpc.Server
	port   int
}

// NewServer creates a new ExtProc gRPC server
func NewServer(configPath string, port int) (*Server, error) {
	router, err := NewOpenAIRouter(configPath)
	if err != nil {
		return nil, err
	}

	return &Server{
		router: router,
		port:   port,
	}, nil
}

// Start starts the gRPC server
func (s *Server) Start() error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.port))
	if err != nil {
		return fmt.Errorf("failed to listen on port %d: %w", s.port, err)
	}

	s.server = grpc.NewServer()
	ext_proc.RegisterExternalProcessorServer(s.server, s.router)

	log.Printf("Starting LLM Router ExtProc server on port %d...", s.port)

	// Run the server in a separate goroutine
	serverErrCh := make(chan error, 1)
	go func() {
		if err := s.server.Serve(lis); err != nil && err != grpc.ErrServerStopped {
			log.Printf("Server error: %v", err)
			serverErrCh <- err
		} else {
			serverErrCh <- nil
		}
	}()

	// Wait for interrupt signal to gracefully shut down the server
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)

	// Wait for either server error or shutdown signal
	select {
	case err := <-serverErrCh:
		if err != nil {
			log.Printf("Server exited with error: %v", err)
			return err
		}
	case <-signalChan:
		log.Println("Received shutdown signal, gracefully stopping server...")
	}

	s.Stop()
	return nil
}

// Stop stops the gRPC server
func (s *Server) Stop() {
	if s.server != nil {
		s.server.GracefulStop()
		log.Println("Server stopped")
	}
}
