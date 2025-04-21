package candle_binding

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"
)

// ResetModel completely resets the model in Rust side to allow loading a new model
func ResetModel() {
	// Clean up the model state
	modelInitialized = false
	runtime.GC()
	SetMemoryCleanupHandler()
	// Create a new sync.Once to allow reinitialization
	initOnce = sync.Once{}
	time.Sleep(200 * time.Millisecond)
}

// Test models to benchmark
var testModels = []struct {
	name    string
	modelID string
	size    string // Model size category for comparison
}{
	// per https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads
	// TF models like bert-base-uncased are not supported by candle yet, need to convert to pytorch through https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py
	{
		name:    "Paraphrase-multilingual-MiniLM-L12-v2",
		modelID: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
		size:    "small",
	},
	{
		name:    "MiniLM-L12",
		modelID: "sentence-transformers/all-MiniLM-L12-v2",
		size:    "medium",
	},
	{
		name:    "BGE-large",
		modelID: "BAAI/bge-large-zh-v1.5",
		size:    "large",
	},
}

// More comprehensive test dataset for better model differentiation
var testDataset = []struct {
	text1 string
	text2 string
	class string // Semantic class for evaluation
}{
	{
		text1: "I love machine learning",
		text2: "I enjoy artificial intelligence",
		class: "SIMILAR",
	},
	{
		text1: "I need to buy groceries",
		text2: "The stock market is volatile",
		class: "DIFFERENT",
	},
	{
		text1: "Python is a programming language",
		text2: "Python is a type of snake",
		class: "AMBIGUOUS",
	},
	{
		text1: "This is a test sentence",
		text2: "This is a test sentence",
		class: "IDENTICAL",
	},
}

// Initialize the model before running tests
func initBERTModel(t *testing.T, modelID string, useCPU bool) bool {
	// Print current working directory and environment
	pwd, _ := os.Getwd()
	fmt.Printf("Current working directory: %s\n", pwd)
	fmt.Printf("HF_HOME: %s\n", os.Getenv("HF_HOME"))

	// Use the model ID directly for Hugging Face download
	fmt.Printf("Initializing BERT model with ID: %s (CPU: %v)\n", modelID, useCPU)

	// Force cleanup of any previous model
	ResetModel()

	// Initialize the model
	err := InitModel(modelID, useCPU)
	if err != nil {
		t.Logf("Failed to initialize BERT model: %v", err)
		return false
	}

	// Verify model is initialized
	if !IsModelInitialized() {
		t.Logf("BERT model initialization failed")
		return false
	}

	// Do a quick test to verify it works
	score := CalculateSimilarity("test", "test")
	if score < 0 {
		t.Logf("Model initialization succeeded but similarity calculation failed")
		return false
	}

	fmt.Printf("Model successfully initialized\n")
	return true
}

// TestModelBenchmarking tests different models for accuracy and performance
func TestModelBenchmarking(t *testing.T) {
	// Check if GPU is available first
	gpuAvailable := false
	if os.Getenv("FORCE_CPU_ONLY") != "1" {
		t.Log("Checking GPU availability...")
		gpuAvailable = checkGPUAvailability(t)
		t.Logf("GPU available: %v", gpuAvailable)
	}

	// Define device types to test
	devices := []struct {
		name   string
		useCPU bool
	}{
		{name: "CPU", useCPU: true},
	}

	// Only add GPU if it's available
	if gpuAvailable {
		devices = append(devices, struct {
			name   string
			useCPU bool
		}{name: "GPU", useCPU: false})
	}

	results := make(map[string]map[string]map[string]float64) // model -> device -> metric -> value

	// Initialize results map
	for _, model := range testModels {
		results[model.name] = make(map[string]map[string]float64)
		for _, device := range devices {
			results[model.name][device.name] = make(map[string]float64)
		}
	}

	// Run benchmarks for each model and device combination
	for _, model := range testModels {
		for _, device := range devices {
			modelName := model.name
			deviceName := device.name

			t.Logf("\n===== Benchmarking %s on %s =====", modelName, deviceName)

			// Completely reset the model between tests
			ResetModel()

			// Initialize model
			if !initBERTModel(t, model.modelID, device.useCPU) {
				t.Logf("Failed to initialize %s on %s, skipping", modelName, deviceName)
				continue
			}

			// Warm up with repeated calls
			for i := 0; i < 3; i++ {
				CalculateSimilarity("Warm up", "Warm up")
			}

			// Measure first inference time
			initStart := time.Now()
			CalculateSimilarity("First inference", "First inference")
			initDuration := time.Since(initStart)
			results[modelName][deviceName]["init_ms"] = float64(initDuration.Milliseconds())

			// Measure classification accuracy
			totalLatency := time.Duration(0)

			// Categorize results by semantic class
			categoryLatency := make(map[string]time.Duration)

			for i, test := range testDataset {
				start := time.Now()
				score := CalculateSimilarity(test.text1, test.text2)
				latency := time.Since(start)
				totalLatency += latency

				categoryLatency[test.class] += latency

				t.Logf("Test %d: %s - Score: %.4f, Latency: %.2f ms",
					i, test.class, score, float64(latency.Milliseconds()))
			}
			// Clean up between models
			ResetModel()
		}
	}

	// Print summary table
	t.Log("\n====== BENCHMARK RESULTS ======")
	t.Log("Model\tSize\tDevice\tInit Time (ms)")

	for _, model := range testModels {
		for _, device := range devices {
			if metrics, ok := results[model.name][device.name]; ok && len(metrics) > 0 {
				initTime := results[model.name][device.name]["init_ms"]

				t.Logf("%s\t%s\t%s\t%.2f",
					model.name, model.size, device.name, initTime)
			}
		}
	}

	// Print comparative analysis tables
	printComparativeTables(t, results, devices, testModels)
}

// printComparativeTables prints detailed comparison tables for model performance
func printComparativeTables(t *testing.T, results map[string]map[string]map[string]float64,
	devices []struct {
		name   string
		useCPU bool
	}, models []struct{ name, modelID, size string }) {

	// Check if we have enough data to make comparisons
	if len(results) == 0 {
		t.Log("\nNot enough data for comparative analysis")
		return
	}
}

// checkGPUAvailability performs a more careful check for GPU availability
func checkGPUAvailability(t *testing.T) bool {
	// Force cleanup first
	ResetModel()

	// Try to initialize a small model with GPU
	t.Log("initialize model on GPU...")
	modelID := "sentence-transformers/all-MiniLM-L6-v2"

	err := InitModel(modelID, false) // false = use GPU
	if err != nil {
		t.Logf("GPU initialization failed: %v", err)
		return false
	}

	// Verify initialization succeeded
	if !IsModelInitialized() {
		t.Log("GPU not available: model initialization failed")
		return false
	}

	// Try a simple inference to confirm
	score := CalculateSimilarity("test", "test")
	if score < 0 {
		t.Log("GPU not available: inference failed")
		return false
	}

	// Clean up after test
	ResetModel()

	t.Log("GPU is available and working")
	return true
}
