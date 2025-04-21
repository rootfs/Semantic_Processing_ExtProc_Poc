package config

import (
	"fmt"
	"os"
	"sync"

	"gopkg.in/yaml.v3"
)

// RouterConfig represents the main configuration for the LLM Router
type RouterConfig struct {
	// BERT model configuration for Candle BERT similarity comparison
	BertModel struct {
		ModelID   string  `yaml:"model_id"`
		Threshold float32 `yaml:"threshold"`
		UseCPU    bool    `yaml:"use_cpu"`
	} `yaml:"bert_model"`

	// Categories of tasks for routing
	Categories []Category `yaml:"categories"`

	// Default LLM model to use if no match is found
	DefaultModel string `yaml:"default_model"`

	// Semantic cache configuration
	SemanticCache SemanticCacheConfig `yaml:"semantic_cache"`
}

// SemanticCacheConfig represents configuration for the semantic cache
type SemanticCacheConfig struct {
	// Enable semantic caching
	Enabled bool `yaml:"enabled"`

	// Similarity threshold for cache hits (0.0-1.0)
	// If not specified, will use the BertModel.Threshold
	SimilarityThreshold *float32 `yaml:"similarity_threshold,omitempty"`

	// Maximum number of cache entries to keep
	MaxEntries int `yaml:"max_entries,omitempty"`

	// Time-to-live for cache entries in seconds (0 means no expiration)
	TTLSeconds int `yaml:"ttl_seconds,omitempty"`
}

// GetCacheSimilarityThreshold returns the effective threshold for the semantic cache
func (c *RouterConfig) GetCacheSimilarityThreshold() float32 {
	if c.SemanticCache.SimilarityThreshold != nil {
		return *c.SemanticCache.SimilarityThreshold
	}
	return c.BertModel.Threshold
}

// Category represents a category of tasks
type Category struct {
	Name        string      `yaml:"name"`
	Description string      `yaml:"description,omitempty"`
	Model       string      `yaml:"model"` // LLM model name like "llama3-70b", "gpt-4o", etc.
	Tasks       []TaskEntry `yaml:"tasks,omitempty"`
}

// TaskEntry represents a task description and its associated typical prompt
type TaskEntry struct {
	Name          string `yaml:"name"`
	Description   string `yaml:"description,omitempty"`
	TypicalPrompt string `yaml:"typical_prompt,omitempty"`
	Model         string `yaml:"model,omitempty"` // Optional override of category LLM model
}

var (
	config     *RouterConfig
	configOnce sync.Once
	configErr  error
)

// LoadConfig loads the configuration from the specified YAML file
func LoadConfig(configPath string) (*RouterConfig, error) {
	configOnce.Do(func() {
		data, err := os.ReadFile(configPath)
		if err != nil {
			configErr = fmt.Errorf("failed to read config file: %w", err)
			return
		}

		config = &RouterConfig{}
		if err := yaml.Unmarshal(data, config); err != nil {
			configErr = fmt.Errorf("failed to parse config file: %w", err)
			return
		}
	})

	if configErr != nil {
		return nil, configErr
	}
	return config, nil
}

// GetConfig returns the current configuration
func GetConfig() *RouterConfig {
	return config
}

// GetTaskDescriptions returns all task descriptions for similarity matching
func (c *RouterConfig) GetTaskDescriptions() []string {
	var descriptions []string
	for _, category := range c.Categories {
		for _, task := range category.Tasks {
			descriptions = append(descriptions, task.Description)
		}
	}
	return descriptions
}

// GetModelForTaskIndex returns the LLM model name for a task at the given index
func (c *RouterConfig) GetModelForTaskIndex(index int) string {
	if index < 0 {
		return c.DefaultModel
	}

	count := 0
	for _, category := range c.Categories {
		for _, task := range category.Tasks {
			if count == index {
				// If task has a specific model, use it
				if task.Model != "" {
					return task.Model
				}
				// Otherwise, use the category's model
				if category.Model != "" {
					return category.Model
				}
				// Fall back to default model
				return c.DefaultModel
			}
			count++
		}
	}
	return c.DefaultModel
}
