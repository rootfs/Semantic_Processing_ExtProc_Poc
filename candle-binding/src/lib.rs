// This file is a binding for the candle-core and candle-transformers libraries.
// It is based on https://github.com/huggingface/candle/tree/main/candle-examples/examples/bert
use std::ffi::{c_char, CStr, CString};
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder};
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

// Structure to hold BERT model and tokenizer for semantic similarity
pub struct BertSimilarity {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

lazy_static::lazy_static! {
    static ref BERT_SIMILARITY: Arc<Mutex<Option<BertSimilarity>>> = Arc::new(Mutex::new(None));
}

impl BertSimilarity {
    pub fn new(model_id: &str, use_cpu: bool) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Default to a sentence transformer model if not specified or empty
        let model_id = if model_id.is_empty() {
            "sentence-transformers/all-MiniLM-L6-v2"
        } else {
            model_id
        };

        // Load model and tokenizer from HF
        let repo = Repo::with_revision(
            model_id.to_string(), 
            RepoType::Model, 
            "main".to_string()  // Use main branch instead of PR/21
        );

        let (config_filename, tokenizer_filename, weights_filename, use_pth) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;

            // Try to get safetensors first, if that fails, fall back to pytorch_model.bin. This is for BAAI models
            // create a special case for BAAI to download the correct weights to avoid downloading the wrong weights
            let (weights, use_pth) = if model_id.starts_with("BAAI/") {
                // BAAI models typically use PyTorch model format
                (api.get("pytorch_model.bin")?, true)
            } else {
                match api.get("model.safetensors") {
                    Ok(weights) => (weights, false),
                    Err(_) => {
                        println!("Safetensors model not found, trying PyTorch model instead...");
                        (api.get("pytorch_model.bin")?, true)
                    }
                }
            };

            (config, tokenizer, weights, use_pth)
        };

        let config = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Use the approximate GELU for better performance
        config.hidden_act = HiddenAct::GeluApproximate;

        let vb = if use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };

        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    // Get embedding for a text
    pub fn get_embedding(&self, text: &str) -> Result<Tensor> {
        // Encode the text with the tokenizer
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(E::msg)?;
        
        // Get token IDs and attention mask
        let token_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();
        
        // Create tensors
        let token_ids_tensor = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let attention_mask_tensor = Tensor::new(&attention_mask[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids_tensor.zeros_like()?;
        
        // Run the text through BERT with attention mask
        let embeddings = self.model.forward(&token_ids_tensor, &token_type_ids, Some(&attention_mask_tensor))?;
        
        // Mean pooling: sum over tokens and divide by attention mask sum
        let sum_embeddings = embeddings.sum(1)?;
        let attention_sum = attention_mask_tensor.sum(1)?.to_dtype(embeddings.dtype())?;
        let pooled = sum_embeddings.broadcast_div(&attention_sum)?;
        
        // Convert to float32 and normalize
        let embedding = pooled.to_dtype(DType::F32)?;
        
        normalize_l2(&embedding)
    }

    // Calculate cosine similarity between two texts
    pub fn calculate_similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        let embedding1 = self.get_embedding(text1)?;
        let embedding2 = self.get_embedding(text2)?;
        
        // For normalized vectors, dot product equals cosine similarity
        let dot_product = embedding1.matmul(&embedding2.transpose(0, 1)?)?;
        
        // Extract the scalar value from the result
        let sim_value = dot_product.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?;
        
        Ok(sim_value)
    }

    // Find most similar text from a list
    pub fn find_most_similar(&self, query_text: &str, candidates: &[&str]) -> Result<(usize, f32)> {
        if candidates.is_empty() {
            return Err(E::msg("Empty candidate list"));
        }
        
        let query_embedding = self.get_embedding(query_text)?;
        
        // Calculate similarity for each candidate individually
        let mut best_idx = 0;
        let mut best_score = -1.0;
        
        for (idx, candidate) in candidates.iter().enumerate() {
            let candidate_embedding = self.get_embedding(candidate)?;
            
            // Calculate similarity (dot product of normalized vectors = cosine similarity)
            let sim = query_embedding.matmul(&candidate_embedding.transpose(0, 1)?)?;
            let score = sim.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?;
            
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        
        Ok((best_idx, best_score))
    }
}

// Initialize the BERT model (called from Go)
#[no_mangle]
pub extern "C" fn init_similarity_model(model_id: *const c_char, use_cpu: bool) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match BertSimilarity::new(model_id, use_cpu) {
        Ok(model) => {
            let mut bert_opt = BERT_SIMILARITY.lock().unwrap();
            *bert_opt = Some(model);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize BERT: {}", e);
            false
        }
    }
}

// Structure to hold similarity result
#[repr(C)]
pub struct SimilarityResult {
    pub index: i32,  // Index of the most similar text
    pub score: f32,  // Similarity score
}

// Structure to hold embedding result
#[repr(C)]
pub struct EmbeddingResult {
    pub data: *mut f32,
    pub length: i32,
    pub error: bool,
}

// Get embedding for a text (called from Go)
#[no_mangle]
pub extern "C" fn get_text_embedding(text: *const c_char) -> EmbeddingResult {
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return EmbeddingResult {
                data: std::ptr::null_mut(),
                length: 0,
                error: true
            },
        }
    };

    let bert_opt = BERT_SIMILARITY.lock().unwrap();
    let bert = match &*bert_opt {
        Some(b) => b,
        None => {
            eprintln!("BERT model not initialized");
            return EmbeddingResult {
                data: std::ptr::null_mut(),
                length: 0,
                error: true
            };
        }
    };

    match bert.get_embedding(text) {
        Ok(embedding) => {
            match embedding.flatten_all() {
                Ok(flat_embedding) => {
                    match flat_embedding.to_vec1::<f32>() {
                        Ok(vec) => {
                            let length = vec.len() as i32;
                            // Allocate memory that will be freed by Go
                            let data = vec.as_ptr() as *mut f32;
                            std::mem::forget(vec); // Don't drop the vector - Go will own the memory now
                            EmbeddingResult {
                                data,
                                length,
                                error: false
                            }
                        },
                        Err(_) => EmbeddingResult {
                            data: std::ptr::null_mut(),
                            length: 0,
                            error: true
                        }
                    }
                },
                Err(_) => EmbeddingResult {
                    data: std::ptr::null_mut(),
                    length: 0,
                    error: true
                }
            }
        },
        Err(e) => {
            eprintln!("Error getting embedding: {}", e);
            EmbeddingResult {
                data: std::ptr::null_mut(),
                length: 0,
                error: true
            }
        }
    }
}

// Calculate similarity between two texts (called from Go)
#[no_mangle]
pub extern "C" fn calculate_similarity(text1: *const c_char, text2: *const c_char) -> f32 {
    let text1 = unsafe {
        match CStr::from_ptr(text1).to_str() {
            Ok(s) => s,
            Err(_) => return -1.0,
        }
    };
    
    let text2 = unsafe {
        match CStr::from_ptr(text2).to_str() {
            Ok(s) => s,
            Err(_) => return -1.0,
        }
    };

    let bert_opt = BERT_SIMILARITY.lock().unwrap();
    let bert = match &*bert_opt {
        Some(b) => b,
        None => {
            eprintln!("BERT model not initialized");
            return -1.0;
        }
    };

    match bert.calculate_similarity(text1, text2) {
        Ok(similarity) => similarity,
        Err(e) => {
            eprintln!("Error calculating similarity: {}", e);
            -1.0
        }
    }
}

// Find most similar text from a list (called from Go)
#[no_mangle]
pub extern "C" fn find_most_similar(
    query: *const c_char, 
    candidates_ptr: *const *const c_char,
    num_candidates: i32
) -> SimilarityResult {
    let query = unsafe {
        match CStr::from_ptr(query).to_str() {
            Ok(s) => s,
            Err(_) => return SimilarityResult { index: -1, score: -1.0 },
        }
    };
    
    // Convert the array of C strings to Rust strings
    let candidates: Vec<&str> = unsafe {
        let mut result = Vec::with_capacity(num_candidates as usize);
        let candidates_slice = std::slice::from_raw_parts(candidates_ptr, num_candidates as usize);
        
        for &cstr in candidates_slice {
            match CStr::from_ptr(cstr).to_str() {
                Ok(s) => result.push(s),
                Err(_) => return SimilarityResult { index: -1, score: -1.0 },
            }
        }
        
        result
    };

    let bert_opt = BERT_SIMILARITY.lock().unwrap();
    let bert = match &*bert_opt {
        Some(b) => b,
        None => {
            eprintln!("BERT model not initialized");
            return SimilarityResult { index: -1, score: -1.0 };
        }
    };

    match bert.find_most_similar(query, &candidates) {
        Ok((idx, score)) => SimilarityResult { 
            index: idx as i32, 
            score 
        },
        Err(e) => {
            eprintln!("Error finding most similar: {}", e);
            SimilarityResult { index: -1, score: -1.0 }
        }
    }
}

// Free a C string allocated by Rust
#[no_mangle]
pub extern "C" fn free_cstring(s: *mut c_char) {
    unsafe {
        if !s.is_null() {
            let _ = CString::from_raw(s);
        }
    }
}

// Free embedding data allocated by Rust
#[no_mangle]
pub extern "C" fn free_embedding(data: *mut f32, length: i32) {
    if !data.is_null() && length > 0 {
        unsafe {
            // Reconstruct the vector so that Rust can properly deallocate it
            let _vec = Vec::from_raw_parts(data, length as usize, length as usize);
            // The vector will be dropped and the memory freed when _vec goes out of scope
        }
    }
}

// Helper function to L2 normalize a tensor
fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    let norm = v.sqr()?.sum_keepdim(1)?.sqrt()?;
    Ok(v.broadcast_div(&norm)?)
} 