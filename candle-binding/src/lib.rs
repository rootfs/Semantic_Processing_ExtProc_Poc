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
struct BertSimilarity {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

lazy_static::lazy_static! {
    static ref BERT_SIMILARITY: Arc<Mutex<Option<BertSimilarity>>> = Arc::new(Mutex::new(None));
}

impl BertSimilarity {
    fn new(model_id: &str, use_cpu: bool) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Load model and tokenizer from Hugging Face Hub
        let repo = Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        );

        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = api.get("model.safetensors")?;
            (config, tokenizer, weights)
        };

        let config = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Use approximate GELU for better performance
        config.hidden_act = HiddenAct::GeluApproximate;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    // Get embedding for a text
    fn get_embedding(&self, text: &str) -> Result<Tensor> {
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(E::msg)?;
        
        let token_ids = encoding.get_ids().to_vec();
        let token_ids_tensor = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids_tensor.zeros_like()?;
        
        // Run the text through BERT
        let embeddings = self.model.forward(&token_ids_tensor, &token_type_ids, None)?;
        
        // Extract the CLS token (first token) embedding
        let cls_embedding = embeddings.narrow(1, 0, 1)?.squeeze(1)?;
        
        // Convert to the right type and normalize
        let embedding = cls_embedding.to_dtype(DType::F32)?;
        
        // Normalize the embedding (L2 norm)
        let norm = embedding.sqr()?.sum_all()?.sqrt()?;
        let normalized = embedding.broadcast_div(&norm)?;
        
        Ok(normalized)
    }

    // Calculate cosine similarity between two texts
    fn calculate_similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        let embedding1 = self.get_embedding(text1)?;
        let embedding2 = self.get_embedding(text2)?;
        
        // Calculate cosine similarity
        let similarity = embedding1.matmul(&embedding2.transpose(0, 1)?)?;
        
        // Convert to scalar
        let sim_value = similarity.to_vec0::<f32>()?;
        
        Ok(sim_value)
    }

    // Find most similar text from a list
    fn find_most_similar(&self, query_text: &str, candidates: &[&str]) -> Result<(usize, f32)> {
        if candidates.is_empty() {
            return Err(E::msg("Empty candidate list"));
        }
        
        let query_embedding = self.get_embedding(query_text)?;
        
        // Track best match
        let mut best_idx = 0;
        let mut best_score = -1.0;
        
        for (idx, candidate) in candidates.iter().enumerate() {
            let candidate_embedding = self.get_embedding(candidate)?;
            
            // Calculate similarity
            let similarity = query_embedding.matmul(&candidate_embedding.transpose(0, 1)?)?;
            let sim_value = similarity.to_vec0::<f32>()?;
            
            if sim_value > best_score {
                best_score = sim_value;
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