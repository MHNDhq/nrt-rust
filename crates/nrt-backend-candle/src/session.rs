//! Per-request Candle inference loop.
//!
//! The loop is short and intentional: apply the chat template, tokenize,
//! feed the prompt tokens in one forward pass (prefill), then generate one
//! token at a time until we hit max_tokens or a stop sequence. Greedy ArgMax
//! decoding keeps specialist output deterministic enough to benchmark against.

use crate::{CandleError, ModelProfile};
use candle_core::{Device, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_llama::ModelWeights,
};
use std::sync::Mutex as StdMutex;
use tokenizers::Tokenizer;

/// Abstraction over `LoadedModel` so tests can swap in a stub.
pub trait InferenceHost: Send + Sync {
    fn tokenizer(&self) -> &Tokenizer;
    fn device(&self) -> &Device;
    fn weights(&self) -> &StdMutex<ModelWeights>;
    fn kv_pos(&self) -> &StdMutex<usize>;
    fn profile(&self) -> &ModelProfile;
}

pub fn generate(
    host: &dyn InferenceHost,
    user_prompt: &str,
    max_tokens: usize,
) -> Result<(String, Option<String>, u32), CandleError> {
    let profile = host.profile();
    let prompt = profile.prompt.apply(user_prompt);

    let tokenizer = host.tokenizer();
    let encoded = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| CandleError::Tokenizer(e.to_string()))?;
    let prompt_tokens: Vec<u32> = encoded.get_ids().to_vec();
    if prompt_tokens.is_empty() {
        return Err(CandleError::EmptyGeneration);
    }

    let device = host.device().clone();

    // Fresh KV cache for each generation. The spec's Per-Session Linear Memory
    // pillar is about *not* thrashing this across generations within a session;
    // that's a future milestone. For now, isolation per call is correct.
    *host.kv_pos().lock().unwrap() = 0;

    let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut processor = LogitsProcessor::from_sampling(42, Sampling::ArgMax);

    let mut last_token: u32 = {
        let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
        let mut weights = host.weights().lock().unwrap();
        let logits = weights.forward(&input, 0)?.squeeze(0)?;
        let mut pos_guard = host.kv_pos().lock().unwrap();
        *pos_guard = prompt_tokens.len();
        drop(pos_guard);
        processor.sample(&logits)?
    };
    generated_tokens.push(last_token);

    for _ in 1..max_tokens {
        if is_eos(last_token) || is_stop_hit(&generated_tokens, tokenizer, profile) {
            break;
        }
        let pos = {
            let mut pos_guard = host.kv_pos().lock().unwrap();
            let pos = *pos_guard;
            *pos_guard = pos + 1;
            pos
        };
        let input = Tensor::new(&[last_token], &device)?.unsqueeze(0)?;
        let next = {
            let mut weights = host.weights().lock().unwrap();
            let logits = weights.forward(&input, pos)?.squeeze(0)?;
            processor.sample(&logits)?
        };
        generated_tokens.push(next);
        last_token = next;
    }

    let raw_completion = tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| CandleError::Tokenizer(e.to_string()))?;

    let completion = trim_stop_sequences(&raw_completion, profile);

    // If this model is registered as a router, map output to intent. Try:
    //   1. Exact label match (lowercase substring) — what a perfect classifier would produce.
    //   2. Keyword-trigger match — fallback for weak classifiers that answer
    //      the question instead of following the format instruction.
    // Ties are broken by declaration order in the intent list.
    let intent = profile.router.as_ref().and_then(|rc| {
        let lower = completion.to_lowercase();
        if let Some(direct) = rc
            .intents
            .iter()
            .find(|i| lower.contains(&i.to_lowercase()))
        {
            return Some(direct.clone());
        }
        for (intent, triggers) in &rc.keyword_triggers {
            if triggers.iter().any(|k| lower.contains(&k.to_lowercase())) {
                return Some(intent.clone());
            }
        }
        None
    });

    Ok((completion, intent, generated_tokens.len() as u32))
}

fn trim_stop_sequences(s: &str, profile: &ModelProfile) -> String {
    let mut out = s.to_string();
    for stop in &profile.prompt.stop_sequences {
        if let Some(idx) = out.find(stop) {
            out.truncate(idx);
        }
    }
    out.trim().to_string()
}

/// TinyLlama uses token id 2 as </s> (EOS). We hardcode it for the prototype.
fn is_eos(token: u32) -> bool {
    token == 2
}

fn is_stop_hit(tokens: &[u32], tokenizer: &Tokenizer, profile: &ModelProfile) -> bool {
    let Ok(text) = tokenizer.decode(tokens, true) else {
        return false;
    };
    profile
        .prompt
        .stop_sequences
        .iter()
        .any(|stop| text.contains(stop))
}
