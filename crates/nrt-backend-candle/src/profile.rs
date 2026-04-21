use nrt_core::ModelId;
use serde::{Deserialize, Serialize};

/// Describes which HuggingFace repo to pull weights/tokenizer from, plus
/// prompt-shaping knobs (system prompt, tokenizer template, stop tokens).
/// Profiles are registered per-ModelId at backend construction time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    /// HuggingFace repo that holds the GGUF file (e.g., "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF").
    pub gguf_repo: String,
    /// File name inside the GGUF repo (e.g., "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf").
    pub gguf_file: String,
    /// HuggingFace repo that holds `tokenizer.json` (usually the non-quantized repo).
    pub tokenizer_repo: String,
    /// File name for the tokenizer (default: "tokenizer.json").
    pub tokenizer_file: String,

    /// System / prefix header prepended to every prompt. For TinyLlama-chat,
    /// this looks like `<|system|>\n{sys}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n`.
    pub prompt: PromptHeader,

    /// Nominal VRAM footprint for tier accounting.
    pub nominal_vram_mb: u64,
    /// Nominal system-RAM footprint for Standby tier.
    pub nominal_ram_mb: u64,

    /// If this model is configured as a router, the set of intent labels it
    /// is allowed to emit. The generator clamps its output to one of these
    /// via a simple substring match on the first ~20 tokens.
    #[serde(default)]
    pub router: Option<RouterConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptHeader {
    pub system: String,
    /// Format string with `{user}` placeholder, e.g.,
    /// `"<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n"`.
    pub template: String,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

impl PromptHeader {
    pub fn apply(&self, user: &str) -> String {
        let with_sys = self.template.replace("{system}", &self.system);
        with_sys.replace("{user}", user)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Intent labels the router may emit. The first one that appears as a
    /// substring of the model's generation is returned as the intent.
    pub intents: Vec<String>,
    /// Keyword triggers per intent. If the model's free-form output doesn't
    /// contain the intent label verbatim (common with small weak classifiers
    /// like TinyLlama 1.1B), we fall back to checking whether any trigger
    /// keyword associated with an intent appears in the output.
    /// Example: {"billing": ["refund", "charge", "payment"], ...}
    #[serde(default)]
    pub keyword_triggers: Vec<(String, Vec<String>)>,
    /// Override the default system prompt for routers, which should be
    /// something like: "Classify into one of: billing, technical, legal. Output ONLY the label."
    pub system_override: Option<String>,
}

/// A starter profile for TinyLlama-1.1B-Chat-v1.0 Q4_K_M. Useful default for
/// the demo; override `system` and `router` per model at registration time.
pub fn default_tinyllama_profile() -> ModelProfile {
    ModelProfile {
        gguf_repo: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into(),
        gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".into(),
        // Tokenizer fetched via tokenizers::Tokenizer::from_pretrained (http feature).
        tokenizer_repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".into(),
        tokenizer_file: "tokenizer.json".into(),
        prompt: PromptHeader {
            system: "You are a concise assistant.".into(),
            template: "<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n".into(),
            stop_sequences: vec!["</s>".into(), "<|user|>".into(), "<|system|>".into()],
        },
        nominal_vram_mb: 720,
        nominal_ram_mb: 900,
        router: None,
    }
}

/// Router profile variant that classifies intents. Also populates sensible
/// keyword triggers per known customer-support intent so that TinyLlama's
/// non-compliant free-form output still routes correctly most of the time.
/// For production, the router is a fine-tuned classifier and label-verbatim
/// matching is sufficient — the keyword triggers are a demo-time safety net.
pub fn router_profile(intents: Vec<String>) -> ModelProfile {
    let mut p = default_tinyllama_profile();
    let list = intents.join(", ");
    p.prompt.system = format!(
        "Intent classifier. Pick one of: {list}. Output ONLY the label. \
         One word. No explanation. No punctuation. Your entire reply = the label."
    );
    let keyword_triggers = default_keyword_triggers(&intents);
    p.router = Some(RouterConfig {
        intents,
        keyword_triggers,
        system_override: None,
    });
    p
}

/// Known customer-support keyword hints. If a specialist id isn't in this
/// table, its only trigger is its own label.
fn default_keyword_triggers(intents: &[String]) -> Vec<(String, Vec<String>)> {
    intents
        .iter()
        .map(|i| {
            let label = i.to_string();
            let mut words = vec![label.clone()];
            match label.as_str() {
                "billing" => words.extend([
                    "refund", "charge", "charged", "payment", "paid", "bill", "subscription",
                    "card", "invoice", "receipt", "pricing", "money",
                ].iter().map(|s| s.to_string())),
                "technical" => words.extend([
                    "error", "500", "502", "404", "debug", "login", "server", "bug", "crash",
                    "timeout", "latency", "api", "endpoint", "config",
                ].iter().map(|s| s.to_string())),
                "legal" => words.extend([
                    "law", "legal", "arbitration", "contract", "tos", "terms", "enforce",
                    "jurisdiction", "privacy", "gdpr", "copyright", "license",
                ].iter().map(|s| s.to_string())),
                _ => {}
            }
            (label, words)
        })
        .collect()
}

/// Specialist profile that answers as domain expert `domain`.
pub fn specialist_profile(domain: &str) -> ModelProfile {
    let mut p = default_tinyllama_profile();
    p.prompt.system = format!(
        "You are the {domain} specialist in a customer-support agent. \
         Answer concisely, in under 25 words, in the {domain} domain only."
    );
    p
}
