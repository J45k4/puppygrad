use super::{Gpt2Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gpt2BackendName {
    Rust,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gpt2BackendConfig {
    Rust(Gpt2RustConfig),
}

impl Gpt2BackendConfig {
    pub fn rust(threads: usize) -> Result<Self> {
        let config = Gpt2RustConfig { threads };
        config.validate()?;
        Ok(Self::Rust(config))
    }

    pub fn name(&self) -> Gpt2BackendName {
        match self {
            Gpt2BackendConfig::Rust(_) => Gpt2BackendName::Rust,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Gpt2BackendConfig::Rust(config) => format!("rust (threads={})", config.threads),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gpt2RustConfig {
    pub threads: usize,
}

impl Default for Gpt2RustConfig {
    fn default() -> Self {
        Self { threads: 1 }
    }
}

impl Gpt2RustConfig {
    pub fn validate(&self) -> Result<()> {
        if self.threads == 0 {
            return Err(Gpt2Error::InvalidConfig(
                "rust backend threads must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gpt2GenerationConfig {
    pub max_new_tokens: usize,
}

impl Gpt2GenerationConfig {
    pub fn new(max_new_tokens: usize) -> Self {
        Self { max_new_tokens }
    }

    pub fn validate(&self) -> Result<()> {
        Ok(())
    }
}
