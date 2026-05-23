use std::error;
use std::fmt;

#[derive(Debug)]
pub enum BarkError {
    Asset(String),
    InvalidConfig(String),
    InvalidInput(String),
    Tokenizer(String),
    Unsupported(String),
}

impl fmt::Display for BarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BarkError::Asset(msg) => write!(f, "Bark asset error: {msg}"),
            BarkError::InvalidConfig(msg) => write!(f, "invalid Bark config: {msg}"),
            BarkError::InvalidInput(msg) => write!(f, "invalid Bark input: {msg}"),
            BarkError::Tokenizer(msg) => write!(f, "Bark tokenizer error: {msg}"),
            BarkError::Unsupported(msg) => write!(f, "unsupported Bark operation: {msg}"),
        }
    }
}

impl error::Error for BarkError {}

pub type Result<T> = std::result::Result<T, BarkError>;
