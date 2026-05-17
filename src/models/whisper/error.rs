use std::error;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WhisperError {
    Asset(String),
    Audio(String),
    InvalidConfig(String),
    InvalidInput(String),
    InvalidWeights(String),
    Unsupported(String),
}

impl fmt::Display for WhisperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WhisperError::Asset(msg) => write!(f, "Whisper asset error: {msg}"),
            WhisperError::Audio(msg) => write!(f, "Whisper audio error: {msg}"),
            WhisperError::InvalidConfig(msg) => write!(f, "invalid Whisper config: {msg}"),
            WhisperError::InvalidInput(msg) => write!(f, "invalid Whisper input: {msg}"),
            WhisperError::InvalidWeights(msg) => write!(f, "invalid Whisper weights: {msg}"),
            WhisperError::Unsupported(msg) => write!(f, "unsupported Whisper operation: {msg}"),
        }
    }
}

impl error::Error for WhisperError {}

pub type Result<T> = std::result::Result<T, WhisperError>;
