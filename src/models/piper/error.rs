use std::error;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PiperError {
    InvalidConfig(String),
    InvalidInput(String),
    Unsupported(String),
}

impl fmt::Display for PiperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PiperError::InvalidConfig(msg) => write!(f, "invalid Piper config: {msg}"),
            PiperError::InvalidInput(msg) => write!(f, "invalid Piper input: {msg}"),
            PiperError::Unsupported(msg) => write!(f, "unsupported Piper operation: {msg}"),
        }
    }
}

impl error::Error for PiperError {}

pub type Result<T> = std::result::Result<T, PiperError>;
