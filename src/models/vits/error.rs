use std::error;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VitsError {
    InvalidInput(String),
    Unsupported(String),
}

impl fmt::Display for VitsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VitsError::InvalidInput(msg) => write!(f, "invalid VITS input: {msg}"),
            VitsError::Unsupported(msg) => write!(f, "unsupported VITS operation: {msg}"),
        }
    }
}

impl error::Error for VitsError {}

pub type Result<T> = std::result::Result<T, VitsError>;
