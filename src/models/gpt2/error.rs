use std::error;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gpt2Error {
    Asset(String),
    InvalidConfig(String),
    InvalidInput(String),
    InvalidWeights(String),
}

impl fmt::Display for Gpt2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Gpt2Error::Asset(msg) => write!(f, "GPT-2 asset error: {msg}"),
            Gpt2Error::InvalidConfig(msg) => write!(f, "invalid GPT-2 config: {msg}"),
            Gpt2Error::InvalidInput(msg) => write!(f, "invalid GPT-2 input: {msg}"),
            Gpt2Error::InvalidWeights(msg) => write!(f, "invalid GPT-2 weights: {msg}"),
        }
    }
}

impl error::Error for Gpt2Error {}

pub type Result<T> = std::result::Result<T, Gpt2Error>;
