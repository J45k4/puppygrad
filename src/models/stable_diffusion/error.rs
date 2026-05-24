use std::error;
use std::fmt;
use std::io;

#[derive(Debug)]
pub enum StableDiffusionError {
    Asset(String),
    Backend(String),
    Config(String),
    Image(String),
    InvalidInput(String),
    Io(io::Error),
    Python(String),
    Unsupported(String),
}

impl fmt::Display for StableDiffusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asset(message) => write!(f, "Stable Diffusion asset error: {message}"),
            Self::Backend(message) => write!(f, "Stable Diffusion backend error: {message}"),
            Self::Config(message) => write!(f, "invalid Stable Diffusion config: {message}"),
            Self::Image(message) => write!(f, "Stable Diffusion image error: {message}"),
            Self::InvalidInput(message) => write!(f, "invalid Stable Diffusion input: {message}"),
            Self::Io(err) => write!(f, "Stable Diffusion I/O error: {err}"),
            Self::Python(message) => write!(f, "Stable Diffusion python backend error: {message}"),
            Self::Unsupported(message) => {
                write!(f, "unsupported Stable Diffusion operation: {message}")
            }
        }
    }
}

impl error::Error for StableDiffusionError {}

impl From<io::Error> for StableDiffusionError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, StableDiffusionError>;
