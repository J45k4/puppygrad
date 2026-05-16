use std::error;
use std::fmt;
use std::fs;
use std::path::Path;

use safetensors::{Dtype, SafeTensors};

#[derive(Debug)]
pub enum SafeTensorLoadError {
    ReadFile {
        path: String,
        source: std::io::Error,
    },
    ParseFile {
        path: String,
        source: safetensors::SafeTensorError,
    },
    TensorNotFound {
        name: String,
        source: safetensors::SafeTensorError,
    },
    WrongDtype {
        name: String,
        actual: Dtype,
        expected: Dtype,
    },
    WrongShape {
        name: String,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
    MisalignedF32 {
        name: String,
        byte_len: usize,
    },
}

impl fmt::Display for SafeTensorLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafeTensorLoadError::ReadFile { path, source } => {
                write!(f, "failed to read safetensors file {path}: {source}")
            }
            SafeTensorLoadError::ParseFile { path, source } => {
                write!(f, "failed to parse safetensors file {path}: {source}")
            }
            SafeTensorLoadError::TensorNotFound { name, source } => {
                write!(f, "failed to read tensor {name}: {source}")
            }
            SafeTensorLoadError::WrongDtype {
                name,
                actual,
                expected,
            } => write!(
                f,
                "tensor {name} has dtype {actual:?}, expected {expected:?}"
            ),
            SafeTensorLoadError::WrongShape {
                name,
                actual,
                expected,
            } => write!(
                f,
                "tensor {name} shape {actual:?} does not match expected {expected:?}"
            ),
            SafeTensorLoadError::MisalignedF32 { name, byte_len } => write!(
                f,
                "tensor {name} byte length {byte_len} is not divisible by 4"
            ),
        }
    }
}

impl error::Error for SafeTensorLoadError {}

pub type Result<T> = std::result::Result<T, SafeTensorLoadError>;

pub fn read_safetensors_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|source| SafeTensorLoadError::ReadFile {
        path: path.display().to_string(),
        source,
    })
}

pub fn parse_safetensors<'a>(path: &Path, bytes: &'a [u8]) -> Result<SafeTensors<'a>> {
    SafeTensors::deserialize(bytes).map_err(|source| SafeTensorLoadError::ParseFile {
        path: path.display().to_string(),
        source,
    })
}

pub fn tensor_f32(
    tensors: &SafeTensors<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .map_err(|source| SafeTensorLoadError::TensorNotFound {
            name: name.to_string(),
            source,
        })?;
    if tensor.dtype() != Dtype::F32 {
        return Err(SafeTensorLoadError::WrongDtype {
            name: name.to_string(),
            actual: tensor.dtype(),
            expected: Dtype::F32,
        });
    }
    if tensor.shape() != expected_shape {
        return Err(SafeTensorLoadError::WrongShape {
            name: name.to_string(),
            actual: tensor.shape().to_vec(),
            expected: expected_shape.to_vec(),
        });
    }

    let data = tensor.data();
    if data.len() % 4 != 0 {
        return Err(SafeTensorLoadError::MisalignedF32 {
            name: name.to_string(),
            byte_len: data.len(),
        });
    }

    Ok(data
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}
