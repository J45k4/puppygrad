use std::error;
use std::fmt;
use std::fs;
use std::path::Path;

use safetensors::tensor::TensorView;
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
    MisalignedF16 {
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
            SafeTensorLoadError::MisalignedF16 { name, byte_len } => write!(
                f,
                "tensor {name} byte length {byte_len} is not divisible by 2"
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

pub struct TensorStore<'a> {
    tensors: SafeTensors<'a>,
}

impl<'a> TensorStore<'a> {
    pub fn from_bytes(path: &Path, bytes: &'a [u8]) -> Result<Self> {
        parse_safetensors(path, bytes).map(Self::new)
    }

    pub fn new(tensors: SafeTensors<'a>) -> Self {
        Self { tensors }
    }

    pub fn required(&self, name: &str) -> Result<TensorView<'a>> {
        self.tensors
            .tensor(name)
            .map_err(|source| SafeTensorLoadError::TensorNotFound {
                name: name.to_string(),
                source,
            })
    }

    pub fn optional(&self, name: &str) -> Result<Option<TensorView<'a>>> {
        match self.tensors.tensor(name) {
            Ok(tensor) => Ok(Some(tensor)),
            Err(safetensors::SafeTensorError::TensorNotFound(_)) => Ok(None),
            Err(source) => Err(SafeTensorLoadError::TensorNotFound {
                name: name.to_string(),
                source,
            }),
        }
    }

    pub fn required_f32(&self, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
        let tensor = self.required(name)?;
        validate_tensor(name, &tensor, Dtype::F32, expected_shape)?;
        f32_data(name, &tensor)
    }

    pub fn required_f32_lossy(&self, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
        let tensor = self.required(name)?;
        validate_tensor_shape(name, &tensor, expected_shape)?;
        tensor_data_as_f32(name, &tensor)
    }

    pub fn optional_f32(&self, name: &str, expected_shape: &[usize]) -> Result<Option<Vec<f32>>> {
        let Some(tensor) = self.optional(name)? else {
            return Ok(None);
        };
        validate_tensor(name, &tensor, Dtype::F32, expected_shape)?;
        f32_data(name, &tensor).map(Some)
    }

    pub fn names(&self) -> Vec<&str> {
        self.tensors.names()
    }
}

pub fn tensor_f32(
    store: &TensorStore<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>> {
    store.required_f32(name, expected_shape)
}

pub fn validate_tensor(
    name: &str,
    tensor: &TensorView<'_>,
    expected_dtype: Dtype,
    expected_shape: &[usize],
) -> Result<()> {
    if tensor.dtype() != expected_dtype {
        return Err(SafeTensorLoadError::WrongDtype {
            name: name.to_string(),
            actual: tensor.dtype(),
            expected: expected_dtype,
        });
    }
    validate_tensor_shape(name, tensor, expected_shape)
}

pub fn validate_tensor_shape(
    name: &str,
    tensor: &TensorView<'_>,
    expected_shape: &[usize],
) -> Result<()> {
    if tensor.shape() != expected_shape {
        return Err(SafeTensorLoadError::WrongShape {
            name: name.to_string(),
            actual: tensor.shape().to_vec(),
            expected: expected_shape.to_vec(),
        });
    }
    Ok(())
}

pub fn tensor_data_as_f32(name: &str, tensor: &TensorView<'_>) -> Result<Vec<f32>> {
    match tensor.dtype() {
        Dtype::F32 => f32_data(name, tensor),
        Dtype::F16 => f16_data(name, tensor),
        Dtype::BF16 => bf16_data(name, tensor),
        actual => Err(SafeTensorLoadError::WrongDtype {
            name: name.to_string(),
            actual,
            expected: Dtype::F32,
        }),
    }
}

fn f32_data(name: &str, tensor: &TensorView<'_>) -> Result<Vec<f32>> {
    let data = tensor.data();
    if !data.len().is_multiple_of(4) {
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

fn f16_data(name: &str, tensor: &TensorView<'_>) -> Result<Vec<f32>> {
    let data = tensor.data();
    if !data.len().is_multiple_of(2) {
        return Err(SafeTensorLoadError::MisalignedF16 {
            name: name.to_string(),
            byte_len: data.len(),
        });
    }

    Ok(data
        .chunks_exact(2)
        .map(|bytes| {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect())
}

fn bf16_data(name: &str, tensor: &TensorView<'_>) -> Result<Vec<f32>> {
    let data = tensor.data();
    if !data.len().is_multiple_of(2) {
        return Err(SafeTensorLoadError::MisalignedF16 {
            name: name.to_string(),
            byte_len: data.len(),
        });
    }

    Ok(data
        .chunks_exact(2)
        .map(|bytes| {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            half::bf16::from_bits(bits).to_f32()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};

    #[test]
    fn tensor_store_reads_required_and_optional_f32_tensors() -> Result<()> {
        let values = [1.0f32, 2.5, -3.0, 4.0];
        let data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::F32, vec![2, 2], &data).unwrap();
        let bytes = serialize([("weight", view)], None).unwrap();
        let store = TensorStore::from_bytes(Path::new("memory.safetensors"), &bytes)?;

        assert_eq!(store.required_f32("weight", &[2, 2])?, values.to_vec());
        assert!(store.optional("missing")?.is_none());
        assert!(store.optional_f32("missing", &[1])?.is_none());
        Ok(())
    }

    #[test]
    fn tensor_store_validates_shape() {
        let data = [0u8; 4];
        let view = TensorView::new(Dtype::F32, vec![1], &data).unwrap();
        let bytes = serialize([("weight", view)], None).unwrap();
        let store = TensorStore::from_bytes(Path::new("memory.safetensors"), &bytes).unwrap();

        let err = store.required_f32("weight", &[2]).unwrap_err();

        assert!(matches!(err, SafeTensorLoadError::WrongShape { .. }));
    }

    #[test]
    fn tensor_store_converts_f16_and_bf16_to_f32() -> Result<()> {
        let f16_values = [half::f16::from_f32(1.5), half::f16::from_f32(-2.0)];
        let f16_data = f16_values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let bf16_values = [half::bf16::from_f32(3.0), half::bf16::from_f32(-4.5)];
        let bf16_data = bf16_values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let f16_view = TensorView::new(Dtype::F16, vec![2], &f16_data).unwrap();
        let bf16_view = TensorView::new(Dtype::BF16, vec![2], &bf16_data).unwrap();
        let bytes = serialize([("f16", f16_view), ("bf16", bf16_view)], None).unwrap();
        let store = TensorStore::from_bytes(Path::new("memory.safetensors"), &bytes)?;

        assert_eq!(store.required_f32_lossy("f16", &[2])?, vec![1.5, -2.0]);
        assert_eq!(store.required_f32_lossy("bf16", &[2])?, vec![3.0, -4.5]);
        Ok(())
    }
}
