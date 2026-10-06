//! Tensor file loaders. Formats describe arrays, never model or training behavior.
pub mod csv;
pub mod idx;
use crate::compiler::{cpu::Tensor, pop::DType};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, io::Read, path::Path};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    U8,
    I8,
    I16,
    I32,
    F32,
    F64,
    Bool,
}
impl DataType {
    pub fn size(self) -> usize {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::I16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
    pub fn compiler(self) -> Result<DType> {
        Ok(match self {
            Self::U8 => DType::U8,
            Self::I32 => DType::I32,
            Self::F32 => DType::F32,
            Self::Bool => DType::Bool,
            _ => {
                return Err(format!(
                    "file tensor dtype {self:?} is not supported by the compute compiler yet"
                )
                .into())
            }
        })
    }
}
/// Contiguous, little-endian elements; no normalization, flattening or inferred casts.
#[derive(Debug)]
pub struct LoadedTensor {
    pub dtype: DataType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}
impl LoadedTensor {
    pub fn rows(&self) -> Result<usize> {
        self.shape
            .first()
            .copied()
            .ok_or_else(|| "a batched tensor needs a sample dimension".into())
    }
    pub fn batch(&self, rows: &[usize], batch_size: usize) -> Result<Tensor> {
        if rows.len() > batch_size {
            return Err("too many rows for batch".into());
        }
        let count = self.rows()?;
        let stride = elements(&self.shape[1..])?
            .checked_mul(self.dtype.size())
            .ok_or("row byte size overflow")?;
        let mut bytes = vec![
            0;
            stride
                .checked_mul(batch_size)
                .ok_or("batch byte size overflow")?
        ];
        for (i, &row) in rows.iter().enumerate() {
            if row >= count {
                return Err("dataset row out of bounds".into());
            }
            bytes[i * stride..(i + 1) * stride]
                .copy_from_slice(&self.bytes[row * stride..(row + 1) * stride]);
        }
        Ok(match self.dtype {
            DataType::U8 => Tensor::U8(bytes.into()),
            DataType::Bool => Tensor::Bool(bytes.into()),
            DataType::I32 => Tensor::I32(
                bytes
                    .chunks_exact(4)
                    .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
                    .collect::<Vec<_>>()
                    .into(),
            ),
            DataType::F32 => Tensor::F32(
                bytes
                    .chunks_exact(4)
                    .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                    .collect::<Vec<_>>()
                    .into(),
            ),
            _ => {
                return Err(format!("cannot batch {:?} into a compiler tensor", self.dtype).into())
            }
        })
    }
    pub fn class_label(&self, row: usize) -> Result<usize> {
        if elements(&self.shape[1..])? != 1 || row >= self.rows()? {
            return Err("classification labels must be scalar rows".into());
        }
        match self.dtype {
            DataType::U8 => Ok(self.bytes[row] as usize),
            DataType::I32 => Ok(usize::try_from(i32::from_le_bytes(
                self.bytes[row * 4..row * 4 + 4].try_into().unwrap(),
            ))?),
            _ => Err("classification labels require u8 or i32".into()),
        }
    }
}
pub fn elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b)
            .ok_or_else(|| "tensor shape overflow".into())
    })
}
pub type Tensors = BTreeMap<String, LoadedTensor>;
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Idx,
    Csv,
    Safetensors,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadOptions {
    pub format: Option<Format>,
    pub csv: Option<csv::Schema>,
}
/// Compression and format are separate layers. Extensions hint; parsers validate.
pub fn load(path: &Path, options: &LoadOptions) -> Result<Tensors> {
    let bytes = fs::read(path)?;
    let gzip = bytes.starts_with(&[0x1f, 0x8b]);
    if path.extension().is_some_and(|e| e == "gz") && !gzip {
        return Err(".gz input has no gzip header".into());
    }
    let bytes = if gzip {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&bytes[..])
            .take(1024 * 1024 * 1024 + 1)
            .read_to_end(&mut out)?;
        if out.len() > 1024 * 1024 * 1024 {
            return Err("decompressed tensor file exceeds 1 GiB".into());
        }
        out
    } else {
        bytes
    };
    let inner = if path.extension().is_some_and(|e| e == "gz") {
        Path::new(path.file_stem().unwrap())
    } else {
        path
    };
    let format = options
        .format
        .or_else(|| match inner.extension().and_then(|e| e.to_str()) {
            Some("csv") => Some(Format::Csv),
            Some("idx") => Some(Format::Idx),
            Some("safetensors") => Some(Format::Safetensors),
            _ => None,
        })
        .or_else(|| {
            if bytes.starts_with(&[0, 0]) {
                Some(Format::Idx)
            } else {
                None
            }
        })
        .ok_or("unknown tensor file format; specify format explicitly")?;
    match format {
        Format::Idx => Ok(BTreeMap::from([("tensor".into(), idx::read(&bytes)?)])),
        Format::Csv => csv::read(
            &bytes,
            options
                .csv
                .as_ref()
                .ok_or("CSV needs an explicit column schema")?,
        ),
        Format::Safetensors => {
            let file = safetensors::SafeTensors::deserialize(&bytes)?;
            let mut result = BTreeMap::new();
            for name in file.names() {
                let view = file.tensor(name)?;
                use safetensors::Dtype;
                let dtype = match view.dtype() {
                    Dtype::U8 => DataType::U8,
                    Dtype::I8 => DataType::I8,
                    Dtype::I16 => DataType::I16,
                    Dtype::I32 => DataType::I32,
                    Dtype::F32 => DataType::F32,
                    Dtype::F64 => DataType::F64,
                    Dtype::BOOL => DataType::Bool,
                    d => return Err(format!("unsupported safetensors dtype {d:?}").into()),
                };
                result.insert(
                    name.to_owned(),
                    LoadedTensor {
                        dtype,
                        shape: view.shape().to_vec(),
                        bytes: view.data().to_vec(),
                    },
                );
            }
            Ok(result)
        }
    }
}
