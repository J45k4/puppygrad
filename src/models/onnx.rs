use std::collections::BTreeMap;
use std::error;
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnnxTensorType {
    Float32,
    Int64,
    Other(i32),
}

impl OnnxTensorType {
    fn from_i32(value: i32) -> Self {
        match value {
            1 => Self::Float32,
            7 => Self::Int64,
            other => Self::Other(other),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OnnxTensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub data_type: OnnxTensorType,
    pub raw_data: Vec<u8>,
    pub float_data: Vec<f32>,
    pub int64_data: Vec<i64>,
}

impl OnnxTensor {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn storage_bytes(&self) -> usize {
        if !self.raw_data.is_empty() {
            return self.raw_data.len();
        }
        self.float_data.len() * std::mem::size_of::<f32>()
            + self.int64_data.len() * std::mem::size_of::<i64>()
    }

    pub fn f32_values(&self) -> Result<Vec<f32>> {
        if self.data_type != OnnxTensorType::Float32 {
            return Err(OnnxLoadError::WrongDtype {
                name: self.name.clone(),
                actual: self.data_type,
                expected: OnnxTensorType::Float32,
            });
        }
        if !self.raw_data.is_empty() {
            if !self.raw_data.len().is_multiple_of(4) {
                return Err(OnnxLoadError::MisalignedRawData {
                    name: self.name.clone(),
                    dtype: self.data_type,
                    byte_len: self.raw_data.len(),
                });
            }
            return Ok(self
                .raw_data
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect());
        }
        Ok(self.float_data.clone())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OnnxInitializerStore {
    tensors: BTreeMap<String, OnnxTensor>,
}

impl OnnxInitializerStore {
    pub fn from_model_bytes(bytes: &[u8]) -> Result<Self> {
        let mut parser = ProtoParser::new(bytes);
        let mut tensors = BTreeMap::new();
        while !parser.is_done() {
            let Some((field, wire)) = parser.read_key()? else {
                break;
            };
            if field == 7 && wire == WIRE_LEN {
                let graph = parser.read_len()?;
                parse_graph(graph, &mut tensors)?;
            } else {
                parser.skip_value(wire)?;
            }
        }
        Ok(Self { tensors })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&OnnxTensor> {
        self.tensors.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &OnnxTensor> {
        self.tensors.values()
    }

    pub fn required(&self, name: &str) -> Result<&OnnxTensor> {
        self.get(name)
            .ok_or_else(|| OnnxLoadError::TensorNotFound(name.to_string()))
    }

    pub fn required_f32(&self, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
        let tensor = self.required(name)?;
        if tensor.dims != expected_shape {
            return Err(OnnxLoadError::WrongShape {
                name: name.to_string(),
                actual: tensor.dims.clone(),
                expected: expected_shape.to_vec(),
            });
        }
        tensor.f32_values()
    }
}

#[derive(Debug)]
pub enum OnnxLoadError {
    ReadFile {
        path: String,
        source: std::io::Error,
    },
    InvalidWire(String),
    TensorNotFound(String),
    WrongDtype {
        name: String,
        actual: OnnxTensorType,
        expected: OnnxTensorType,
    },
    WrongShape {
        name: String,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
    MisalignedRawData {
        name: String,
        dtype: OnnxTensorType,
        byte_len: usize,
    },
}

impl fmt::Display for OnnxLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxLoadError::ReadFile { path, source } => {
                write!(f, "failed to read ONNX file {path}: {source}")
            }
            OnnxLoadError::InvalidWire(msg) => write!(f, "invalid ONNX protobuf: {msg}"),
            OnnxLoadError::TensorNotFound(name) => {
                write!(f, "ONNX initializer tensor {name} was not found")
            }
            OnnxLoadError::WrongDtype {
                name,
                actual,
                expected,
            } => write!(
                f,
                "ONNX initializer {name} has dtype {actual:?}, expected {expected:?}"
            ),
            OnnxLoadError::WrongShape {
                name,
                actual,
                expected,
            } => write!(
                f,
                "ONNX initializer {name} shape {actual:?} does not match expected {expected:?}"
            ),
            OnnxLoadError::MisalignedRawData {
                name,
                dtype,
                byte_len,
            } => write!(
                f,
                "ONNX initializer {name} raw {dtype:?} data byte length {byte_len} is misaligned"
            ),
        }
    }
}

impl error::Error for OnnxLoadError {}

pub type Result<T> = std::result::Result<T, OnnxLoadError>;

pub fn read_onnx_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|source| OnnxLoadError::ReadFile {
        path: path.display().to_string(),
        source,
    })
}

pub fn load_onnx_initializers(path: &Path) -> Result<OnnxInitializerStore> {
    let bytes = read_onnx_file(path)?;
    OnnxInitializerStore::from_model_bytes(&bytes)
}

const WIRE_VARINT: u8 = 0;
const WIRE_FIXED64: u8 = 1;
const WIRE_LEN: u8 = 2;
const WIRE_FIXED32: u8 = 5;

fn parse_graph(bytes: &[u8], tensors: &mut BTreeMap<String, OnnxTensor>) -> Result<()> {
    let mut parser = ProtoParser::new(bytes);
    while !parser.is_done() {
        let Some((field, wire)) = parser.read_key()? else {
            break;
        };
        if field == 5 && wire == WIRE_LEN {
            let tensor = parse_tensor(parser.read_len()?)?;
            if !tensor.name.is_empty() {
                tensors.insert(tensor.name.clone(), tensor);
            }
        } else {
            parser.skip_value(wire)?;
        }
    }
    Ok(())
}

fn parse_tensor(bytes: &[u8]) -> Result<OnnxTensor> {
    let mut parser = ProtoParser::new(bytes);
    let mut tensor = OnnxTensor {
        name: String::new(),
        dims: Vec::new(),
        data_type: OnnxTensorType::Other(0),
        raw_data: Vec::new(),
        float_data: Vec::new(),
        int64_data: Vec::new(),
    };

    while !parser.is_done() {
        let Some((field, wire)) = parser.read_key()? else {
            break;
        };
        match (field, wire) {
            (1, WIRE_VARINT) => tensor.dims.push(varint_to_usize(parser.read_varint()?)?),
            (1, WIRE_LEN) => {
                let mut packed = ProtoParser::new(parser.read_len()?);
                while !packed.is_done() {
                    tensor.dims.push(varint_to_usize(packed.read_varint()?)?);
                }
            }
            (2, WIRE_VARINT) => {
                tensor.data_type = OnnxTensorType::from_i32(varint_to_i32(parser.read_varint()?)?)
            }
            (4, WIRE_FIXED32) => tensor
                .float_data
                .push(f32::from_bits(parser.read_fixed32()?)),
            (4, WIRE_LEN) => {
                let packed = parser.read_len()?;
                if !packed.len().is_multiple_of(4) {
                    return Err(OnnxLoadError::InvalidWire(format!(
                        "packed float_data length {} is not divisible by 4",
                        packed.len()
                    )));
                }
                tensor.float_data.extend(
                    packed
                        .chunks_exact(4)
                        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
                );
            }
            (7, WIRE_VARINT) => tensor
                .int64_data
                .push(varint_to_i64(parser.read_varint()?)?),
            (7, WIRE_LEN) => {
                let mut packed = ProtoParser::new(parser.read_len()?);
                while !packed.is_done() {
                    tensor
                        .int64_data
                        .push(varint_to_i64(packed.read_varint()?)?);
                }
            }
            (8, WIRE_LEN) => tensor.name = read_utf8(parser.read_len()?)?,
            (9, WIRE_LEN) => tensor.raw_data = parser.read_len()?.to_vec(),
            _ => parser.skip_value(wire)?,
        }
    }

    Ok(tensor)
}

fn read_utf8(bytes: &[u8]) -> Result<String> {
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|err| OnnxLoadError::InvalidWire(format!("invalid UTF-8 string: {err}")))
}

fn varint_to_i32(value: u64) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| OnnxLoadError::InvalidWire(format!("varint {value} overflows i32")))
}

fn varint_to_i64(value: u64) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| OnnxLoadError::InvalidWire(format!("varint {value} overflows i64")))
}

fn varint_to_usize(value: u64) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| OnnxLoadError::InvalidWire(format!("varint {value} overflows usize")))
}

struct ProtoParser<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ProtoParser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn is_done(&self) -> bool {
        self.offset >= self.bytes.len()
    }

    fn read_key(&mut self) -> Result<Option<(u32, u8)>> {
        if self.is_done() {
            return Ok(None);
        }
        let key = self.read_varint()?;
        let field = u32::try_from(key >> 3)
            .map_err(|_| OnnxLoadError::InvalidWire(format!("field key {key} overflows u32")))?;
        let wire = (key & 0b111) as u8;
        if field == 0 {
            return Err(OnnxLoadError::InvalidWire(
                "field number 0 is invalid".to_string(),
            ));
        }
        Ok(Some((field, wire)))
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.read_byte()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(OnnxLoadError::InvalidWire(
            "unterminated varint".to_string(),
        ))
    }

    fn read_fixed32(&mut self) -> Result<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_fixed64(&mut self) -> Result<u64> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_len(&mut self) -> Result<&'a [u8]> {
        let len = varint_to_usize(self.read_varint()?)?;
        self.read_exact(len)
    }

    fn skip_value(&mut self, wire: u8) -> Result<()> {
        match wire {
            WIRE_VARINT => {
                self.read_varint()?;
            }
            WIRE_FIXED64 => {
                self.read_fixed64()?;
            }
            WIRE_LEN => {
                self.read_len()?;
            }
            WIRE_FIXED32 => {
                self.read_fixed32()?;
            }
            other => {
                return Err(OnnxLoadError::InvalidWire(format!(
                    "unsupported protobuf wire type {other}"
                )));
            }
        }
        Ok(())
    }

    fn read_byte(&mut self) -> Result<u8> {
        if self.offset >= self.bytes.len() {
            return Err(OnnxLoadError::InvalidWire(
                "unexpected end of protobuf".to_string(),
            ));
        }
        let byte = self.bytes[self.offset];
        self.offset += 1;
        Ok(byte)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| OnnxLoadError::InvalidWire("length overflow".to_string()))?;
        if end > self.bytes.len() {
            return Err(OnnxLoadError::InvalidWire(format!(
                "length {len} at offset {} exceeds protobuf size {}",
                self.offset,
                self.bytes.len()
            )));
        }
        let out = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_f32_initializer_from_raw_data() {
        let tensor = message([
            field_varints(1, &[2, 2]),
            field_varint(2, 1),
            field_len(8, b"weight".to_vec()),
            field_len(
                9,
                [1.0f32, 2.0, 3.0, 4.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            ),
        ]);
        let graph = message([field_len(5, tensor)]);
        let model = message([field_len(7, graph)]);

        let store = OnnxInitializerStore::from_model_bytes(&model).unwrap();

        assert_eq!(store.len(), 1);
        assert_eq!(
            store.required_f32("weight", &[2, 2]).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn loads_f32_initializer_from_float_data() {
        let tensor = message([
            field_varint(1, 3),
            field_varint(2, 1),
            field_len(8, b"bias".to_vec()),
            field_fixed32s(4, &[0.5f32.to_bits(), 1.5f32.to_bits(), 2.5f32.to_bits()]),
        ]);
        let graph = message([field_len(5, tensor)]);
        let model = message([field_len(7, graph)]);

        let store = OnnxInitializerStore::from_model_bytes(&model).unwrap();

        assert_eq!(
            store.required_f32("bias", &[3]).unwrap(),
            vec![0.5, 1.5, 2.5]
        );
    }

    #[test]
    fn validates_initializer_shape_and_dtype() {
        let tensor = message([
            field_varint(1, 1),
            field_varint(2, 7),
            field_len(8, b"ids".to_vec()),
            field_varint(7, 42),
        ]);
        let graph = message([field_len(5, tensor)]);
        let model = message([field_len(7, graph)]);
        let store = OnnxInitializerStore::from_model_bytes(&model).unwrap();

        let err = store.required_f32("ids", &[1]).unwrap_err();

        assert!(matches!(err, OnnxLoadError::WrongDtype { .. }));
    }

    fn message<const N: usize>(fields: [Vec<u8>; N]) -> Vec<u8> {
        fields.into_iter().flatten().collect()
    }

    fn field_varint(field: u32, value: u64) -> Vec<u8> {
        let mut out = key(field, WIRE_VARINT);
        encode_varint(value, &mut out);
        out
    }

    fn field_varints(field: u32, values: &[u64]) -> Vec<u8> {
        values
            .iter()
            .copied()
            .flat_map(|value| field_varint(field, value))
            .collect()
    }

    fn field_fixed32s(field: u32, values: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for value in values {
            out.extend(key(field, WIRE_FIXED32));
            out.extend(value.to_le_bytes());
        }
        out
    }

    fn field_len(field: u32, data: Vec<u8>) -> Vec<u8> {
        let mut out = key(field, WIRE_LEN);
        encode_varint(data.len() as u64, &mut out);
        out.extend(data);
        out
    }

    fn key(field: u32, wire: u8) -> Vec<u8> {
        let mut out = Vec::new();
        encode_varint((u64::from(field) << 3) | u64::from(wire), &mut out);
        out
    }

    fn encode_varint(mut value: u64, out: &mut Vec<u8>) {
        while value >= 0x80 {
            out.push((value as u8 & 0x7f) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }
}
