//! IDX is a typed tensor: rank and big-endian dimensions followed by dense elements.
use super::{elements, DataType, LoadedTensor, Result};
pub fn read(bytes: &[u8]) -> Result<LoadedTensor> {
    if bytes.len() < 4 || bytes[..2] != [0, 0] || bytes[3] == 0 {
        return Err("invalid IDX header".into());
    }
    let dtype = match bytes[2] {
        0x08 => DataType::U8,
        0x09 => DataType::I8,
        0x0b => DataType::I16,
        0x0c => DataType::I32,
        0x0d => DataType::F32,
        0x0e => DataType::F64,
        _ => return Err("invalid IDX dtype".into()),
    };
    let header = 4 + bytes[3] as usize * 4;
    if bytes.len() < header {
        return Err("truncated IDX dimensions".into());
    }
    let shape = bytes[4..header]
        .chunks_exact(4)
        .map(|b| u32::from_be_bytes(b.try_into().unwrap()) as usize)
        .collect::<Vec<_>>();
    let size = elements(&shape)?
        .checked_mul(dtype.size())
        .ok_or("IDX byte size overflow")?;
    if bytes.len() - header != size {
        return Err("IDX payload size does not match its shape".into());
    }
    let mut data = bytes[header..].to_vec();
    if dtype.size() > 1 {
        for element in data.chunks_exact_mut(dtype.size()) {
            element.reverse();
        }
    }
    Ok(LoadedTensor {
        dtype,
        shape,
        bytes: data,
    })
}
