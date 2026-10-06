//! Strict numeric CSV with explicit field groups, dtypes and per-row shapes.
use super::{elements, DataType, LoadedTensor, Result, Tensors};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Schema {
    pub columns: Vec<Column>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Column {
    pub name: String,
    pub fields: Vec<String>,
    pub dtype: DataType,
    #[serde(default)]
    pub shape: Vec<usize>,
}
fn records(text: &str) -> Result<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut cell = String::new();
    let mut chars = text.chars().peekable();
    let mut quoted = false;
    let mut closed = false;
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    cell.push('"');
                } else {
                    quoted = false;
                    closed = true;
                }
            } else {
                cell.push(c);
            }
            continue;
        }
        match c {
            '"' if cell.is_empty() && !closed => quoted = true,
            ',' => {
                row.push(std::mem::take(&mut cell));
                closed = false;
            }
            '\n' | '\r' => {
                if c == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                row.push(std::mem::take(&mut cell));
                rows.push(std::mem::take(&mut row));
                closed = false;
            }
            '"' => return Err("quote inside unquoted CSV field".into()),
            _ if closed => return Err("characters after closing CSV quote".into()),
            _ => cell.push(c),
        }
    }
    if quoted {
        return Err("unterminated CSV quote".into());
    }
    if closed || !cell.is_empty() || !row.is_empty() {
        row.push(cell);
        rows.push(row);
    }
    Ok(rows)
}
pub fn read(bytes: &[u8], schema: &Schema) -> Result<Tensors> {
    let rows = records(std::str::from_utf8(bytes)?.trim_start_matches('\u{feff}'))?;
    let header = rows.first().ok_or("CSV is empty")?;
    if header.iter().collect::<HashSet<_>>().len() != header.len() {
        return Err("duplicate CSV header fields".into());
    }
    if schema.columns.is_empty() {
        return Err("CSV schema has no columns".into());
    }
    let mut result = Tensors::new();
    for column in &schema.columns {
        if column.name.is_empty() || result.contains_key(&column.name) {
            return Err("duplicate or empty CSV tensor name".into());
        }
        if elements(&column.shape)? != column.fields.len() {
            return Err("CSV field count does not match tensor row shape".into());
        }
        let indices = column
            .fields
            .iter()
            .map(|f| {
                header
                    .iter()
                    .position(|h| h == f)
                    .ok_or_else(|| format!("CSV field {f:?} not found"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut data = Vec::new();
        for (i, row) in rows.iter().enumerate().skip(1) {
            if row.len() != header.len() {
                return Err(format!(
                    "CSV row {} has {} fields, expected {}",
                    i + 1,
                    row.len(),
                    header.len()
                )
                .into());
            }
            for &index in &indices {
                let field = row[index].trim();
                let parsed: Result<Vec<u8>> = (|| {
                    Ok(match column.dtype {
                        DataType::U8 => vec![field.parse::<u8>()?],
                        DataType::I8 => vec![field.parse::<i8>()? as u8],
                        DataType::I16 => field.parse::<i16>()?.to_le_bytes().to_vec(),
                        DataType::I32 => field.parse::<i32>()?.to_le_bytes().to_vec(),
                        DataType::F32 => {
                            let v = field.parse::<f32>()?;
                            if !v.is_finite() {
                                return Err("non-finite value".into());
                            }
                            v.to_le_bytes().to_vec()
                        }
                        DataType::F64 => {
                            let v = field.parse::<f64>()?;
                            if !v.is_finite() {
                                return Err("non-finite value".into());
                            }
                            v.to_le_bytes().to_vec()
                        }
                        DataType::Bool => match field {
                            "true" | "1" => vec![1],
                            "false" | "0" => vec![0],
                            _ => return Err("expected boolean".into()),
                        },
                    })
                })();
                data.extend(
                    parsed.map_err(|e| {
                        format!("CSV row {}, field {:?}: {e}", i + 1, header[index])
                    })?,
                );
            }
        }
        let mut shape = vec![rows.len() - 1];
        shape.extend(&column.shape);
        result.insert(
            column.name.clone(),
            LoadedTensor {
                dtype: column.dtype,
                shape,
                bytes: data,
            },
        );
    }
    Ok(result)
}
