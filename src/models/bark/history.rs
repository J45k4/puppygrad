use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use flate2::read::DeflateDecoder;
use serde::Deserialize;

use super::{
    BarkAssetPaths, BarkError, BarkGenerationConfig, Result, BARK_SPEAKER_EMBEDDINGS_JSON,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarkHistoryPrompt {
    pub semantic_prompt: Vec<usize>,
    pub coarse_prompt: Vec<Vec<usize>>,
    pub fine_prompt: Vec<Vec<usize>>,
}

impl BarkHistoryPrompt {
    pub fn validate(&self, generation_config: &BarkGenerationConfig) -> Result<()> {
        if self.semantic_prompt.is_empty() {
            return Err(BarkError::InvalidInput(
                "semantic history prompt must not be empty".to_string(),
            ));
        }
        if self.coarse_prompt.len() != generation_config.coarse_acoustics_config.n_coarse_codebooks
        {
            return Err(BarkError::InvalidInput(format!(
                "coarse history prompt has {} codebooks, expected {}",
                self.coarse_prompt.len(),
                generation_config.coarse_acoustics_config.n_coarse_codebooks
            )));
        }
        if self.fine_prompt.len() != generation_config.fine_acoustics_config.n_fine_codebooks {
            return Err(BarkError::InvalidInput(format!(
                "fine history prompt has {} codebooks, expected {}",
                self.fine_prompt.len(),
                generation_config.fine_acoustics_config.n_fine_codebooks
            )));
        }
        validate_equal_non_empty_rows("coarse", &self.coarse_prompt)?;
        validate_equal_non_empty_rows("fine", &self.fine_prompt)?;
        if self
            .semantic_prompt
            .iter()
            .any(|token| *token >= generation_config.semantic_config.semantic_vocab_size)
        {
            return Err(BarkError::InvalidInput(
                "semantic history prompt contains token outside semantic vocabulary".to_string(),
            ));
        }
        validate_codebooks(
            "coarse",
            &self.coarse_prompt,
            generation_config.codebook_size,
        )?;
        validate_codebooks("fine", &self.fine_prompt, generation_config.codebook_size)?;
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct SpeakerEmbeddingsFile {
    #[serde(default)]
    repo_or_path: Option<String>,
    #[serde(flatten)]
    presets: HashMap<String, SpeakerEmbeddingPaths>,
}

#[derive(Debug, Deserialize)]
struct SpeakerEmbeddingPaths {
    semantic_prompt: String,
    coarse_prompt: String,
    fine_prompt: String,
}

#[derive(Debug)]
struct NpyArray {
    shape: Vec<usize>,
    values: Vec<usize>,
}

pub fn load_bark_history_prompt(
    paths: &BarkAssetPaths,
    voice_preset: &str,
    generation_config: &BarkGenerationConfig,
) -> Result<BarkHistoryPrompt> {
    let prompt = if voice_preset.ends_with(".npz") || Path::new(voice_preset).exists() {
        let path = resolve_prompt_path(&paths.model_dir, voice_preset);
        load_history_prompt_npz(&path)?
    } else if let Some(prompt) = load_history_prompt_from_mapping(paths, voice_preset)? {
        prompt
    } else {
        let path = paths.model_dir.join(format!("{voice_preset}.npz"));
        if path.exists() {
            load_history_prompt_npz(&path)?
        } else {
            return Err(BarkError::Asset(format!(
                "voice preset {voice_preset:?} was not found in {} or as {}",
                paths.model_dir.join(BARK_SPEAKER_EMBEDDINGS_JSON).display(),
                path.display()
            )));
        }
    };
    prompt.validate(generation_config)?;
    Ok(prompt)
}

fn load_history_prompt_from_mapping(
    paths: &BarkAssetPaths,
    voice_preset: &str,
) -> Result<Option<BarkHistoryPrompt>> {
    let mapping_path = paths.model_dir.join(BARK_SPEAKER_EMBEDDINGS_JSON);
    if !mapping_path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&mapping_path).map_err(|err| {
        BarkError::Asset(format!("failed to read {}: {err}", mapping_path.display()))
    })?;
    let mapping: SpeakerEmbeddingsFile = serde_json::from_str(&text).map_err(|err| {
        BarkError::Asset(format!("failed to parse {}: {err}", mapping_path.display()))
    })?;
    let Some(preset) = mapping.presets.get(voice_preset) else {
        return Ok(None);
    };
    let base = mapping
        .repo_or_path
        .as_deref()
        .map(|path| resolve_prompt_path(&paths.model_dir, path))
        .unwrap_or_else(|| paths.model_dir.clone());
    Ok(Some(BarkHistoryPrompt {
        semantic_prompt: load_npy_usize(&base.join(&preset.semantic_prompt))?.values,
        coarse_prompt: load_prompt_matrix(&base.join(&preset.coarse_prompt))?,
        fine_prompt: load_prompt_matrix(&base.join(&preset.fine_prompt))?,
    }))
}

fn resolve_prompt_path(model_dir: &Path, value: &str) -> PathBuf {
    let path = Path::new(value);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        model_dir.join(path)
    }
}

fn load_history_prompt_npz(path: &Path) -> Result<BarkHistoryPrompt> {
    let entries = read_npz_entries(path)?;
    let semantic = npz_required_npy(&entries, "semantic_prompt.npy")?;
    let coarse = npz_required_npy(&entries, "coarse_prompt.npy")?;
    let fine = npz_required_npy(&entries, "fine_prompt.npy")?;
    Ok(BarkHistoryPrompt {
        semantic_prompt: semantic.values,
        coarse_prompt: npy_to_matrix(coarse, "coarse_prompt")?,
        fine_prompt: npy_to_matrix(fine, "fine_prompt")?,
    })
}

fn load_prompt_matrix(path: &Path) -> Result<Vec<Vec<usize>>> {
    npy_to_matrix(load_npy_usize(path)?, &path.display().to_string())
}

fn validate_equal_non_empty_rows(name: &str, rows: &[Vec<usize>]) -> Result<()> {
    let Some(first) = rows.first() else {
        return Err(BarkError::InvalidInput(format!(
            "{name} history prompt must contain rows"
        )));
    };
    if first.is_empty() {
        return Err(BarkError::InvalidInput(format!(
            "{name} history prompt must not be empty"
        )));
    }
    if rows.iter().any(|row| row.len() != first.len()) {
        return Err(BarkError::InvalidInput(format!(
            "{name} history prompt rows must have equal lengths"
        )));
    }
    Ok(())
}

fn validate_codebooks(name: &str, rows: &[Vec<usize>], codebook_size: usize) -> Result<()> {
    if rows.iter().flatten().any(|code| *code >= codebook_size) {
        return Err(BarkError::InvalidInput(format!(
            "{name} history prompt contains code outside codebook size {codebook_size}"
        )));
    }
    Ok(())
}

fn read_npz_entries(path: &Path) -> Result<HashMap<String, Vec<u8>>> {
    let bytes = fs::read(path)
        .map_err(|err| BarkError::Asset(format!("failed to read {}: {err}", path.display())))?;
    let mut offset = 0usize;
    let mut entries = HashMap::new();
    while offset + 30 <= bytes.len() {
        if &bytes[offset..offset + 4] != b"PK\x03\x04" {
            break;
        }
        let method = read_u16_le(&bytes, offset + 8)?;
        let compressed_size = read_u32_le(&bytes, offset + 18)? as usize;
        let uncompressed_size = read_u32_le(&bytes, offset + 22)? as usize;
        let name_len = read_u16_le(&bytes, offset + 26)? as usize;
        let extra_len = read_u16_le(&bytes, offset + 28)? as usize;
        let name_start = offset + 30;
        let data_start = name_start + name_len + extra_len;
        let data_end = data_start + compressed_size;
        if data_end > bytes.len() {
            return Err(BarkError::Asset(format!(
                "npz entry in {} exceeds file length",
                path.display()
            )));
        }
        let name = std::str::from_utf8(&bytes[name_start..name_start + name_len])
            .map_err(|err| BarkError::Asset(format!("invalid npz entry name: {err}")))?
            .to_string();
        let data = match method {
            0 => bytes[data_start..data_end].to_vec(),
            8 => {
                let mut decoder = DeflateDecoder::new(&bytes[data_start..data_end]);
                let mut decoded = Vec::with_capacity(uncompressed_size);
                decoder.read_to_end(&mut decoded).map_err(|err| {
                    BarkError::Asset(format!("failed to inflate npz entry {name}: {err}"))
                })?;
                decoded
            }
            other => {
                return Err(BarkError::Asset(format!(
                    "unsupported npz compression method {other} for entry {name}"
                )));
            }
        };
        entries.insert(name, data);
        offset = data_end;
    }
    Ok(entries)
}

fn npz_required_npy(entries: &HashMap<String, Vec<u8>>, name: &str) -> Result<NpyArray> {
    let bytes = entries
        .get(name)
        .ok_or_else(|| BarkError::Asset(format!("npz history prompt missing {name}")))?;
    parse_npy_usize(bytes, name)
}

fn load_npy_usize(path: &Path) -> Result<NpyArray> {
    let bytes = fs::read(path)
        .map_err(|err| BarkError::Asset(format!("failed to read {}: {err}", path.display())))?;
    parse_npy_usize(&bytes, &path.display().to_string())
}

fn parse_npy_usize(bytes: &[u8], label: &str) -> Result<NpyArray> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(BarkError::Asset(format!("{label} is not a .npy file")));
    }
    let major = bytes[6];
    let header_len_offset = 8usize;
    let (header_len, data_offset) = match major {
        1 => (
            read_u16_le(bytes, header_len_offset)? as usize,
            header_len_offset + 2,
        ),
        2 | 3 => (
            read_u32_le(bytes, header_len_offset)? as usize,
            header_len_offset + 4,
        ),
        other => {
            return Err(BarkError::Asset(format!(
                "{label} has unsupported npy version {other}"
            )));
        }
    };
    let header_end = data_offset + header_len;
    if header_end > bytes.len() {
        return Err(BarkError::Asset(format!(
            "{label} npy header exceeds file length"
        )));
    }
    let header = std::str::from_utf8(&bytes[data_offset..header_end])
        .map_err(|err| BarkError::Asset(format!("{label} has invalid npy header: {err}")))?;
    let dtype = parse_header_string(header, "'descr':")
        .or_else(|| parse_header_string(header, "\"descr\":"))
        .ok_or_else(|| BarkError::Asset(format!("{label} npy header missing descr")))?;
    let fortran_order = header.contains("'fortran_order': True")
        || header.contains("\"fortran_order\": true")
        || header.contains("\"fortran_order\": True");
    if fortran_order {
        return Err(BarkError::Asset(format!(
            "{label} npy arrays must be C-contiguous"
        )));
    }
    let shape = parse_shape(header)
        .ok_or_else(|| BarkError::Asset(format!("{label} npy header missing shape")))?;
    let elem_count = shape.iter().product::<usize>();
    let data = &bytes[header_end..];
    let values = match dtype.as_str() {
        "<i8" | "|i8" => read_i64_values(data, elem_count, label)?,
        "<i4" | "|i4" => read_i32_values(data, elem_count, label)?,
        "<u4" | "|u4" => read_u32_values(data, elem_count, label)?,
        "<u8" | "|u8" => read_u64_values(data, elem_count, label)?,
        other => {
            return Err(BarkError::Asset(format!(
                "{label} npy dtype {other:?} is unsupported for Bark history prompts"
            )));
        }
    };
    Ok(NpyArray { shape, values })
}

fn npy_to_matrix(array: NpyArray, label: &str) -> Result<Vec<Vec<usize>>> {
    if array.shape.len() != 2 {
        return Err(BarkError::Asset(format!(
            "{label} must be a 2D npy array, got shape {:?}",
            array.shape
        )));
    }
    let rows = array.shape[0];
    let cols = array.shape[1];
    Ok((0..rows)
        .map(|row| array.values[row * cols..(row + 1) * cols].to_vec())
        .collect())
}

fn parse_header_string(header: &str, key: &str) -> Option<String> {
    let start = header.find(key)? + key.len();
    let rest = header[start..].trim_start();
    let quote = rest.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let rest = &rest[quote.len_utf8()..];
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

fn parse_shape(header: &str) -> Option<Vec<usize>> {
    let start = header
        .find("'shape':")
        .or_else(|| header.find("\"shape\":"))?;
    let tuple_start = header[start..].find('(')? + start + 1;
    let tuple_end = header[tuple_start..].find(')')? + tuple_start;
    let dims = header[tuple_start..tuple_end]
        .split(',')
        .filter_map(|part| {
            let part = part.trim();
            if part.is_empty() {
                None
            } else {
                part.parse::<usize>().ok()
            }
        })
        .collect::<Vec<_>>();
    Some(dims)
}

fn read_i64_values(data: &[u8], count: usize, label: &str) -> Result<Vec<usize>> {
    read_chunks(data, count, 8, label)?
        .map(|bytes| {
            let value = i64::from_le_bytes(bytes.try_into().unwrap());
            if value < 0 {
                Err(BarkError::Asset(format!(
                    "{label} contains negative history prompt value {value}"
                )))
            } else {
                Ok(value as usize)
            }
        })
        .collect()
}

fn read_i32_values(data: &[u8], count: usize, label: &str) -> Result<Vec<usize>> {
    read_chunks(data, count, 4, label)?
        .map(|bytes| {
            let value = i32::from_le_bytes(bytes.try_into().unwrap());
            if value < 0 {
                Err(BarkError::Asset(format!(
                    "{label} contains negative history prompt value {value}"
                )))
            } else {
                Ok(value as usize)
            }
        })
        .collect()
}

fn read_u32_values(data: &[u8], count: usize, label: &str) -> Result<Vec<usize>> {
    read_chunks(data, count, 4, label)?
        .map(|bytes| Ok(u32::from_le_bytes(bytes.try_into().unwrap()) as usize))
        .collect()
}

fn read_u64_values(data: &[u8], count: usize, label: &str) -> Result<Vec<usize>> {
    read_chunks(data, count, 8, label)?
        .map(|bytes| Ok(u64::from_le_bytes(bytes.try_into().unwrap()) as usize))
        .collect()
}

fn read_chunks<'a>(
    data: &'a [u8],
    count: usize,
    width: usize,
    label: &str,
) -> Result<std::slice::ChunksExact<'a, u8>> {
    let expected = count * width;
    if data.len() < expected {
        return Err(BarkError::Asset(format!(
            "{label} npy data length {} is shorter than expected {expected}",
            data.len()
        )));
    }
    Ok(data[..expected].chunks_exact(width))
}

fn read_u16_le(bytes: &[u8], offset: usize) -> Result<u16> {
    if offset + 2 > bytes.len() {
        return Err(BarkError::Asset(
            "unexpected end while reading u16".to_string(),
        ));
    }
    Ok(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
    if offset + 4 > bytes.len() {
        return Err(BarkError::Asset(
            "unexpected end while reading u32".to_string(),
        ));
    }
    Ok(u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_int64_npy_vector() -> Result<()> {
        let bytes = npy_i64(&[3], &[1, 2, 3]);

        let array = parse_npy_usize(&bytes, "semantic_prompt.npy")?;

        assert_eq!(array.shape, vec![3]);
        assert_eq!(array.values, vec![1, 2, 3]);
        Ok(())
    }

    #[test]
    fn validates_history_prompt_shapes_and_ranges() -> Result<()> {
        let prompt = BarkHistoryPrompt {
            semantic_prompt: vec![1, 2],
            coarse_prompt: vec![vec![1, 2], vec![3, 4]],
            fine_prompt: vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8]],
        };

        prompt.validate(&tiny_generation_config())?;
        Ok(())
    }

    #[test]
    fn loads_history_prompt_from_stored_npz() -> Result<()> {
        let dir =
            std::env::temp_dir().join(format!("puppygrad-bark-history-{}", std::process::id()));
        fs::create_dir_all(&dir).map_err(|err| BarkError::Asset(err.to_string()))?;
        let path = dir.join("speaker.npz");
        let entries = [
            ("semantic_prompt.npy", npy_i64(&[2], &[1, 2])),
            ("coarse_prompt.npy", npy_i64(&[2, 2], &[1, 2, 3, 4])),
            (
                "fine_prompt.npy",
                npy_i64(&[4, 2], &[1, 2, 3, 4, 5, 6, 7, 8]),
            ),
        ];
        fs::write(&path, stored_npz(&entries)).map_err(|err| BarkError::Asset(err.to_string()))?;

        let prompt = load_history_prompt_npz(&path)?;

        fs::remove_dir_all(&dir).ok();
        assert_eq!(prompt.semantic_prompt, vec![1, 2]);
        assert_eq!(prompt.coarse_prompt, vec![vec![1, 2], vec![3, 4]]);
        assert_eq!(prompt.fine_prompt[3], vec![7, 8]);
        Ok(())
    }

    #[test]
    fn loads_history_prompt_from_speaker_embedding_mapping() -> Result<()> {
        let dir =
            std::env::temp_dir().join(format!("puppygrad-bark-history-map-{}", std::process::id()));
        fs::create_dir_all(dir.join("speaker_embeddings/v2"))
            .map_err(|err| BarkError::Asset(err.to_string()))?;
        fs::write(
            dir.join(BARK_SPEAKER_EMBEDDINGS_JSON),
            r#"{
              "repo_or_path": ".",
              "v2/en_speaker_0": {
                "semantic_prompt": "speaker_embeddings/v2/en_speaker_0_semantic_prompt.npy",
                "coarse_prompt": "speaker_embeddings/v2/en_speaker_0_coarse_prompt.npy",
                "fine_prompt": "speaker_embeddings/v2/en_speaker_0_fine_prompt.npy"
              }
            }"#,
        )
        .map_err(|err| BarkError::Asset(err.to_string()))?;
        fs::write(
            dir.join("speaker_embeddings/v2/en_speaker_0_semantic_prompt.npy"),
            npy_i64(&[2], &[1, 2]),
        )
        .map_err(|err| BarkError::Asset(err.to_string()))?;
        fs::write(
            dir.join("speaker_embeddings/v2/en_speaker_0_coarse_prompt.npy"),
            npy_i64(&[2, 2], &[1, 2, 3, 4]),
        )
        .map_err(|err| BarkError::Asset(err.to_string()))?;
        fs::write(
            dir.join("speaker_embeddings/v2/en_speaker_0_fine_prompt.npy"),
            npy_i64(&[4, 2], &[1, 2, 3, 4, 5, 6, 7, 8]),
        )
        .map_err(|err| BarkError::Asset(err.to_string()))?;

        let paths = BarkAssetPaths::new(&dir);
        let prompt =
            load_bark_history_prompt(&paths, "v2/en_speaker_0", &tiny_generation_config())?;

        fs::remove_dir_all(&dir).ok();
        assert_eq!(prompt.semantic_prompt, vec![1, 2]);
        assert_eq!(prompt.coarse_prompt[1], vec![3, 4]);
        Ok(())
    }

    fn tiny_generation_config() -> BarkGenerationConfig {
        BarkGenerationConfig {
            sample_rate: 24_000,
            codebook_size: 10,
            semantic_config: super::super::BarkSemanticGenerationConfig {
                eos_token_id: 10,
                max_input_semantic_length: 4,
                max_new_tokens: 8,
                semantic_infer_token: 100,
                semantic_pad_token: 10,
                semantic_rate_hz: 49.9,
                semantic_vocab_size: 10,
                text_encoding_offset: 1000,
                text_pad_token: 1001,
                temperature: 0.7,
                top_k: 50,
                top_p: 1.0,
            },
            coarse_acoustics_config: super::super::BarkCoarseGenerationConfig {
                coarse_infer_token: 77,
                coarse_rate_hz: 75,
                coarse_semantic_pad_token: 99,
                max_coarse_history: 4,
                max_coarse_input_length: 4,
                n_coarse_codebooks: 2,
                sliding_window_len: 2,
                temperature: 0.7,
                top_k: 50,
                top_p: 1.0,
            },
            fine_acoustics_config: super::super::BarkFineGenerationConfig {
                max_fine_history_length: 4,
                max_fine_input_length: 4,
                n_fine_codebooks: 4,
                temperature: 0.5,
                top_k: 50,
                top_p: 1.0,
            },
            model_type: Some("bark".to_string()),
        }
    }

    fn npy_i64(shape: &[usize], values: &[i64]) -> Vec<u8> {
        let shape_text = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            format!(
                "({})",
                shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let mut header =
            format!("{{'descr': '<i8', 'fortran_order': False, 'shape': {shape_text}, }}");
        let base_len = 10 + header.len() + 1;
        let padding = (16 - (base_len % 16)) % 16;
        header.push_str(&" ".repeat(padding));
        header.push('\n');
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.extend_from_slice(&[1, 0]);
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn stored_npz(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for (name, data) in entries {
            bytes.extend_from_slice(b"PK\x03\x04");
            bytes.extend_from_slice(&20u16.to_le_bytes());
            bytes.extend_from_slice(&0u16.to_le_bytes());
            bytes.extend_from_slice(&0u16.to_le_bytes());
            bytes.extend_from_slice(&0u16.to_le_bytes());
            bytes.extend_from_slice(&0u16.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
            bytes.extend_from_slice(&0u16.to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(data);
        }
        bytes
    }
}
