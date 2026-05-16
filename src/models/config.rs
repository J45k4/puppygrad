use serde::de::DeserializeOwned;
use std::error;
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Debug)]
pub enum ConfigLoadError {
    Read {
        path: String,
        source: std::io::Error,
    },
    Parse {
        path: String,
        source: serde_json::Error,
    },
}

impl fmt::Display for ConfigLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigLoadError::Read { path, source } => {
                write!(f, "failed to read config {path}: {source}")
            }
            ConfigLoadError::Parse { path, source } => {
                write!(f, "failed to parse config {path}: {source}")
            }
        }
    }
}

impl error::Error for ConfigLoadError {}

pub type Result<T> = std::result::Result<T, ConfigLoadError>;

pub fn load_json_config<T>(path: impl AsRef<Path>) -> Result<T>
where
    T: DeserializeOwned,
{
    let path = path.as_ref();
    let data = fs::read_to_string(path).map_err(|source| ConfigLoadError::Read {
        path: path.display().to_string(),
        source,
    })?;
    serde_json::from_str(&data).map_err(|source| ConfigLoadError::Parse {
        path: path.display().to_string(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TinyConfig {
        hidden_size: usize,
    }

    #[test]
    fn loads_typed_json_config() {
        let path =
            std::env::temp_dir().join(format!("puppygrad-config-test-{}.json", std::process::id()));
        fs::write(&path, r#"{"hidden_size":32}"#).unwrap();

        let config: TinyConfig = load_json_config(&path).unwrap();

        assert_eq!(config, TinyConfig { hidden_size: 32 });
        let _ = fs::remove_file(path);
    }
}
