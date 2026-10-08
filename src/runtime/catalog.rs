//! Download metadata and embedded programs for the interactive model browser.
use crate::models::assets::{download_huggingface_file_with_progress, DownloadProgress};
use serde::Deserialize;
use std::{
    fs, io,
    path::{Component, Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub id: String,
    pub name: String,
    pub description: String,
    pub program: String,
    pub checkpoint: Checkpoint,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub repo: String,
    pub revision: String,
    pub files: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub manifest: Manifest,
    program_text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Missing,
    Partial { present: usize, total: usize },
    Downloaded,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing => write!(f, "not downloaded"),
            Self::Partial { present, total } => write!(f, "partial ({present}/{total} files)"),
            Self::Downloaded => write!(f, "downloaded"),
        }
    }
}

fn relative_file(value: &str) -> bool {
    !value.is_empty()
        && Path::new(value)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Self> {
        let value: Self = serde_json::from_str(text)?;
        if value.id.is_empty()
            || !value
                .id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
            || matches!(value.id.as_str(), "." | "..")
        {
            return Err(
                "model id must contain only letters, numbers, dots, hyphens or underscores".into(),
            );
        }
        if !relative_file(&value.program) || !value.program.ends_with(".pup") {
            return Err("program must be a relative .pup filename".into());
        }
        if value.checkpoint.repo.is_empty()
            || !value.checkpoint.repo.split('/').all(|s| {
                !s.is_empty()
                    && s != "."
                    && s != ".."
                    && s.chars()
                        .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
            })
        {
            return Err("invalid Hugging Face repository id".into());
        }
        if value.checkpoint.revision.is_empty()
            || matches!(value.checkpoint.revision.as_str(), "." | "..")
            || !value
                .checkpoint
                .revision
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
        {
            return Err("invalid checkpoint revision".into());
        }
        if value.checkpoint.files.is_empty()
            || value.checkpoint.files.iter().any(|s| !relative_file(s))
        {
            return Err(
                "checkpoint files must be nonempty relative paths without parent traversal".into(),
            );
        }
        for required in ["config.json", "tokenizer.json"] {
            if !value.checkpoint.files.iter().any(|s| s == required) {
                return Err(format!("LLM manifest must include {required}").into());
            }
        }
        if !value
            .checkpoint
            .files
            .iter()
            .any(|s| s == "model.safetensors")
            && (!value
                .checkpoint
                .files
                .iter()
                .any(|s| s == "model.safetensors.index.json")
                || !value
                    .checkpoint
                    .files
                    .iter()
                    .any(|s| s.ends_with(".safetensors")))
        {
            return Err("LLM manifest must include model.safetensors or a shard index and its .safetensors files".into());
        }
        Ok(value)
    }
}

impl Entry {
    /// Reuse local checkout assets; otherwise keep checkpoints separated by revision.
    pub fn model_dir(&self, cache: &Path) -> PathBuf {
        let local = Path::new("models").join(&self.manifest.id);
        if local.is_dir() {
            return local;
        }
        cache
            .join("models")
            .join(&self.manifest.id)
            .join(&self.manifest.checkpoint.revision)
    }

    pub fn status(&self, cache: &Path) -> Status {
        self.status_in(&self.model_dir(cache))
    }

    pub fn status_in(&self, dir: &Path) -> Status {
        let total = self.manifest.checkpoint.files.len();
        let present = self
            .manifest
            .checkpoint
            .files
            .iter()
            .filter(|name| fs::metadata(dir.join(name)).is_ok_and(|m| m.is_file() && m.len() > 0))
            .count();
        match present {
            0 => Status::Missing,
            n if n == total => Status::Downloaded,
            n => Status::Partial { present: n, total },
        }
    }

    pub fn download(
        &self,
        cache: &Path,
        on_progress: &mut dyn FnMut(DownloadProgress<'_>) -> io::Result<()>,
    ) -> Result<()> {
        let dir = self.model_dir(cache);
        for file in &self.manifest.checkpoint.files {
            download_huggingface_file_with_progress(
                &self.manifest.checkpoint.repo,
                &self.manifest.checkpoint.revision,
                file,
                &dir.join(file),
                on_progress,
            )?;
        }
        Ok(())
    }

    pub fn materialize_program(&self, cache: &Path) -> Result<PathBuf> {
        let dir = cache.join("programs").join(&self.manifest.id);
        fs::create_dir_all(&dir)?;
        let path = dir.join(&self.manifest.program);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, &self.program_text)?;
        Ok(path)
    }
}

pub fn default_cache_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME").filter(|p| !p.is_empty()) {
        return PathBuf::from(path).join("puppygrad");
    }
    if let Some(path) = std::env::var_os("HOME").filter(|p| !p.is_empty()) {
        return PathBuf::from(path).join(".cache/puppygrad");
    }
    PathBuf::from(".cache/puppygrad")
}

/// Custom catalogs contain *.model.json manifests and their relative .pup programs.
pub fn load(directory: Option<&Path>) -> Result<Vec<Entry>> {
    let mut entries = if let Some(directory) = directory {
        let mut paths = fs::read_dir(directory)?
            .map(|e| e.map(|e| e.path()))
            .collect::<io::Result<Vec<_>>>()?;
        paths.sort();
        let mut entries = vec![];
        for path in paths.into_iter().filter(|p| {
            p.file_name()
                .is_some_and(|s| s.to_string_lossy().ends_with(".model.json"))
        }) {
            let manifest = Manifest::parse(&fs::read_to_string(&path)?)?;
            let program_text = fs::read_to_string(directory.join(&manifest.program))?;
            entries.push(Entry {
                manifest,
                program_text,
            });
        }
        entries
    } else {
        [
            (
                include_str!("../../examples/llm.model.json"),
                include_str!("../../examples/llm.pup"),
            ),
            (
                include_str!("../../examples/qwen3-0.6b.model.json"),
                include_str!("../../examples/qwen3_cached.pup"),
            ),
            (
                include_str!("../../examples/qwen3-1.7b.model.json"),
                include_str!("../../examples/qwen3_cached.pup"),
            ),
        ]
        .into_iter()
        .map(|(text, program)| {
            Ok(Entry {
                manifest: Manifest::parse(text)?,
                program_text: program.to_owned(),
            })
        })
        .collect::<Result<Vec<_>>>()?
    };
    entries.sort_by(|a, b| a.manifest.id.cmp(&b.manifest.id));
    if entries.is_empty() {
        return Err("model catalog is empty".into());
    }
    if entries
        .windows(2)
        .any(|pair| pair[0].manifest.id == pair[1].manifest.id)
    {
        return Err("model ids in the catalog must be unique".into());
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_distinguishes_missing_partial_and_downloaded_assets() {
        let cache = std::env::temp_dir().join(format!("pup-catalog-{}", std::process::id()));
        for entry in load(None).unwrap() {
            let dir = cache.join(&entry.manifest.id);
            fs::create_dir_all(&dir).unwrap();
            assert_eq!(entry.status_in(&dir), Status::Missing);
            fs::write(dir.join("config.json"), "{}").unwrap();
            fs::write(dir.join("tokenizer.json"), []).unwrap();
            assert_eq!(
                entry.status_in(&dir),
                Status::Partial {
                    present: 1,
                    total: entry.manifest.checkpoint.files.len()
                }
            );
            for file in &entry.manifest.checkpoint.files {
                fs::write(dir.join(file), "data").unwrap();
            }
            assert_eq!(entry.status_in(&dir), Status::Downloaded);
        }
        fs::remove_dir_all(cache).unwrap();
    }

    #[test]
    fn sharded_manifest_requires_the_index_and_weight_files() {
        let mut manifest: serde_json::Value =
            serde_json::from_str(include_str!("../../examples/qwen3-1.7b.model.json")).unwrap();
        assert!(Manifest::parse(&manifest.to_string()).is_ok());
        manifest["checkpoint"]["files"]
            .as_array_mut()
            .unwrap()
            .retain(|file| !file.as_str().unwrap().ends_with(".safetensors"));
        assert!(Manifest::parse(&manifest.to_string()).is_err());
        manifest["checkpoint"]["files"]
            .as_array_mut()
            .unwrap()
            .push("model-00001-of-00002.safetensors".into());
        assert!(Manifest::parse(&manifest.to_string()).is_ok());
        manifest["checkpoint"]["files"]
            .as_array_mut()
            .unwrap()
            .retain(|file| file.as_str().unwrap() != "model.safetensors.index.json");
        assert!(Manifest::parse(&manifest.to_string()).is_err());
    }

    #[test]
    fn manifest_rejects_paths_that_escape_model_storage() {
        let mut manifest: serde_json::Value =
            serde_json::from_str(include_str!("../../examples/llm.model.json")).unwrap();
        manifest["checkpoint"]["files"][0] = "../config.json".into();
        assert!(Manifest::parse(&manifest.to_string()).is_err());
        manifest["checkpoint"]["files"][0] = "config.json".into();
        manifest["id"] = "../gpt2".into();
        assert!(Manifest::parse(&manifest.to_string()).is_err());
        manifest["id"] = "gpt2".into();
        manifest["checkpoint"]["revision"] = "..".into();
        assert!(Manifest::parse(&manifest.to_string()).is_err());
    }
}
