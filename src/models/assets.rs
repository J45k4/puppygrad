use std::error;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum AssetError {
    CreateParentDir {
        path: String,
        source: io::Error,
    },
    BuildHttpClient(reqwest::Error),
    Download {
        url: String,
        source: reqwest::Error,
    },
    CreateTempFile {
        path: String,
        source: io::Error,
    },
    WriteDownload {
        path: String,
        source: io::Error,
    },
    RenameDownload {
        from: String,
        to: String,
        source: io::Error,
    },
    MissingRequiredFiles {
        model_dir: String,
        files: Vec<String>,
    },
}

impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssetError::CreateParentDir { path, source } => {
                write!(f, "failed to create parent directory for {path}: {source}")
            }
            AssetError::BuildHttpClient(err) => write!(f, "failed to build HTTP client: {err}"),
            AssetError::Download { url, source } => write!(f, "failed to download {url}: {source}"),
            AssetError::CreateTempFile { path, source } => {
                write!(f, "failed to create temporary file {path}: {source}")
            }
            AssetError::WriteDownload { path, source } => {
                write!(f, "failed to write download {path}: {source}")
            }
            AssetError::RenameDownload { from, to, source } => {
                write!(f, "failed to move {from} to {to}: {source}")
            }
            AssetError::MissingRequiredFiles { model_dir, files } => {
                write!(
                    f,
                    "model directory {model_dir} is missing required files: {}",
                    files.join(", ")
                )
            }
        }
    }
}

impl error::Error for AssetError {}

pub type Result<T> = std::result::Result<T, AssetError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HuggingFaceAsset {
    pub filename: &'static str,
}

impl HuggingFaceAsset {
    pub const fn required(filename: &'static str) -> Self {
        Self { filename }
    }
}

pub fn default_model_dir(short_name: &str) -> PathBuf {
    PathBuf::from("models").join(short_name)
}

pub fn huggingface_cache_model_dir(
    cache_root: impl AsRef<Path>,
    model_id: &str,
    revision: &str,
) -> PathBuf {
    cache_root
        .as_ref()
        .join("huggingface")
        .join(sanitize_cache_component(model_id))
        .join(sanitize_cache_component(revision))
}

pub fn resolve_model_dir(explicit: Option<PathBuf>, default: impl FnOnce() -> PathBuf) -> PathBuf {
    explicit.unwrap_or_else(default)
}

pub fn check_required_files(model_dir: &Path, filenames: &[&str]) -> Result<()> {
    let missing = missing_required_files(model_dir, filenames);
    if missing.is_empty() {
        return Ok(());
    }
    Err(AssetError::MissingRequiredFiles {
        model_dir: model_dir.display().to_string(),
        files: missing,
    })
}

pub fn missing_required_files(model_dir: &Path, filenames: &[&str]) -> Vec<String> {
    filenames
        .iter()
        .copied()
        .filter(|filename| !model_dir.join(filename).is_file())
        .map(str::to_string)
        .collect()
}

pub fn download_huggingface_files(
    model_id: &str,
    revision: &str,
    model_dir: &Path,
    assets: &[HuggingFaceAsset],
) -> Result<()> {
    for asset in assets {
        download_huggingface_file(
            model_id,
            revision,
            asset.filename,
            &model_dir.join(asset.filename),
        )?;
    }
    Ok(())
}

pub fn prepare_huggingface_model_dir(
    model_id: &str,
    revision: &str,
    model_dir: &Path,
    assets: &[HuggingFaceAsset],
    download: bool,
) -> Result<()> {
    if download {
        download_huggingface_files(model_id, revision, model_dir, assets)?;
    }
    let filenames = assets
        .iter()
        .map(|asset| asset.filename)
        .collect::<Vec<_>>();
    check_required_files(model_dir, &filenames)
}

#[derive(Clone, Copy, Debug)]
pub struct DownloadProgress<'a> {
    pub filename: &'a str,
    pub bytes: u64,
    pub total: Option<u64>,
}

pub fn download_huggingface_file(
    model_id: &str,
    revision: &str,
    filename: &str,
    dst: &Path,
) -> Result<()> {
    download_huggingface_file_with_progress(model_id, revision, filename, dst, &mut |progress| {
        match progress.total {
            Some(total) => eprintln!(
                "download: {} {} / {} bytes",
                progress.filename, progress.bytes, total
            ),
            None => eprintln!("download: {} {} bytes", progress.filename, progress.bytes),
        }
        Ok(())
    })
}

/// Callbacks can return an error to cancel. Only complete files become visible at dst.
pub fn download_huggingface_file_with_progress(
    model_id: &str,
    revision: &str,
    filename: &str,
    dst: &Path,
    on_progress: &mut dyn FnMut(DownloadProgress<'_>) -> io::Result<()>,
) -> Result<()> {
    if let Ok(metadata) = fs::metadata(dst) {
        if metadata.is_file() && metadata.len() > 0 {
            on_progress(DownloadProgress {
                filename,
                bytes: metadata.len(),
                total: Some(metadata.len()),
            })
            .map_err(|source| AssetError::WriteDownload {
                path: dst.display().to_string(),
                source,
            })?;
            return Ok(());
        }
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).map_err(|source| AssetError::CreateParentDir {
            path: dst.display().to_string(),
            source,
        })?;
    }
    let url = format!("https://huggingface.co/{model_id}/resolve/{revision}/{filename}");
    let client = reqwest::blocking::Client::builder()
        .user_agent("puppygrad/0.1")
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .map_err(AssetError::BuildHttpClient)?;
    let mut response = client
        .get(&url)
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|source| AssetError::Download { url, source })?;
    let total = response.content_length();
    let tmp = dst.with_extension("download");
    let mut file = File::create(&tmp).map_err(|source| AssetError::CreateTempFile {
        path: tmp.display().to_string(),
        source,
    })?;
    let copied = copy_with_progress(&mut response, &mut file, filename, total, on_progress)
        .and_then(|bytes| {
            file.flush()?;
            Ok(bytes)
        });
    drop(file);
    if let Err(source) = copied {
        let _ = fs::remove_file(&tmp);
        return Err(AssetError::WriteDownload {
            path: tmp.display().to_string(),
            source,
        });
    }
    fs::rename(&tmp, dst).map_err(|source| AssetError::RenameDownload {
        from: tmp.display().to_string(),
        to: dst.display().to_string(),
        source,
    })?;
    Ok(())
}

fn copy_with_progress(
    reader: &mut impl Read,
    writer: &mut impl Write,
    filename: &str,
    expected_len: Option<u64>,
    on_progress: &mut dyn FnMut(DownloadProgress<'_>) -> io::Result<()>,
) -> io::Result<u64> {
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut written = 0_u64;
    let mut reported = std::time::Instant::now();
    on_progress(DownloadProgress {
        filename,
        bytes: 0,
        total: expected_len,
    })?;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        writer.write_all(&buffer[..read])?;
        written += read as u64;
        if reported.elapsed() >= std::time::Duration::from_millis(100) {
            on_progress(DownloadProgress {
                filename,
                bytes: written,
                total: expected_len,
            })?;
            reported = std::time::Instant::now();
        }
    }
    if expected_len.is_some_and(|length| length != written) || written == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "download length does not match response",
        ));
    }
    on_progress(DownloadProgress {
        filename,
        bytes: written,
        total: expected_len,
    })?;
    Ok(written)
}

fn sanitize_cache_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '.' | '-' | '_' => ch,
            _ => '-',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_copy_rejects_truncation_and_can_be_cancelled() {
        let mut output = Vec::new();
        let result = copy_with_progress(
            &mut &b"abc"[..],
            &mut output,
            "weights",
            Some(4),
            &mut |_| Ok(()),
        );
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        let mut output = Vec::new();
        let result = copy_with_progress(
            &mut &b"abc"[..],
            &mut output,
            "weights",
            Some(3),
            &mut |_| Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled")),
        );
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::Interrupted);
        assert!(output.is_empty());
        let mut progress = vec![];
        let result = copy_with_progress(
            &mut &b"abc"[..],
            &mut output,
            "weights",
            Some(3),
            &mut |p| {
                progress.push(p.bytes);
                Ok(())
            },
        );
        assert_eq!(result.unwrap(), 3);
        assert_eq!(progress, [0, 3]);
    }

    #[test]
    fn reports_missing_required_files() {
        let tmp =
            std::env::temp_dir().join(format!("puppygrad-assets-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join("config.json"), "{}").unwrap();

        let missing = missing_required_files(&tmp, &["config.json", "model.safetensors"]);

        assert_eq!(missing, ["model.safetensors"]);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn builds_huggingface_cache_path() {
        let path = huggingface_cache_model_dir("models", "Qwen/Qwen2.5-0.5B-Instruct", "main");

        assert_eq!(
            path,
            PathBuf::from("models")
                .join("huggingface")
                .join("Qwen-Qwen2.5-0.5B-Instruct")
                .join("main")
        );
    }
}
