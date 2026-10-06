//! Target selection and identity for compiled C artifacts.
use crate::compiler::pop::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    hash::{Hash, Hasher},
    path::PathBuf,
    process::{Command, Stdio},
};

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum CpuTarget {
    #[default]
    Generic,
    Native,
    Avx2,
}
impl std::fmt::Display for CpuTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Generic => "generic",
            Self::Native => "native",
            Self::Avx2 => "avx2",
        })
    }
}
impl CpuTarget {
    /// Libraries are compiled for immediate execution on the host.
    pub fn validate(self) -> Result<()> {
        if self == Self::Avx2 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if std::is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            return Err(Error(
                "--cpu-target avx2 requires an x86 host with AVX2 support".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct BuildOptions {
    pub cpu_target: CpuTarget,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct BuildInfo {
    pub cpu_target: CpuTarget,
    pub compiler: String,
    pub compiler_path: PathBuf,
    pub compiler_target: String,
    pub compiler_identity: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_fingerprint: Option<String>,
    pub flags: Vec<String>,
}

fn fingerprint(value: &impl Hash) -> String {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hash);
    format!("{:016x}", hash.finish())
}
fn compiler_path() -> Result<PathBuf> {
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let path = directory.join("cc");
        #[cfg(unix)]
        let executable = {
            use std::os::unix::fs::PermissionsExt;
            path.metadata()
                .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        };
        #[cfg(not(unix))]
        let executable = path.is_file();
        if executable {
            // Preserve the invocation name: compiler driver symlinks can affect behavior.
            return Ok(directory
                .canonicalize()
                .map_err(|e| Error(e.to_string()))?
                .join("cc"));
        }
    }
    Err(Error("C CPU backend needs cc on PATH".into()))
}
fn query(command: &mut Command) -> Result<String> {
    let output = command
        .stdin(Stdio::null())
        .output()
        .map_err(|e| Error(format!("C compiler query failed: {e}")))?;
    if !output.status.success() {
        return Err(Error(format!(
            "C compiler query failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}
impl BuildInfo {
    pub fn resolve(options: &BuildOptions) -> Result<Self> {
        options.cpu_target.validate()?;
        let compiler_path = compiler_path()?;
        let version = query(Command::new(&compiler_path).arg("--version"))?;
        let compiler_target = query(Command::new(&compiler_path).arg("-dumpmachine"))?
            .trim()
            .to_owned();
        let metadata = compiler_path.metadata().map_err(|e| Error(e.to_string()))?;
        let identity = fingerprint(&(
            compiler_path
                .canonicalize()
                .map_err(|e| Error(e.to_string()))?,
            &version,
            &compiler_target,
            metadata.len(),
            metadata.modified().ok(),
        ));
        let mut flags: Vec<_> = [
            "-std=c11",
            "-pthread",
            "-shared",
            "-fPIC",
            "-O2",
            "-fno-math-errno",
            "-fwrapv",
            "-ffp-contract=off",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        match options.cpu_target {
            CpuTarget::Generic => (),
            CpuTarget::Native => flags.push("-march=native".into()),
            CpuTarget::Avx2 => flags.push("-mavx2".into()),
        }
        let native_fingerprint = if options.cpu_target == CpuTarget::Native {
            let macros = query(Command::new(&compiler_path).args([
                "-march=native",
                "-dM",
                "-E",
                "-x",
                "c",
                "-",
            ]))?;
            let cpu = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
            let cpu: Vec<_> = cpu
                .lines()
                .filter(|s| {
                    [
                        "vendor_id",
                        "cpu family",
                        "model",
                        "stepping",
                        "flags",
                        "Features",
                        "CPU implementer",
                        "CPU part",
                    ]
                    .iter()
                    .any(|p| s.starts_with(p))
                })
                .collect();
            Some(fingerprint(&(macros, cpu)))
        } else {
            None
        };
        Ok(Self {
            cpu_target: options.cpu_target,
            compiler: version.lines().next().unwrap_or("cc").into(),
            compiler_path,
            compiler_target,
            compiler_identity: identity,
            native_fingerprint,
            flags,
        })
    }
}
