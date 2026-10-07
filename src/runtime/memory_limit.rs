//! User-facing byte budgets. Bare numbers are GiB; explicit units are supported.
use std::{fmt, str::FromStr};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct MemoryLimit(pub usize);

impl FromStr for MemoryLimit {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let text = text.trim();
        let split = text
            .find(|c: char| c.is_ascii_alphabetic())
            .unwrap_or(text.len());
        let (number, unit) = text.split_at(split);
        let scale = match unit.trim().to_ascii_lowercase().as_str() {
            "" | "gib" | "g" => 1024f64.powi(3),
            "mib" | "m" => 1024f64.powi(2),
            "kib" | "k" => 1024.,
            "gb" => 1_000_000_000.,
            "mb" => 1_000_000.,
            "kb" => 1_000.,
            "b" => 1.,
            _ => return Err("memory unit must be B, KiB, MiB, GiB, KB, MB or GB".into()),
        };
        let bytes = number
            .trim()
            .parse::<f64>()
            .map_err(|_| "invalid memory size")?
            * scale;
        if !bytes.is_finite() || bytes < 1. || bytes >= usize::MAX as f64 {
            return Err(
                "memory size must be positive and fit in this platform's address space".into(),
            );
        }
        Ok(Self(bytes as usize))
    }
}
impl fmt::Display for MemoryLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}B", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_gib_defaults_and_explicit_units_without_overflow() {
        for (text, bytes) in [
            ("10", 10 * 1024usize.pow(3)),
            ("0.5", 512 * 1024usize.pow(2)),
            ("512MiB", 512 * 1024usize.pow(2)),
            ("2GB", 2_000_000_000),
            ("1B", 1),
        ] {
            assert_eq!(text.parse::<MemoryLimit>().unwrap().0, bytes);
        }
        for text in [
            "0",
            "-1",
            "NaN",
            "inf",
            "1e100",
            "999999999999999999999GiB",
            "12watts",
            "0.5B",
        ] {
            assert!(text.parse::<MemoryLimit>().is_err(), "{text}");
        }
    }
}
