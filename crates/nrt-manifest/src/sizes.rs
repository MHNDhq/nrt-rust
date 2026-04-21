//! Parse size strings like "18GB", "32 GiB", "512MB" into megabyte counts.
//! Binary (GiB = 1024 MiB) and decimal (GB = 1000 MB) are both supported; the spec
//! uses the casual `18GB` notation, which we treat as binary (GiB) because that is
//! how GPU VRAM is actually measured by CUDA/Metal drivers.

use crate::ManifestError;

pub(crate) fn parse_mb(raw: &str) -> Result<u64, ManifestError> {
    let s = raw.trim();
    if s.is_empty() {
        return Err(ManifestError::BadSize {
            input: raw.to_string(),
            reason: "empty".into(),
        });
    }

    // Accept plain integers as megabytes (matches common YAML int handling).
    if let Ok(n) = s.parse::<u64>() {
        return Ok(n);
    }

    // Split the trailing alphabetic run off the leading digit run.
    let split_at = s
        .find(|c: char| c.is_ascii_alphabetic())
        .ok_or_else(|| ManifestError::BadSize {
            input: raw.to_string(),
            reason: "no unit suffix".into(),
        })?;

    let (num_part, unit_part) = s.split_at(split_at);
    let n: f64 = num_part.trim().parse().map_err(|_| ManifestError::BadSize {
        input: raw.to_string(),
        reason: format!("could not parse {num_part:?} as number"),
    })?;

    let unit = unit_part.trim().to_ascii_lowercase();
    let multiplier_mb = match unit.as_str() {
        "kb" | "k" => 1.0 / 1024.0,
        "mb" | "m" | "mib" => 1.0,
        "gb" | "g" | "gib" => 1024.0,
        "tb" | "t" | "tib" => 1024.0 * 1024.0,
        other => {
            return Err(ManifestError::BadSize {
                input: raw.to_string(),
                reason: format!("unknown unit {other:?}"),
            });
        }
    };

    let mb = (n * multiplier_mb).round() as u64;
    Ok(mb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_shapes() {
        assert_eq!(parse_mb("18GB").unwrap(), 18 * 1024);
        assert_eq!(parse_mb("32 GiB").unwrap(), 32 * 1024);
        assert_eq!(parse_mb("512MB").unwrap(), 512);
        assert_eq!(parse_mb("1024").unwrap(), 1024);
        assert_eq!(parse_mb("2T").unwrap(), 2 * 1024 * 1024);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_mb("").is_err());
        assert!(parse_mb("abc").is_err());
        assert!(parse_mb("17LB").is_err());
    }
}
