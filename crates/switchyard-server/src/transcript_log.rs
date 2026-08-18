// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional per-request transcript records (normalized + raw provider JSON).
//!
//! Transcript logging is opt-in and best-effort: records are redacted and
//! size-capped on the calling thread, then handed to a bounded background writer
//! so request handling never blocks on disk I/O. When the queue is full, records
//! are dropped and counted rather than applying backpressure to live traffic.
//!
//! # Privacy
//!
//! Records can contain prompts, tool arguments, tool outputs, and (when raw
//! provider payloads are captured) exact request/response bodies. Redaction is
//! heuristic; `unsafe_full_raw` disables it entirely. The log file is created
//! with owner-only permissions where the platform supports it, but operators are
//! responsible for retention, rotation, and access control.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use humantime::format_rfc3339_millis;
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::{ServerError, ServerResult};

pub(crate) const TRANSCRIPT_SCHEMA_VERSION: u32 = 1;

/// Bounded queue depth for the background writer. Chosen so a brief disk stall
/// buffers a burst without letting an unbounded backlog grow.
const WRITER_QUEUE_CAPACITY: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RedactionMode {
    /// Redact suspicious keys and any string that looks like a credential.
    Strict,
    /// Redact suspicious keys only; leave free-text message content intact.
    Balanced,
    /// No redaction. String truncation still applies unless `unsafe_full_raw`.
    Off,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TranscriptPolicy {
    pub(crate) redaction: RedactionMode,
    pub(crate) unsafe_full_raw: bool,
    pub(crate) max_bytes_per_record: usize,
    pub(crate) max_string_chars: usize,
}

impl Default for TranscriptPolicy {
    fn default() -> Self {
        Self {
            redaction: RedactionMode::Strict,
            unsafe_full_raw: false,
            max_bytes_per_record: 256 * 1024,
            max_string_chars: 4096,
        }
    }
}

impl TranscriptPolicy {
    /// Human-readable summary of the privacy posture for startup logging.
    pub(crate) fn privacy_summary(&self) -> String {
        let redaction = match self.redaction {
            RedactionMode::Strict => "strict",
            RedactionMode::Balanced => "balanced",
            RedactionMode::Off => "off",
        };
        format!(
            "redaction={redaction} unsafe_full_raw={} max_bytes_per_record={}",
            self.unsafe_full_raw, self.max_bytes_per_record
        )
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Whether a string looks like a bearer token or API key that must be redacted
/// even under `balanced` mode when it appears as a bare value.
fn looks_like_credential(value: &str) -> bool {
    let lowered = value.to_ascii_lowercase();
    lowered.contains("bearer ") || lowered.contains("sk-")
}

fn should_redact_key(key: &str) -> bool {
    let lowered = key.to_ascii_lowercase();
    lowered.contains("authorization")
        || lowered.contains("api_key")
        || lowered.ends_with("_key")
        || lowered.contains("token")
        || lowered.contains("secret")
        || lowered.contains("password")
}

/// Replace an oversized string with a hash + bounded preview so the record stays
/// small while preserving enough signal to correlate and mine.
fn truncate_string(value: &str, max_chars: usize) -> Value {
    if value.chars().count() <= max_chars {
        return Value::String(value.to_string());
    }
    let preview: String = value.chars().take(max_chars).collect();
    json!({
        "sha256": sha256_hex(value.as_bytes()),
        "len_chars": value.chars().count(),
        "preview": preview,
        "truncated": true,
    })
}

pub(crate) fn redact_and_truncate_value(value: &Value, policy: TranscriptPolicy) -> Value {
    if policy.unsafe_full_raw {
        return value.clone();
    }
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => value.clone(),
        Value::String(s) => {
            // Strict redacts credential-looking free text; balanced leaves free
            // text intact but still catches bare credentials; off leaves both.
            let redacted = match policy.redaction {
                RedactionMode::Strict if looks_like_credential(s) => "[REDACTED]".to_string(),
                RedactionMode::Balanced if looks_like_credential(s) => "[REDACTED]".to_string(),
                _ => s.clone(),
            };
            truncate_string(&redacted, policy.max_string_chars)
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| redact_and_truncate_value(item, policy))
                .collect(),
        ),
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, item) in map {
                if policy.redaction != RedactionMode::Off && should_redact_key(key) {
                    out.insert(key.clone(), Value::String("[REDACTED]".to_string()));
                    continue;
                }
                out.insert(key.clone(), redact_and_truncate_value(item, policy));
            }
            Value::Object(out)
        }
    }
}

/// Enforce a hard per-record size limit by dropping the heavy `raw`/`normalized`
/// payloads (replacing them with a hash summary) until the serialized record fits
/// under the cap. Metadata fields needed for correlation and mining are retained.
pub(crate) fn enforce_size_cap(mut record: Value, policy: TranscriptPolicy) -> Value {
    fn record_len(record: &Value) -> usize {
        serde_json::to_vec(record)
            .map(|bytes| bytes.len())
            .unwrap_or(0)
    }

    fn summarize(field: &mut Value) -> bool {
        // Already summarized (no payload to shed further).
        if field.is_null() {
            return false;
        }
        let bytes = serde_json::to_vec(field).unwrap_or_default();
        *field = json!({
            "sha256": sha256_hex(&bytes),
            "len_bytes": bytes.len(),
            "dropped": true,
        });
        true
    }

    if record_len(&record) <= policy.max_bytes_per_record {
        return record;
    }

    // Shed the largest payloads first, re-checking after each so we keep as much
    // as fits. Order: raw provider body, then normalized IR.
    for field in ["raw", "normalized"] {
        if record_len(&record) <= policy.max_bytes_per_record {
            break;
        }
        if let Value::Object(map) = &mut record
            && let Some(value) = map.get_mut(field)
            && summarize(value)
        {
            map.insert("truncated".to_string(), Value::Bool(true));
        }
    }

    record
}

pub(crate) fn now_rfc3339_millis() -> String {
    format_rfc3339_millis(std::time::SystemTime::now()).to_string()
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TranscriptEventKind {
    #[serde(rename = "normalized_request")]
    NormalizedRequest,
    #[serde(rename = "provider_request")]
    ProviderRequest,
    #[serde(rename = "provider_response")]
    ProviderResponse,
    #[serde(rename = "normalized_response")]
    NormalizedResponse,
    #[serde(rename = "error")]
    Error,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct TranscriptRecord {
    pub(crate) v: u32,
    pub(crate) ts: String,
    pub(crate) event: TranscriptEventKind,
    pub(crate) request_id: String,
    pub(crate) session_id: Option<String>,
    pub(crate) trial_id: Option<String>,
    pub(crate) wire_format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) route_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) selected_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) streaming: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) normalized: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) raw: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<Value>,
}

/// Serialize, redact, and size-cap a record into the exact bytes to append.
///
/// This runs on the calling thread so the background writer only performs I/O.
fn encode_line(record: &TranscriptRecord, policy: TranscriptPolicy) -> serde_json::Result<Vec<u8>> {
    let value = serde_json::to_value(record)?;
    let redacted = redact_and_truncate_value(&value, policy);
    let capped = enforce_size_cap(redacted, policy);
    let mut line = serde_json::to_vec(&capped)?;
    line.push(b'\n');
    Ok(line)
}

/// A non-blocking transcript log. Encoding happens on the caller; a dedicated
/// thread owns the file and drains a bounded queue. When the queue is full,
/// records are dropped and counted so a slow disk cannot stall serving.
pub(crate) struct TranscriptLog {
    sender: std::sync::mpsc::SyncSender<Vec<u8>>,
    policy: TranscriptPolicy,
    dropped: Arc<AtomicU64>,
    path: PathBuf,
}

impl TranscriptLog {
    pub(crate) fn new(path: impl Into<PathBuf>, policy: TranscriptPolicy) -> ServerResult<Self> {
        let path = path.into();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|error| transcript_log_error(&path, error))?;
        }
        let file = open_owner_only(&path).map_err(|error| transcript_log_error(&path, error))?;

        let (sender, receiver) = std::sync::mpsc::sync_channel::<Vec<u8>>(WRITER_QUEUE_CAPACITY);
        let writer_path = path.clone();
        std::thread::Builder::new()
            .name("switchyard-transcript".to_string())
            .spawn(move || background_writer(file, receiver, writer_path))
            .map_err(|error| transcript_log_error(&path, error))?;

        Ok(Self {
            sender,
            policy,
            dropped: Arc::new(AtomicU64::new(0)),
            path,
        })
    }

    /// Encode on the calling thread and enqueue for the background writer.
    ///
    /// Returns immediately. A full queue drops the record and increments the
    /// dropped counter; encoding failures are also counted as drops.
    pub(crate) fn append(&self, record: &TranscriptRecord) {
        let line = match encode_line(record, self.policy) {
            Ok(line) => line,
            Err(error) => {
                tracing::warn!(path = %self.path.display(), %error, "transcript encode failed");
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        if self.sender.try_send(line).is_err() {
            let dropped = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
            // Rate-limit the warning: only on the first drop and each power of two.
            if dropped.is_power_of_two() {
                tracing::warn!(
                    path = %self.path.display(),
                    dropped,
                    "transcript queue full; dropping records"
                );
            }
        }
    }
}

/// Drains the queue and appends each pre-encoded line, warning on write errors.
fn background_writer(
    mut file: fs::File,
    receiver: std::sync::mpsc::Receiver<Vec<u8>>,
    path: PathBuf,
) {
    for line in receiver {
        if let Err(error) = file.write_all(&line) {
            tracing::warn!(path = %path.display(), %error, "transcript log append failed");
        }
    }
}

/// Open the log for appending with owner-only permissions on Unix.
fn open_owner_only(path: &Path) -> std::io::Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    options.open(path)
}

fn transcript_log_error(path: &Path, error: std::io::Error) -> ServerError {
    ServerError::new(format!(
        "failed to initialize transcript log {}: {error}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(redaction: RedactionMode) -> TranscriptPolicy {
        TranscriptPolicy {
            redaction,
            ..TranscriptPolicy::default()
        }
    }

    #[test]
    fn strict_redacts_credential_free_text_and_suspicious_keys() {
        let value = json!({
            "authorization": "Bearer abc",
            "note": "use sk-secret-value now",
            "safe": "plain text",
        });
        let out = redact_and_truncate_value(&value, policy(RedactionMode::Strict));
        assert_eq!(out["authorization"], "[REDACTED]");
        assert_eq!(out["note"], "[REDACTED]");
        assert_eq!(out["safe"], "plain text");
    }

    #[test]
    fn balanced_keeps_free_text_but_still_redacts_keys_and_bare_credentials() {
        let value = json!({
            "api_key": "xyz",
            "prose": "a long normal sentence with no secrets",
            "leaked": "Bearer abc",
        });
        let out = redact_and_truncate_value(&value, policy(RedactionMode::Balanced));
        assert_eq!(out["api_key"], "[REDACTED]");
        assert_eq!(out["prose"], "a long normal sentence with no secrets");
        assert_eq!(out["leaked"], "[REDACTED]");
    }

    #[test]
    fn off_leaves_content_but_still_truncates_long_strings() {
        let long = "x".repeat(10_000);
        let value = json!({ "authorization": "Bearer abc", "body": long });
        let out = redact_and_truncate_value(&value, policy(RedactionMode::Off));
        assert_eq!(out["authorization"], "Bearer abc");
        assert_eq!(out["body"]["truncated"], true);
    }

    #[test]
    fn unsafe_full_raw_bypasses_redaction_and_truncation() {
        let long = "y".repeat(10_000);
        let value = json!({ "authorization": "Bearer abc", "body": long });
        let out = redact_and_truncate_value(
            &value,
            TranscriptPolicy {
                unsafe_full_raw: true,
                ..TranscriptPolicy::default()
            },
        );
        assert_eq!(out, value);
    }

    #[test]
    fn enforce_size_cap_drops_heavy_payloads_until_it_fits() {
        let big = json!({ "blob": "z".repeat(50_000) });
        let record = json!({
            "v": 1,
            "event": "normalized_response",
            "request_id": "r1",
            "raw": big.clone(),
            "normalized": big,
        });
        let capped = enforce_size_cap(
            record,
            TranscriptPolicy {
                max_bytes_per_record: 1024,
                ..TranscriptPolicy::default()
            },
        );
        assert_eq!(capped["truncated"], true);
        assert_eq!(capped["raw"]["dropped"], true);
        assert!(capped["raw"]["sha256"].is_string());
        // Correlation metadata survives the cap.
        assert_eq!(capped["request_id"], "r1");
        assert!(serde_json::to_vec(&capped).unwrap().len() <= 1024);
    }
}
