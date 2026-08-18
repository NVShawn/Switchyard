// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional per-request transcript records (normalized + raw provider JSON).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use humantime::format_rfc3339_millis;
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::{ServerError, ServerResult};

pub(crate) const TRANSCRIPT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RedactionMode {
    Strict,
    Balanced,
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

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn redact_string(value: &str) -> String {
    let lowered = value.to_ascii_lowercase();
    if lowered.contains("bearer ") || lowered.contains("sk-") {
        return "[REDACTED]".to_string();
    }
    value.to_string()
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

fn truncate_string(value: &str, max_chars: usize) -> Value {
    if value.chars().count() <= max_chars {
        return Value::String(value.to_string());
    }
    let preview: String = value.chars().take(max_chars).collect();
    let digest = sha256_hex(value.as_bytes());
    json!({
        "sha256": digest,
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
            let s = match policy.redaction {
                RedactionMode::Off => s.clone(),
                _ => redact_string(s),
            };
            truncate_string(&s, policy.max_string_chars)
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

pub(crate) fn enforce_size_cap(mut record: Value, policy: TranscriptPolicy) -> Value {
    let Ok(bytes) = serde_json::to_vec(&record) else {
        return record;
    };
    if bytes.len() <= policy.max_bytes_per_record {
        return record;
    }

    if let Value::Object(map) = &mut record {
        map.insert("truncated".to_string(), Value::Bool(true));
        map.insert("len_bytes".to_string(), Value::Number(bytes.len().into()));
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

pub(crate) struct TranscriptLog {
    file: fs::File,
    policy: TranscriptPolicy,
}

impl TranscriptLog {
    pub(crate) fn new(path: impl Into<PathBuf>, policy: TranscriptPolicy) -> ServerResult<Self> {
        let path = path.into();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|error| transcript_log_error(&path, error))?;
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| transcript_log_error(&path, error))?;
        Ok(Self { file, policy })
    }

    pub(crate) fn append(&mut self, record: &TranscriptRecord) -> std::io::Result<()> {
        let value = serde_json::to_value(record).map_err(std::io::Error::other)?;
        let redacted = redact_and_truncate_value(&value, self.policy);
        let capped = enforce_size_cap(redacted, self.policy);
        let mut line = serde_json::to_vec(&capped).map_err(std::io::Error::other)?;
        line.push(b'\n');
        self.file.write_all(&line)
    }
}

fn transcript_log_error(path: &Path, error: std::io::Error) -> ServerError {
    ServerError::new(format!(
        "failed to initialize transcript log {}: {error}",
        path.display()
    ))
}
