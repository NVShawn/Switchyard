// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI entrypoint for running the configured libsy server.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};

fn infer_transcript_path(routing_log: &Path) -> PathBuf {
    routing_log.with_file_name("routing.transcript.jsonl")
}

use clap::Parser;
use switchyard_server::config::{load_server_log_options, load_server_state};
use switchyard_server::{
    DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT, DEFAULT_LISTEN_BACKLOG, ServerError, ServerResult,
    ServerRunOptions, ServerState, TlsOptions, run_server,
};

const DEFAULT_HOST: IpAddr = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
const DEFAULT_PORT: u16 = 4000;

/// Command-line arguments accepted by the Rust server binary.
#[derive(Debug, Parser)]
#[command(
    name = "switchyard-server",
    about = "Serve explicitly configured libsy algorithms",
    version
)]
pub(crate) struct ServerArgs {
    /// TOML file defining LLM clients, targets, and algorithm routes.
    #[arg(long, value_name = "PATH")]
    config: PathBuf,

    /// Host address to bind.
    #[arg(long, default_value_t = DEFAULT_HOST)]
    host: IpAddr,

    /// Port to bind.
    #[arg(short, long, default_value_t = DEFAULT_PORT)]
    port: u16,

    /// TCP listen backlog passed to the socket before Axum accepts traffic.
    #[arg(long, default_value_t = DEFAULT_LISTEN_BACKLOG)]
    backlog: u32,

    /// Maximum time active requests may drain during shutdown.
    #[arg(long, default_value_t = humantime::Duration::from(DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT))]
    shutdown_timeout: humantime::Duration,

    /// Validate the algorithm and client configuration without binding a socket.
    #[arg(long)]
    dry_run: bool,

    /// Append durable per-request routing records to this JSONL file.
    #[arg(long, value_name = "PATH")]
    routing_log_file: Option<PathBuf>,

    /// Append best-effort transcript events (normalized + provider payloads) to this JSONL file.
    ///
    /// Records may contain prompts, tool arguments, and tool output. The file is
    /// created owner-only; retention and access control are the operator's
    /// responsibility.
    #[arg(long, value_name = "PATH")]
    transcript_log_file: Option<PathBuf>,

    /// Redaction mode for transcript records: strict, balanced, or off.
    ///
    /// strict redacts credential-like keys and free text; balanced keeps free
    /// text but still redacts suspicious keys and bare credentials; off disables
    /// redaction (string truncation still applies).
    #[arg(long, value_name = "MODE", default_value = "strict")]
    transcript_redaction: String,

    /// Store full unredacted provider JSON in transcript records (UNSAFE: writes
    /// secrets and prompts verbatim to disk).
    #[arg(long)]
    transcript_unsafe_full_raw: bool,

    /// Maximum serialized bytes retained per transcript record.
    #[arg(long, value_name = "BYTES", default_value_t = 262144)]
    transcript_max_bytes_per_record: usize,

    /// TLS certificate path in PEM format.
    #[arg(long, requires = "tls_key")]
    tls_cert: Option<PathBuf>,

    /// TLS private-key path in PEM format.
    #[arg(long, requires = "tls_cert")]
    tls_key: Option<PathBuf>,
}

impl ServerArgs {
    /// Parses command-line arguments using clap.
    pub(crate) fn parse_args() -> Self {
        Self::parse()
    }

    fn into_runtime(self) -> ServerResult<(ServerState, ServerRunOptions)> {
        let mut state = load_server_state(&self.config)?;
        let toml_logs = load_server_log_options(&self.config)?;

        let routing_path = self.routing_log_file.clone().or(toml_logs.routing_log_file);
        let transcript_path = self
            .transcript_log_file
            .clone()
            .or(toml_logs.transcript_log_file)
            .or_else(|| routing_path.as_deref().map(infer_transcript_path));

        if let Some(path) = routing_path {
            state = state.with_routing_log(path)?;
        }
        if let Some(path) = transcript_path {
            let redaction = toml_logs
                .transcript_redaction
                .filter(|_| self.transcript_redaction == "strict")
                .unwrap_or(self.transcript_redaction);
            let unsafe_full_raw = self.transcript_unsafe_full_raw
                || toml_logs.transcript_unsafe_full_raw.unwrap_or(false);
            let max_bytes = if self.transcript_max_bytes_per_record != 262144 {
                self.transcript_max_bytes_per_record
            } else {
                toml_logs
                    .transcript_max_bytes_per_record
                    .unwrap_or(self.transcript_max_bytes_per_record)
            };
            state =
                state.with_transcript_log_policy(path, &redaction, unsafe_full_raw, max_bytes)?;
        }
        let tls = match (self.tls_cert, self.tls_key) {
            (Some(cert), Some(key)) => {
                if !cert.exists() || !key.exists() {
                    return Err(ServerError::new(format!(
                        "invalid --tls-cert {} or --tls-key {}: file does not exist",
                        cert.display(),
                        key.display()
                    )));
                }
                Some(TlsOptions { cert, key })
            }
            _ => None,
        };
        let options = ServerRunOptions {
            addr: SocketAddr::new(self.host, self.port),
            backlog: self.backlog,
            dry_run: self.dry_run,
            shutdown_timeout: self.shutdown_timeout.into(),
            tls,
        };
        Ok((state, options))
    }
}

/// Loads the configured algorithms and starts the server.
pub(crate) async fn run(args: ServerArgs) -> ServerResult<()> {
    let (state, options) = args.into_runtime()?;
    run_server(state, options).await
}
