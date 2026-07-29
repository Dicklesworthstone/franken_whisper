//! Criterion benches for SQLite <-> JSONL sync paths.
//!
//! Covers:
//! - `sync::export` throughput from a seeded SQLite store
//! - `sync::import` throughput from a deterministic JSONL snapshot

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use serde_json::json;
use tempfile::tempdir;

use franken_whisper::model::{
    BackendKind, BackendParams, InputSource, ReplayEnvelope, RunEvent, RunReport,
    TranscribeRequest, TranscriptionResult, TranscriptionSegment,
};
use franken_whisper::storage::RunStore;
use franken_whisper::sync::{self, ConflictPolicy};

fn make_report(run_id: &str, db_path: &std::path::Path) -> RunReport {
    let segments = vec![
        TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(0.4),
            text: "hello".to_owned(),
            speaker: Some("SPEAKER_00".to_owned()),
            confidence: Some(0.9),
        },
        TranscriptionSegment {
            start_sec: Some(0.4),
            end_sec: Some(0.8),
            text: "world".to_owned(),
            speaker: Some("SPEAKER_00".to_owned()),
            confidence: Some(0.9),
        },
    ];

    let events = vec![
        RunEvent {
            seq: 1,
            ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            stage: "ingest".to_owned(),
            code: "ingest.ok".to_owned(),
            message: "ok".to_owned(),
            payload: json!({}),
        },
        RunEvent {
            seq: 2,
            ts_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.ok".to_owned(),
            message: "ok".to_owned(),
            payload: json!({"segments": 2}),
        },
    ];

    RunReport {
        run_id: run_id.to_owned(),
        trace_id: "bench-trace-id".to_owned(),
        started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
        finished_at_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
        input_path: "bench.wav".to_owned(),
        normalized_wav_path: "bench.norm.wav".to_owned(),
        request: TranscribeRequest {
            input: InputSource::File {
                path: std::path::PathBuf::from("bench.wav"),
            },
            backend: BackendKind::WhisperCpp,
            model: None,
            language: Some("en".to_owned()),
            translate: false,
            diarize: false,
            persist: true,
            db_path: db_path.to_path_buf(),
            timeout_ms: None,
            backend_params: BackendParams::default(),
        },
        result: TranscriptionResult {
            backend: BackendKind::WhisperCpp,
            transcript: "hello world".to_owned(),
            language: Some("en".to_owned()),
            segments,
            acceleration: None,
            diarization: None,
            raw_output: json!({"text":"hello world"}),
            artifact_paths: vec![],
        },
        events,
        warnings: vec![],
        evidence: vec![],
        replay: ReplayEnvelope::default(),
    }
}

fn seed_db(db_path: &std::path::Path, run_count: usize) {
    let store = RunStore::open(db_path).expect("store should open");
    for i in 0..run_count {
        let report = make_report(&format!("sync-bench-run-{i:04}"), db_path);
        store
            .persist_report(&report)
            .expect("seed persist should succeed");
    }
}

fn directory_size_bytes(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    let entries = std::fs::read_dir(path).expect("read_dir should succeed");
    for entry in entries {
        let entry = entry.expect("dir entry should be readable");
        if let Ok(metadata) = entry.metadata()
            && metadata.is_file()
        {
            total += metadata.len();
        }
    }
    total
}

fn bench_sync_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync/export");
    for run_count in [10usize, 50usize] {
        group.bench_with_input(BenchmarkId::new("runs", run_count), &run_count, |b, &n| {
            b.iter_batched(
                || {
                    let dir = tempdir().expect("tempdir");
                    let db_path = dir.path().join("storage.sqlite3");
                    let state_root = dir.path().join("state");
                    let output_dir = dir.path().join("snapshot");
                    seed_db(&db_path, n);
                    (dir, db_path, state_root, output_dir)
                },
                |(_dir, db_path, state_root, output_dir)| {
                    let manifest = sync::export(&db_path, &output_dir, &state_root)
                        .expect("export should succeed");
                    assert_eq!(manifest.row_counts.runs, n as u64);
                    let bytes = directory_size_bytes(&output_dir);
                    assert!(bytes > 0, "export should write non-empty snapshot");
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_sync_import(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync/import");

    for run_count in [10usize, 50usize] {
        let fixture_dir = tempdir().expect("tempdir");
        let source_db = fixture_dir
            .path()
            .join(format!("source-{run_count}.sqlite3"));
        let source_state = fixture_dir.path().join("source-state");
        let export_dir = fixture_dir.path().join(format!("snapshot-{run_count}"));
        seed_db(&source_db, run_count);
        let manifest = sync::export(&source_db, &export_dir, &source_state)
            .expect("fixture export should succeed");
        let snapshot_size = directory_size_bytes(&export_dir);
        group.throughput(Throughput::Bytes(snapshot_size.max(1)));

        group.bench_with_input(BenchmarkId::new("runs", run_count), &run_count, |b, &n| {
            b.iter_batched(
                || {
                    let iter_dir = tempdir().expect("tempdir");
                    let target_db = iter_dir.path().join("target.sqlite3");
                    let target_state = iter_dir.path().join("target-state");
                    (iter_dir, target_db, target_state)
                },
                |(_iter_dir, target_db, target_state)| {
                    let import_result = sync::import(
                        &target_db,
                        &export_dir,
                        &target_state,
                        ConflictPolicy::Reject,
                    )
                    .expect("import should succeed");
                    assert_eq!(import_result.runs_imported, n as u64);
                    assert_eq!(import_result.conflicts.len(), 0);
                },
                BatchSize::SmallInput,
            );
        });

        assert_eq!(manifest.row_counts.runs, run_count as u64);
    }

    group.finish();
}

/// First-time incremental export of a seeded store (exports all runs, so the
/// `export_table_{segments,events}_for_runs` writers run over every run_id). A/B the
/// per-run N+1 query vs the batched `WHERE run_id IN (…)` query via external env:
/// `FW_SYNC_BATCH_QUERY=0 cargo bench ...` (legacy per-run) vs unset/`1` (batched).
fn bench_sync_export_incremental(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync/export_incremental");

    for run_count in [50usize] {
        let fixture = tempdir().expect("tempdir");
        let src_db = fixture.path().join(format!("src-{run_count}.sqlite3"));
        seed_db(&src_db, run_count);

        group.bench_with_input(BenchmarkId::new("runs", run_count), &run_count, |b, &n| {
            b.iter_batched(
                || {
                    let iter_dir = tempdir().expect("tempdir");
                    let out_dir = iter_dir.path().join("out");
                    let state_root = iter_dir.path().join("state");
                    (iter_dir, out_dir, state_root)
                },
                |(_iter_dir, out_dir, state_root)| {
                    let manifest = sync::export_incremental(&src_db, &out_dir, &state_root)
                        .expect("incremental export should succeed");
                    assert_eq!(manifest.row_counts.runs, n as u64);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Isolated A/B for the `sha256_file` read-buffer bump (8 KiB → 64 KiB): checksum a
/// large JSONL-shaped file, comparing the old 8 KiB read loop to the new 64 KiB one
/// (8× fewer `read()` syscalls). Both arms produce the identical digest (asserted).
fn bench_sha256_buffer(c: &mut Criterion) {
    use sha2::{Digest, Sha256};
    use std::fs::File;
    use std::io::{Read, Write};

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("large.jsonl");
    {
        let mut f = std::io::BufWriter::new(File::create(&path).expect("create"));
        let line =
            b"{\"run_id\":\"sha-bench\",\"idx\":0,\"text\":\"the quick brown fox jumps over the lazy dog\"}\n";
        for _ in 0..200_000 {
            f.write_all(line).expect("write");
        }
        f.flush().expect("flush");
    }

    fn hash_with_buf(path: &std::path::Path, buf_len: usize) -> String {
        let mut file = File::open(path).expect("open");
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; buf_len];
        loop {
            let n = file.read(&mut buf).expect("read");
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        format!("{:x}", hasher.finalize())
    }

    // Byte-exactness guard: both buffer sizes must yield the same digest.
    assert_eq!(hash_with_buf(&path, 8192), hash_with_buf(&path, 64 * 1024));

    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let mut group = c.benchmark_group("sync/sha256_file");
    group.throughput(Throughput::Bytes(size));
    for arm in ["buf8k_r1", "buf64k_r1", "buf8k_r2", "buf64k_r2"] {
        let buf_len = if arm.starts_with("buf64k") { 64 * 1024 } else { 8192 };
        group.bench_function(arm, |b| {
            b.iter(|| hash_with_buf(&path, buf_len));
        });
    }
    group.finish();
}

/// Isolated A/B for the streaming-checksum change: writing a JSONL file then
/// re-reading it to SHA-256 (`sha256_file`) vs streaming the SHA while writing
/// (`HashingWriter`) — the latter drops the whole re-read pass. Both arms write the
/// same file and produce the identical digest (asserted); only the re-read differs.
fn bench_export_streaming_hash(c: &mut Criterion) {
    use sha2::{Digest, Sha256};
    use std::fs::File;
    use std::io::{BufWriter, Read, Write};

    struct Hw<W: Write> {
        inner: W,
        hasher: Sha256,
    }
    impl<W: Write> Write for Hw<W> {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            let n = self.inner.write(b)?;
            self.hasher.update(&b[..n]);
            Ok(n)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.inner.flush()
        }
    }

    let dir = tempdir().expect("tempdir");
    let lines: Vec<String> = (0..120_000)
        .map(|i| format!("{{\"id\":\"run-{i}\",\"text\":\"the quick brown fox jumps {i}\"}}"))
        .collect();

    fn write_then_reread(path: &std::path::Path, lines: &[String]) -> String {
        {
            let mut w = BufWriter::new(File::create(path).expect("create"));
            for l in lines {
                writeln!(w, "{l}").expect("write");
            }
            w.flush().expect("flush");
        }
        let mut f = File::open(path).expect("open");
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = f.read(&mut buf).expect("read");
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        format!("{:x}", hasher.finalize())
    }
    fn write_streaming(path: &std::path::Path, lines: &[String]) -> String {
        let mut w = Hw {
            inner: BufWriter::new(File::create(path).expect("create")),
            hasher: Sha256::new(),
        };
        for l in lines {
            writeln!(w, "{l}").expect("write");
        }
        w.flush().expect("flush");
        format!("{:x}", w.hasher.finalize())
    }

    let pa = dir.path().join("a.jsonl");
    let pb = dir.path().join("b.jsonl");
    assert_eq!(write_then_reread(&pa, &lines), write_streaming(&pb, &lines));

    let mut group = c.benchmark_group("sync/export_hash");
    for arm in ["reread_r1", "stream_r1", "reread_r2", "stream_r2"] {
        let stream = arm.starts_with("stream");
        let path = dir.path().join(format!("{arm}.jsonl"));
        group.bench_function(arm, |b| {
            b.iter(|| {
                if stream {
                    write_streaming(&path, &lines)
                } else {
                    write_then_reread(&path, &lines)
                }
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sync_export,
    bench_sync_import,
    bench_sync_export_incremental,
    bench_sha256_buffer,
    bench_export_streaming_hash,
);
criterion_main!(benches);
