use std::env;
use std::hint::black_box;
use std::io::{self, Read, Write};
use std::process::{self, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const FIXED_POLL_MS: u64 = 50;
const ADAPTIVE_EARLY_POLL_MS: [u64; 6] = [1, 2, 4, 8, 16, 19];
const PROFILE_SAMPLES: usize = 9;
const NULL_PAIRS: usize = 11;
const ABBA_PAIRS: usize = 21;

#[derive(Clone, Copy)]
enum PollPolicy {
    Fixed,
    Adaptive,
}

#[derive(Debug, PartialEq, Eq)]
struct Observation {
    status_code: Option<i32>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn child_mode(delay_ms: u64, exit_code: i32) -> ! {
    thread::sleep(Duration::from_millis(delay_ms));

    let mut stdout = io::stdout().lock();
    stdout
        .write_all(b"fw-poll-stdout\0\xff\n")
        .expect("write child stdout");
    stdout.flush().expect("flush child stdout");

    let mut stderr = io::stderr().lock();
    stderr
        .write_all(b"fw-poll-stderr\0\xfe\n")
        .expect("write child stderr");
    stderr.flush().expect("flush child stderr");
    process::exit(exit_code);
}

fn spawn_child(delay_ms: u64, exit_code: i32) -> io::Result<process::Child> {
    Command::new(env::current_exe()?)
        .arg("child")
        .arg(delay_ms.to_string())
        .arg(exit_code.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
}

fn join_pipe(reader: thread::JoinHandle<io::Result<Vec<u8>>>, name: &str) -> io::Result<Vec<u8>> {
    reader
        .join()
        .map_err(|_| io::Error::other(format!("{name} reader panicked")))?
}

fn pipe_reader<R>(mut pipe: R) -> thread::JoinHandle<io::Result<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut bytes = Vec::new();
        pipe.read_to_end(&mut bytes)?;
        Ok(bytes)
    })
}

fn observation(
    status: ExitStatus,
    stdout: thread::JoinHandle<io::Result<Vec<u8>>>,
    stderr: thread::JoinHandle<io::Result<Vec<u8>>>,
) -> io::Result<Observation> {
    Ok(Observation {
        status_code: status.code(),
        stdout: join_pipe(stdout, "stdout")?,
        stderr: join_pipe(stderr, "stderr")?,
    })
}

fn run_polled(delay_ms: u64, exit_code: i32, policy: PollPolicy) -> io::Result<Observation> {
    let mut child = spawn_child(delay_ms, exit_code)?;
    let stdout = pipe_reader(child.stdout.take().expect("piped child stdout"));
    let stderr = pipe_reader(child.stderr.take().expect("piped child stderr"));
    let mut adaptive_poll_index = 0usize;

    loop {
        if let Some(status) = child.try_wait()? {
            return observation(status, stdout, stderr);
        }

        let poll_ms = match policy {
            PollPolicy::Fixed => FIXED_POLL_MS,
            PollPolicy::Adaptive => ADAPTIVE_EARLY_POLL_MS
                .get(adaptive_poll_index)
                .copied()
                .unwrap_or(FIXED_POLL_MS),
        };
        thread::sleep(Duration::from_millis(poll_ms));
        if matches!(policy, PollPolicy::Adaptive) {
            adaptive_poll_index = adaptive_poll_index.saturating_add(1);
        }
    }
}

fn run_direct(delay_ms: u64, exit_code: i32) -> io::Result<Observation> {
    let output = Command::new(env::current_exe()?)
        .arg("child")
        .arg(delay_ms.to_string())
        .arg(exit_code.to_string())
        .output()?;
    Ok(Observation {
        status_code: output.status.code(),
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

fn checksum(observation: &Observation) -> u64 {
    let mut sum = observation.status_code.unwrap_or(-1) as u64;
    for byte in observation.stdout.iter().chain(&observation.stderr) {
        sum = sum.rotate_left(5) ^ u64::from(*byte);
    }
    sum
}

fn elapsed_ns<F>(run: F) -> u128
where
    F: FnOnce() -> io::Result<Observation>,
{
    let started = Instant::now();
    let observed = run().expect("child invocation");
    let elapsed = started.elapsed().as_nanos();
    black_box(checksum(&observed));
    elapsed
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean
}

fn percentile(sorted: &[f64], percentile: usize) -> f64 {
    let index = (sorted.len() - 1) * percentile / 100;
    sorted[index]
}

fn assert_parity() {
    for delay_ms in [0, 5, 100] {
        for exit_code in [0, 17] {
            let direct = run_direct(delay_ms, exit_code).expect("direct child");
            let fixed = run_polled(delay_ms, exit_code, PollPolicy::Fixed).expect("fixed child");
            let adaptive =
                run_polled(delay_ms, exit_code, PollPolicy::Adaptive).expect("adaptive child");
            assert_eq!(fixed, direct, "fixed parity failed at {delay_ms} ms");
            assert_eq!(adaptive, direct, "adaptive parity failed at {delay_ms} ms");
        }
    }
}

fn profile() {
    assert_parity();
    for delay_ms in [0, 5, 100] {
        let mut direct = Vec::with_capacity(PROFILE_SAMPLES);
        let mut fixed = Vec::with_capacity(PROFILE_SAMPLES);
        let mut adaptive = Vec::with_capacity(PROFILE_SAMPLES);
        for _ in 0..PROFILE_SAMPLES {
            direct.push(elapsed_ns(|| run_direct(delay_ms, 0)));
            fixed.push(elapsed_ns(|| run_polled(delay_ms, 0, PollPolicy::Fixed)));
            adaptive.push(elapsed_ns(|| run_polled(delay_ms, 0, PollPolicy::Adaptive)));
        }

        let direct_ns = median(&mut direct);
        let fixed_ns = median(&mut fixed);
        let adaptive_ns = median(&mut adaptive);
        let polling_share = fixed_ns.saturating_sub(direct_ns) as f64 / fixed_ns as f64;
        println!(
            "PROFILE delay_ms={delay_ms} direct_us={:.3} fixed_us={:.3} adaptive_us={:.3} fixed_polling_share={:.6}",
            direct_ns as f64 / 1_000.0,
            fixed_ns as f64 / 1_000.0,
            adaptive_ns as f64 / 1_000.0,
            polling_share,
        );
    }
    println!("PARITY exact_status_stdout_stderr=true shapes=6");
}

fn null_control() {
    let mut ratios = Vec::with_capacity(NULL_PAIRS * 2);
    for _ in 0..NULL_PAIRS {
        let fixed_a = elapsed_ns(|| run_polled(75, 0, PollPolicy::Fixed));
        let fixed_b = elapsed_ns(|| run_polled(75, 0, PollPolicy::Fixed));
        let fixed_b_again = elapsed_ns(|| run_polled(75, 0, PollPolicy::Fixed));
        let fixed_a_again = elapsed_ns(|| run_polled(75, 0, PollPolicy::Fixed));
        ratios.push(fixed_b as f64 / fixed_a as f64);
        ratios.push(fixed_b_again as f64 / fixed_a_again as f64);
    }
    let cv = coefficient_of_variation(&ratios);
    ratios.sort_by(f64::total_cmp);
    let p10 = percentile(&ratios, 10);
    let median_ratio = percentile(&ratios, 50);
    let p90 = percentile(&ratios, 90);
    println!(
        "NULL delay_ms=75 fixed_b_over_fixed_a_p10={p10:.6} median={median_ratio:.6} p90={p90:.6} cv={cv:.6} samples={} ratios={ratios:?}",
        NULL_PAIRS * 2
    );
}

fn measure_shape(delay_ms: u64) {
    let mut fixed = Vec::with_capacity(ABBA_PAIRS * 2);
    let mut adaptive = Vec::with_capacity(ABBA_PAIRS * 2);
    let mut ratios = Vec::with_capacity(ABBA_PAIRS * 2);
    let mut wins = 0usize;

    for _ in 0..ABBA_PAIRS {
        let fixed_a = elapsed_ns(|| run_polled(delay_ms, 0, PollPolicy::Fixed));
        let adaptive_a = elapsed_ns(|| run_polled(delay_ms, 0, PollPolicy::Adaptive));
        let adaptive_b = elapsed_ns(|| run_polled(delay_ms, 0, PollPolicy::Adaptive));
        let fixed_b = elapsed_ns(|| run_polled(delay_ms, 0, PollPolicy::Fixed));
        wins += usize::from(adaptive_a < fixed_a) + usize::from(adaptive_b < fixed_b);
        ratios.push(adaptive_a as f64 / fixed_a as f64);
        ratios.push(adaptive_b as f64 / fixed_b as f64);
        fixed.extend([fixed_a, fixed_b]);
        adaptive.extend([adaptive_a, adaptive_b]);
    }

    let fixed_ns = median(&mut fixed);
    let adaptive_ns = median(&mut adaptive);
    let cv = coefficient_of_variation(&ratios);
    ratios.sort_by(f64::total_cmp);
    let p10 = percentile(&ratios, 10);
    let median_ratio = percentile(&ratios, 50);
    let p90 = percentile(&ratios, 90);
    println!(
        "AB delay_ms={delay_ms} fixed_us={:.3} adaptive_us={:.3} adaptive_over_fixed_median={median_ratio:.6} p10={p10:.6} p90={p90:.6} cv={cv:.6} median_arm_speedup={:.6} wins={wins}/{} ratios={ratios:?}",
        fixed_ns as f64 / 1_000.0,
        adaptive_ns as f64 / 1_000.0,
        fixed_ns as f64 / adaptive_ns as f64,
        ABBA_PAIRS * 2,
    );
}

fn measure() {
    assert_parity();
    null_control();
    for delay_ms in [0, 5, 100] {
        measure_shape(delay_ms);
    }
    println!("PARITY exact_status_stdout_stderr=true shapes=6");
}

fn parse_child_arg(args: &[String], index: usize, name: &str) -> u64 {
    args.get(index)
        .unwrap_or_else(|| panic!("missing {name}"))
        .parse()
        .unwrap_or_else(|_| panic!("invalid {name}"))
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match args.first().map(String::as_str) {
        Some("child") => {
            let delay_ms = parse_child_arg(&args, 1, "delay_ms");
            let exit_code = parse_child_arg(&args, 2, "exit_code") as i32;
            child_mode(delay_ms, exit_code);
        }
        Some("profile") => profile(),
        Some("measure") => measure(),
        _ => panic!("usage: process_poll_perf <profile|measure>"),
    }
}
