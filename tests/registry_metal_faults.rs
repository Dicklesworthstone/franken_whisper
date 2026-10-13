//! Registry-backed Metal command-buffer regression tests; no models required.
//!
//! Run only through the native release gate: it enables debug assertions for
//! ft-kernel-metal alone in a separate build directory. Production artifacts
//! are checked independently with fault injection absent. Faults leave commands
//! unsubmitted; these tests do NOT simulate an actual OS GPU reset or partially
//! executed GPU work. All checks fail rather than skip without a working GPU.
#![cfg(target_os = "macos")]
#![forbid(unsafe_code)]

use ft_kernel_metal::{CommandBufferState, Error, COMMAND_BUFFER_ATTEMPTS};

struct Disarm;
impl Drop for Disarm {
    fn drop(&mut self) {
        ft_kernel_metal::inject_command_buffer_faults(0);
    }
}

fn require_gpu_and_seam() -> Disarm {
    assert!(ft_kernel_metal::is_available(), "a working Metal GPU is required");
    assert!(
        ft_kernel_metal::inject_command_buffer_faults(0),
        "the Metal dependency must be built with debug assertions for fault tests"
    );
    Disarm
}

fn arm(count: u32) {
    assert!(ft_kernel_metal::inject_command_buffer_faults(count));
}

fn down() -> Error {
    Error::CommandBuffer(CommandBufferState::NotEnqueued)
}

fn inputs() -> (Vec<f32>, Vec<f32>) {
    ((0..65 * 33).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect(),
     (0..33 * 129).map(|i| ((i % 11) as f32 - 5.0) / 16.0).collect())
}

fn gemm(a: &[f32], b: &[f32], c: &mut [f32]) -> Result<(), Error> {
    ft_kernel_metal::sgemm(a, b, c, 65, 33, 129)
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "output bits changed at {i}");
    }
}

#[test]
#[ignore = "requires native Metal fault qualification"]
fn transient_retries_preserve_gemm_output() {
    let _disarm = require_gpu_and_seam();
    let (a, b) = inputs();
    let mut clean = vec![0.0; 65 * 129];
    gemm(&a, &b, &mut clean).expect("clean GPU submission also resets outage state");
    assert!(clean.iter().all(|v| v.is_finite()));
    for faults in 1..COMMAND_BUFFER_ATTEMPTS {
        let before = ft_kernel_metal::command_buffer_failures();
        arm(faults);
        let mut actual = vec![f32::NAN; clean.len()];
        gemm(&a, &b, &mut actual).expect("transient GPU submission must recover");
        arm(0);
        assert_bits(&actual, &clean);
        assert_eq!(ft_kernel_metal::command_buffer_failures() - before, u64::from(faults));
    }
}

#[test]
#[ignore = "requires native Metal fault qualification"]
fn exhausted_attempts_preserve_output_and_recovery_restores_retries() {
    let _disarm = require_gpu_and_seam();
    let (a, b) = inputs();
    let mut clean = vec![0.0; 65 * 129];
    gemm(&a, &b, &mut clean).unwrap();
    let sentinel = vec![f32::from_bits(0x7fc0_1234); clean.len()];
    let mut actual = sentinel.clone();
    let before = ft_kernel_metal::command_buffer_failures();
    arm(COMMAND_BUFFER_ATTEMPTS);
    assert_eq!(gemm(&a, &b, &mut actual), Err(down()));
    assert_eq!(ft_kernel_metal::command_buffer_failures() - before, u64::from(COMMAND_BUFFER_ATTEMPTS));
    assert_bits(&actual, &sentinel);
    // The outage circuit should consume one submission, not retry four times.
    let before = ft_kernel_metal::command_buffer_failures();
    arm(2);
    assert_eq!(gemm(&a, &b, &mut actual), Err(down()));
    assert_eq!(ft_kernel_metal::command_buffer_failures() - before, 1);
    assert_bits(&actual, &sentinel);
    arm(0);
    gemm(&a, &b, &mut actual).expect("the first completed buffer clears outage state");
    assert_bits(&actual, &clean);
    arm(1);
    gemm(&a, &b, &mut actual).expect("retries must be restored after recovery");
    assert_bits(&actual, &clean);
}

#[test]
#[ignore = "requires native Metal fault qualification"]
fn failed_buffers_release_queue_capacity() {
    let _disarm = require_gpu_and_seam();
    let (tx, rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _disarm = require_gpu_and_seam();
        let a = vec![1.0; 64];
        let mut c = vec![-123.0; 64];
        arm(COMMAND_BUFFER_ATTEMPTS);
        assert_eq!(ft_kernel_metal::sgemm(&a, &a, &mut c, 8, 8, 8), Err(down()));
        // Exceed the queue's 64 outstanding-command capacity three times.
        for _ in 0..192 {
            arm(1);
            assert_eq!(ft_kernel_metal::sgemm(&a, &a, &mut c, 8, 8, 8), Err(down()));
            assert!(c.iter().all(|v| *v == -123.0), "failed work changed output");
        }
        arm(0);
        ft_kernel_metal::sgemm(&a, &a, &mut c, 8, 8, 8).expect("queue recovers after failures");
        assert!(c.iter().all(|v| *v == 8.0));
        tx.send(()).unwrap();
    });
    rx.recv_timeout(std::time::Duration::from_secs(60))
        .expect("GPU command queue hung or worker failed after injected failures");
    worker.join().expect("queue-capacity worker panicked");
}

#[test]
#[ignore = "requires native Metal fault qualification"]
fn injected_faults_are_thread_local() {
    let _disarm = require_gpu_and_seam();
    let (a, b) = inputs();
    let mut clean = vec![0.0; 65 * 129];
    gemm(&a, &b, &mut clean).unwrap();
    let before = ft_kernel_metal::command_buffer_failures();
    arm(1);
    let other = std::thread::spawn(|| {
        let (a, b) = inputs();
        let mut c = vec![f32::NAN; 65 * 129];
        gemm(&a, &b, &mut c).expect("another thread must not consume this thread's fault");
        c
    }).join().unwrap();
    assert_bits(&other, &clean);
    assert_eq!(ft_kernel_metal::command_buffer_failures(), before);
    let mut actual = vec![f32::NAN; clean.len()];
    gemm(&a, &b, &mut actual).unwrap();
    assert_eq!(ft_kernel_metal::command_buffer_failures() - before, 1);
    assert_bits(&actual, &clean);
}

#[test]
#[ignore = "requires native Metal fault qualification"]
fn fused_encoder_retries_without_returning_partial_activations() {
    use ft_kernel_metal::fused::{EncoderGpu, LayerWeightsRef};
    let _disarm = require_gpu_and_seam();
    assert!(ft_kernel_metal::fused::is_available(), "fused GPU pipelines are required");
    let (d, ff, seq) = (64, 256, 65);
    let norm = vec![1.0; d];
    let bias = vec![0.0; d];
    let ff_bias = vec![0.0; ff];
    let w: Vec<f32> = (0..d * d).map(|i| ((i % 7) as f32 - 3.0) / 256.0).collect();
    let mlp: Vec<f32> = (0..d * ff).map(|i| ((i % 11) as f32 - 5.0) / 256.0).collect();
    let layer = LayerWeightsRef {
        ln1_g: &norm, ln1_b: &bias, wq: &w, bq: &bias, wk: &w,
        wv: &w, bv: &bias, wo: &w, bo: &bias, ln2_g: &norm, ln2_b: &bias,
        w1: &mlp, b1: &ff_bias, w2: &mlp, b2: &bias,
    };
    let encoder = EncoderGpu::new(d, 1, ff, &[layer]).expect("upload synthetic encoder weights");
    let input: Vec<f32> = (0..seq * d).map(|i| ((i % 17) as f32 - 8.0) / 16.0).collect();
    let clean = encoder.forward(&input, seq).expect("clean native fused encoder");
    assert_eq!(clean.len(), input.len());
    assert!(clean.iter().all(|v| v.is_finite()));
    for faults in 1..COMMAND_BUFFER_ATTEMPTS {
        let before = ft_kernel_metal::command_buffer_failures();
        arm(faults);
        let actual = encoder.forward(&input, seq).expect("fused encoder must retry its failed batch");
        arm(0);
        assert_eq!(ft_kernel_metal::command_buffer_failures() - before, u64::from(faults));
        assert_bits(&actual, &clean);
    }
    arm(COMMAND_BUFFER_ATTEMPTS);
    assert_eq!(encoder.forward(&input, seq), Err(down()));
    arm(0);
    assert_bits(&encoder.forward(&input, seq).expect("encoder recovers"), &clean);
}
