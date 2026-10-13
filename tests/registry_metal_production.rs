//! Hardware-required checks of the production registry Metal build.
//!
//! Deliberately ignored in ordinary tests: the release gate explicitly runs
//! every test with `--ignored` on Apple Silicon and requires named results.
//! A missing GPU is a failure, never a CPU fallback or an early-return pass.
#![cfg(target_os = "macos")]
#![forbid(unsafe_code)]

#[test]
#[ignore = "requires Apple Silicon Metal and a release build without fault injection"]
fn production_gemm_matches_reference_without_fault_injection() {
    assert!(ft_kernel_metal::is_available(), "a working Metal GPU is required");
    assert!(
        !ft_kernel_metal::inject_command_buffer_faults(0),
        "production Metal must not expose the debug-only fault seam"
    );
    // Exact dyadic inputs make this independent reference insensitive to
    // contraction/reduction order. Include edge tiles and a Whisper-like shape.
    for (m, k, n) in [(2, 3, 2), (64, 64, 64), (65, 33, 129), (300, 384, 512)] {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) / 16.0).collect();
        let mut actual = vec![f32::NAN; m * n];
        ft_kernel_metal::sgemm(&a, &b, &mut actual, m, k, n).expect("native Metal GEMM");
        for row in 0..m {
            for col in 0..n {
                let expected: f64 = (0..k)
                    .map(|e| f64::from(a[row * k + e]) * f64::from(b[e * n + col]))
                    .sum();
                assert_eq!(
                    f64::from(actual[row * n + col]), expected,
                    "native GEMM differs at shape {m}x{k}x{n}, ({row}, {col})"
                );
            }
        }
    }
}
