//! Audited scalar ABI adapter for NumPy's original AVX512 float32 atan2 kernel.
//!
//! This private module is the sole unsafe-code allowance in the core. The
//! assembly is compiled only for Linux x86_64. Every entry checks all SKX
//! features and operating-system vector-state support through Rust's detector.

/// Return the original SVML lane result, or None when its instructions are unavailable.
pub(crate) fn atan2f(y: f32, x: f32) -> Option<f32> {
    if !available() {
        return None;
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        // SAFETY: available() checks the full instruction set and OS vector
        // state. The wrapper accepts and returns scalar floats under SysV,
        // broadcasts both initialized operands, and tail-calls the original
        // kernel, which preserves its callee-saved registers and stack frame.
        Some(unsafe { native_atan2f(y, x) })
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let _ = (y, x);
        None
    }
}

/// Test the complete instruction set required by NumPy's AVX512_SKX kernel.
pub(crate) fn available() -> bool {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512cd")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512vl")
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    false
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
extern "C" {
    #[link_name = concat!("__behave_fire_", env!("CARGO_PKG_VERSION"), "_svml_atan2f")]
    fn native_atan2f(y: f32, x: f32) -> f32;
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
std::arch::global_asm!(include_str!("svml/atan2f.s"), options(raw, att_syntax));

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
std::arch::global_asm!(
    concat!(
        ".text\n.p2align 4\n",
        ".globl \"__behave_fire_",
        env!("CARGO_PKG_VERSION"),
        "_svml_atan2f\"\n",
        ".hidden \"__behave_fire_",
        env!("CARGO_PKG_VERSION"),
        "_svml_atan2f\"\n",
        ".type \"__behave_fire_",
        env!("CARGO_PKG_VERSION"),
        "_svml_atan2f\",@function\n",
        "\"__behave_fire_",
        env!("CARGO_PKG_VERSION"),
        "_svml_atan2f\":\n",
        ".cfi_startproc\n",
        "vbroadcastss %xmm0, %zmm0\n",
        "vbroadcastss %xmm1, %zmm1\n",
        "jmp .Lbehave_svml_atan2f16\n",
        ".cfi_endproc\n",
        ".size \"__behave_fire_",
        env!("CARGO_PKG_VERSION"),
        "_svml_atan2f\",.-\"__behave_fire_",
        env!("CARGO_PKG_VERSION"),
        "_svml_atan2f\"\n",
    ),
    options(att_syntax)
);

#[cfg(test)]
mod tests {
    use super::{atan2f, available};

    #[test]
    fn svml_preserves_numpy_angles_and_special_values_or_rejects_unsupported_cpu() {
        if !available() {
            assert_eq!(atan2f(65534.0, -1.0), None);
            assert_eq!(atan2f(f32::NAN, f32::INFINITY), None);
            return;
        }
        for (y, x, expected) in [
            (65534.0, -1.0, 0x3fc9105b),
            (0.0, -1.0, 0x40490fdb),
            (-0.0, -1.0, 0xc0490fdb),
            (0.0, 1.0, 0x00000000),
            (-0.0, 1.0, 0x80000000),
            (f32::INFINITY, 1.0, 0x3fc90fdb),
            (f32::NEG_INFINITY, 1.0, 0xbfc90fdb),
            (f32::INFINITY, f32::INFINITY, 0x3f490fdb),
        ] {
            assert_eq!(atan2f(y, x).unwrap().to_bits(), expected, "atan2({y},{x})");
        }
        assert!(atan2f(f32::NAN, 1.0).unwrap().is_nan());
        assert!(atan2f(1.0, f32::NAN).unwrap().is_nan());
    }

    #[test]
    fn svml_uses_no_shared_mutable_state_across_threads() {
        let expected = atan2f(65534.0, -1.0);
        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| (0..256).map(|_| atan2f(65534.0, -1.0)).collect::<Vec<_>>())
            })
            .collect();
        for thread in threads {
            assert!(thread
                .join()
                .unwrap()
                .iter()
                .all(|result| *result == expected));
        }
    }
}
