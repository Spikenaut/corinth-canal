// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Integration tests for GPU Sentry telemetry capture.
//!
//! These tests verify that CUDA launch failures are properly captured and
//! formatted for Sentry, without requiring an actual Sentry DSN or network
//! connection.

#[cfg(feature = "cuda")]
mod gpu_telemetry_tests {
    use corinth_canal::gpu::wrappers::error::GpuError;
    use corinth_canal::gpu::wrappers::sentry_capture::{
        LaunchContext, LaunchType, capture_launch_failure, error_category, launch_event_message,
    };

    #[test]
    fn test_launch_context_creation() {
        let context = LaunchContext {
            kernel_name: "test_kernel".to_string(),
            launch_type: LaunchType::PtxFatbin,
            grid: (8, 1, 1),
            block: (256, 1, 1),
            shared_mem: 4096,
            neuron_count: Some(2048),
        };

        assert_eq!(context.kernel_name, "test_kernel");
        assert_eq!(context.grid, (8, 1, 1));
        assert_eq!(context.block, (256, 1, 1));
        assert_eq!(context.shared_mem, 4096);
        assert_eq!(context.neuron_count, Some(2048));
    }

    #[test]
    fn test_capture_without_sentry_is_safe() {
        // This should not panic even when Sentry is not initialized
        let context = LaunchContext {
            kernel_name: "gif_step_weighted".to_string(),
            launch_type: LaunchType::PtxFatbin,
            grid: (8, 1, 1),
            block: (256, 1, 1),
            shared_mem: 4096,
            neuron_count: Some(2048),
        };
        let error = GpuError::LaunchFailed("CUDA error: invalid configuration".to_string());

        assert_eq!(error_category(&error), "launch_failed");
        // Should be a no-op when Sentry is not configured.
        capture_launch_failure(context, &error, None);
    }

    #[test]
    fn test_capture_with_jit_logs() {
        let context = LaunchContext {
            kernel_name: "spiking_network".to_string(),
            launch_type: LaunchType::PtxFatbin,
            grid: (0, 0, 0), // N/A for module load
            block: (0, 0, 0),
            shared_mem: 0,
            neuron_count: None,
        };
        let error = GpuError::ModuleLoadFailed(
            "JIT compilation failed: unsupported PTX version".to_string(),
        );
        let jit_logs = (
            "error: ptxas fatal: Unsupported .version 8.5".to_string(),
            "info: 0 bytes gmem".to_string(),
        );

        assert_eq!(error_category(&error), "module_load_failed");
        assert_eq!(
            launch_event_message(&error, &context.kernel_name),
            "CUDA fatbin module load failed: spiking_network"
        );
        capture_launch_failure(context, &error, Some(jit_logs));
    }

    #[test]
    fn test_c_abi_shim_context() {
        let context = LaunchContext {
            kernel_name: "gif_step_weighted_f16".to_string(),
            launch_type: LaunchType::CAbiShim,
            grid: (8, 1, 1),
            block: (256, 1, 1),
            shared_mem: 8192,
            neuron_count: Some(2048),
        };
        let error = GpuError::LaunchFailed("CUDA runtime error code 1".to_string());

        assert_eq!(error_category(&error), "launch_failed");
        assert_eq!(
            launch_event_message(&error, &context.kernel_name),
            "CUDA kernel launch failed: gif_step_weighted_f16"
        );
        capture_launch_failure(context, &error, None);
    }

    #[test]
    fn test_various_error_types() {
        let test_cases = vec![
            (
                "launch_failed",
                GpuError::LaunchFailed("invalid grid size".to_string()),
            ),
            (
                "module_load_failed",
                GpuError::ModuleLoadFailed("PTX parse error".to_string()),
            ),
            (
                "memory_error",
                GpuError::MemoryError("out of memory".to_string()),
            ),
            (
                "kernel_not_found",
                GpuError::KernelNotFound("unknown_kernel".to_string()),
            ),
            ("no_gpu", GpuError::NoGpu),
        ];

        for (expected_category, error) in test_cases {
            let kernel_name = format!("test_{expected_category}");
            let context = LaunchContext {
                kernel_name: kernel_name.clone(),
                launch_type: LaunchType::PtxFatbin,
                grid: (1, 1, 1),
                block: (256, 1, 1),
                shared_mem: 0,
                neuron_count: None,
            };

            // The category is the Sentry grouping key, so assert it rather than
            // only checking that capture does not panic.
            assert_eq!(error_category(&error), expected_category);

            // The event title must carry the kernel name and never the error
            // payload, which can embed a fatbin or .ptx path.
            let message = launch_event_message(&error, &kernel_name);
            assert!(message.contains(&kernel_name), "message: {message}");

            capture_launch_failure(context, &error, None);
        }
    }

    #[test]
    fn test_large_grid_dimensions() {
        // Test with realistic large grid sizes
        let context = LaunchContext {
            kernel_name: "satsolver_aux_update".to_string(),
            launch_type: LaunchType::PtxFatbin,
            grid: (65535, 65535, 1), // Max grid dimensions
            block: (1024, 1, 1),     // Max block size
            shared_mem: 49152,       // Max shared memory
            neuron_count: None,
        };
        let error = GpuError::LaunchFailed("too many resources requested".to_string());

        assert_eq!(error_category(&error), "launch_failed");
        assert_eq!(
            launch_event_message(&error, &context.kernel_name),
            "CUDA kernel launch failed: satsolver_aux_update"
        );
        assert!(
            !launch_event_message(&error, &context.kernel_name)
                .contains("too many resources requested"),
            "event title must not carry the error payload"
        );
        capture_launch_failure(context, &error, None);
    }

    #[test]
    fn test_zero_dimensions() {
        // Test edge case with zero dimensions (module load scenario)
        let context = LaunchContext {
            kernel_name: "module_load".to_string(),
            launch_type: LaunchType::PtxFatbin,
            grid: (0, 0, 0),
            block: (0, 0, 0),
            shared_mem: 0,
            neuron_count: None,
        };
        let error = GpuError::ModuleLoadFailed("empty fatbin".to_string());

        assert_eq!(error_category(&error), "module_load_failed");
        assert_eq!(
            launch_event_message(&error, &context.kernel_name),
            "CUDA fatbin module load failed: module_load"
        );
        // Zero dimensions are the module-load shape; capture must still accept them.
        capture_launch_failure(context, &error, None);
    }

    #[test]
    fn test_neuron_count_variations() {
        // Test with neuron count
        let context_with_neurons = LaunchContext {
            kernel_name: "lif_step".to_string(),
            launch_type: LaunchType::PtxFatbin,
            grid: (8, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
            neuron_count: Some(2048),
        };
        let error = GpuError::LaunchFailed("test".to_string());
        assert_eq!(context_with_neurons.neuron_count, Some(2048));
        capture_launch_failure(context_with_neurons, &error, None);

        // Test without neuron count
        let context_without_neurons = LaunchContext {
            kernel_name: "satsolver_extract".to_string(),
            launch_type: LaunchType::PtxFatbin,
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
            neuron_count: None,
        };
        assert_eq!(context_without_neurons.neuron_count, None);
        capture_launch_failure(context_without_neurons, &error, None);
    }
}

#[cfg(not(feature = "cuda"))]
mod no_cuda_tests {
    #[test]
    fn test_cuda_feature_disabled() {
        // This test exists so the integration test target still compiles in CPU-only CI.
    }
}
