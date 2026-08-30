/// \file test_vulkan_gemm.cpp
/// \brief Standalone hardware benchmark for Radeon 890M Vulkan Compute Acceleration
/// \author FastFlowLM Team
/// \date 2026-08-30

#include "gpu/vulkan_engine.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "=== FastFlowLM AMD Radeon 890M Vulkan Hardware Benchmark ===" << std::endl;
    std::cout << "=================================================================" << std::endl;

    VulkanComputeEngine engine;
    if (!engine.initialize()) {
        std::cerr << "[ERROR] Failed to initialize Vulkan Compute Engine on Radeon 890M!" << std::endl;
        return 1;
    }
    std::cout << "[Vulkan Init] Successfully connected to AMD Radeon 890M (RADV STRIX1) Compute Queue!" << std::endl;

    // Test MoE Grouped GEMM Dimensions: (M=64 tokens, K=2048 hidden, N=512 intermediate)
    uint32_t M = 64;
    uint32_t K = 2048;
    uint32_t N = 512;

    std::vector<float> A(M * K, 1.0f);
    std::vector<uint16_t> B(N * K, 0x3F80); // BF16 for 1.0f
    std::vector<float> C(M * N, 0.0f);

    // Warmup
    engine.execute_gemm(A.data(), B.data(), C.data(), M, K, N);

    // Benchmark 100 iterations
    int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        engine.execute_gemm(A.data(), B.data(), C.data(), M, K, N);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / iterations;
    double gflops = (2.0 * M * K * N * 1e-9) / (avg_ms * 1e-3);

    std::cout << "\n[Vulkan Benchmark Results]" << std::endl;
    std::cout << "  - Matrix Shape:  [" << M << " x " << K << "] * [" << K << " x " << N << "]" << std::endl;
    std::cout << "  - Average Time:  " << avg_ms << " ms per MoE GEMM" << std::endl;
    std::cout << "  - Compute Rate:  " << gflops << " GFLOP/s" << std::endl;
    std::cout << "  - Verification:  C[0] = " << C[0] << " (Expected: " << static_cast<float>(K) << ")" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
