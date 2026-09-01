# ==============================================================================
# Maple-FLM Target Definitions for FastFlowLM
# ==============================================================================

# Add Maple engine sources and flags to the main flm CLI target
if(TARGET flm)
    target_sources(flm PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/common/models/maple/maple_npu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/gpu/vulkan_engine.cpp
    )
    if(NOT WIN32)
        target_compile_options(flm PRIVATE -mavx -mavx2 -mfma -O3)
        target_link_libraries(flm PUBLIC vulkan dl)
    endif()
endif()

# Common Maple standalone sources (zero dependency on upstream AVX-512 model binaries)
set(MAPLE_STANDALONE_SRCS
    common/models/maple/maple_npu.cpp
    common/AutoModel/modeling_maple.cpp
    common/AutoModel/automodel.cpp
    common/utils.cpp
    common/modules/sampler.cpp
    common/tokenizer/tokenizer.cpp
    common/gpu/vulkan_engine.cpp
    common/safetensors.cpp
    common/xrt_stub.cpp
    pull/model_downloader.cpp
    pull/download_model.cpp
)

set(MAPLE_BASE_LIBS
    Boost::program_options
    CURL::libcurl
    PkgConfig::FFTW3
    PkgConfig::FFTW3F
    PkgConfig::FFTW3L
    tokenizers_cpp
    vulkan
    dl
    Threads::Threads
)

# Macro to configure Maple test targets cleanly and portably
macro(add_maple_test_target TARGET_NAME TEST_SRC)
    add_executable(${TARGET_NAME}
        ${MAPLE_STANDALONE_SRCS}
        ${TEST_SRC}
    )
    target_include_directories(${TARGET_NAME} PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/runner
        ${CMAKE_SOURCE_DIR}/server
        ${CMAKE_SOURCE_DIR}/pull
        ${CMAKE_SOURCE_DIR}/include/xrt_headers
    )
    target_compile_definitions(${TARGET_NAME} PUBLIC
        DISABLE_ABI_CHECK=1
        CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
        CMAKE_XCLBIN_PREFIX="${CMAKE_XCLBIN_PREFIX}"
        __FLM_VERSION__=\"${FLM_VERSION}\"
        __NPU_VERSION__=\"${NPU_VERSION}\"
    )
    if(NOT WIN32)
        target_compile_options(${TARGET_NAME} PUBLIC -mavx -mavx2 -mfma -O3)
    endif()
    target_link_libraries(${TARGET_NAME} PUBLIC ${MAPLE_BASE_LIBS})
endmacro()

add_maple_test_target(test_maple_integration test/test_maple_integration.cpp)
add_maple_test_target(test_maple_high_context test/test_maple_high_context.cpp)
add_maple_test_target(test_maple_npu_bench test/test_maple_npu_bench.cpp)
add_maple_test_target(test_maple_capabilities test/test_maple_capabilities.cpp)
add_maple_test_target(test_agentic_benchmark test/test_agentic_benchmark.cpp)
add_maple_test_target(test_maple_vs_qwen36 test/test_maple_vs_qwen36.cpp)
