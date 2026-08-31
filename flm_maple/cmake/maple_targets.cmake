# ==============================================================================
# Maple-FLM Target Definitions for FastFlowLM
# ==============================================================================

# Maple integration test
set(TEST_MAPLE_SRCS ${SOURCES} test/test_maple_integration.cpp)
list(FILTER TEST_MAPLE_SRCS EXCLUDE REGEX ".*/src/main\\.cpp$")
add_executable(test_maple_integration ${TEST_MAPLE_SRCS} ${HEADERS})
target_include_directories(test_maple_integration PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/runner
    ${CMAKE_SOURCE_DIR}/server
    ${CMAKE_SOURCE_DIR}/pull
    ${FFMPEG_INCLUDE_DIRS}
)
if(NOT FLM_USE_HRX)
    if(NOT WIN32 AND XRT_FOUND)
        target_include_directories(test_maple_integration PUBLIC ${XRT_INCLUDE_DIRS})
    else()
        target_include_directories(test_maple_integration PUBLIC ${XRT_INCLUDE_DIR})
    endif()
endif()
target_compile_definitions(test_maple_integration PUBLIC
    DISABLE_ABI_CHECK=1
    CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
    CMAKE_XCLBIN_PREFIX="${CMAKE_XCLBIN_PREFIX}"
    __FLM_VERSION__=\"${FLM_VERSION}\"
    __NPU_VERSION__=\"${NPU_VERSION}\"
)
target_link_directories(test_maple_integration PUBLIC
    ${FLM_ENGINE_LIB_DIR}
    ${CMAKE_SOURCE_DIR}/lib
    ${FFMPEG_LIBRARY_DIRS}
)
target_link_libraries(test_maple_integration PUBLIC
    q4_npu_eXpress
    llama_npu
    qwen2_npu
    qwen2vl_npu
    qwen3_npu
    qwen3vl_npu
    qwen3_5vl_npu
    qwen3_5_omni_npu
    qwen3_6_moe_npu
    gemma_npu
    gemma_text_npu
    gemma4e_npu
    gpt_oss_npu
    whisper_npu
    gemma_embedding
    lfm2_npu
    phi4_npu
    nanbeige_npu
    dequant
    gemm
    lm_head
    mha
    ${FFMPEG_LIBRARIES}
    Boost::program_options
    CURL::libcurl
    PkgConfig::FFTW3
    PkgConfig::FFTW3F
    PkgConfig::FFTW3L
    tokenizers_cpp
)
if(NOT WIN32)
    target_compile_options(test_maple_integration PUBLIC -mavx -mavx2 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512bf16 -mavx512vnni -mfma -O3)
    if(NOT FLM_PORTABLE_BUILD)
        target_compile_definitions(test_maple_integration PUBLIC FASTFLOWLM_USE_READLINE=1)
        target_link_libraries(test_maple_integration PUBLIC PkgConfig::readline PkgConfig::ncurses)
    endif()
    target_link_libraries(test_maple_integration PUBLIC dl aiebu xrt_coreutil vulkan Threads::Threads)
    set_target_properties(test_maple_integration PROPERTIES
        BUILD_RPATH "${FLM_ENGINE_LIB_DIR}")
endif()

# Maple high-context test
set(TEST_MAPLE_HIGH_CTX_SRCS ${SOURCES} test/test_maple_high_context.cpp)
list(FILTER TEST_MAPLE_HIGH_CTX_SRCS EXCLUDE REGEX ".*/src/main\\.cpp$")
add_executable(test_maple_high_context ${TEST_MAPLE_HIGH_CTX_SRCS} ${HEADERS})
target_include_directories(test_maple_high_context PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/runner
    ${CMAKE_SOURCE_DIR}/server
    ${CMAKE_SOURCE_DIR}/pull
    ${FFMPEG_INCLUDE_DIRS}
)
if(NOT FLM_USE_HRX)
    if(NOT WIN32 AND XRT_FOUND)
        target_include_directories(test_maple_high_context PUBLIC ${XRT_INCLUDE_DIRS})
    else()
        target_include_directories(test_maple_high_context PUBLIC ${XRT_INCLUDE_DIR})
    endif()
endif()
target_compile_definitions(test_maple_high_context PUBLIC
    DISABLE_ABI_CHECK=1
    CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
    CMAKE_XCLBIN_PREFIX="${CMAKE_XCLBIN_PREFIX}"
    __FLM_VERSION__=\"${FLM_VERSION}\"
    __NPU_VERSION__=\"${NPU_VERSION}\"
)
target_link_directories(test_maple_high_context PUBLIC
    ${FLM_ENGINE_LIB_DIR}
    ${CMAKE_SOURCE_DIR}/lib
    ${FFMPEG_LIBRARY_DIRS}
)
target_link_libraries(test_maple_high_context PUBLIC
    q4_npu_eXpress
    llama_npu
    qwen2_npu
    qwen2vl_npu
    qwen3_npu
    qwen3vl_npu
    qwen3_5vl_npu
    qwen3_5_omni_npu
    qwen3_6_moe_npu
    gemma_npu
    gemma_text_npu
    gemma4e_npu
    gpt_oss_npu
    whisper_npu
    gemma_embedding
    lfm2_npu
    phi4_npu
    nanbeige_npu
    dequant
    gemm
    lm_head
    mha
    ${FFMPEG_LIBRARIES}
    Boost::program_options
    CURL::libcurl
    PkgConfig::FFTW3
    PkgConfig::FFTW3F
    PkgConfig::FFTW3L
    tokenizers_cpp
)
if(NOT WIN32)
    target_compile_options(test_maple_high_context PUBLIC -mavx -mavx2 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512bf16 -mavx512vnni -mfma -O3)
    if(NOT FLM_PORTABLE_BUILD)
        target_compile_definitions(test_maple_high_context PUBLIC FASTFLOWLM_USE_READLINE=1)
        target_link_libraries(test_maple_high_context PUBLIC PkgConfig::readline PkgConfig::ncurses)
    endif()
    target_link_libraries(test_maple_high_context PUBLIC dl aiebu xrt_coreutil vulkan Threads::Threads)
    set_target_properties(test_maple_high_context PROPERTIES
        BUILD_RPATH "${FLM_ENGINE_LIB_DIR}")
endif()

# Maple NPU Benchmark suite
set(TEST_MAPLE_NPU_BENCH_SRCS ${SOURCES} test/test_maple_npu_bench.cpp)
list(FILTER TEST_MAPLE_NPU_BENCH_SRCS EXCLUDE REGEX ".*/src/main\\.cpp$")
add_executable(test_maple_npu_bench ${TEST_MAPLE_NPU_BENCH_SRCS} ${HEADERS})
target_include_directories(test_maple_npu_bench PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/runner
    ${CMAKE_SOURCE_DIR}/server
    ${CMAKE_SOURCE_DIR}/pull
    ${FFMPEG_INCLUDE_DIRS}
)
if(NOT FLM_USE_HRX)
    if(NOT WIN32 AND XRT_FOUND)
        target_include_directories(test_maple_npu_bench PUBLIC ${XRT_INCLUDE_DIRS})
    else()
        target_include_directories(test_maple_npu_bench PUBLIC ${XRT_INCLUDE_DIR})
    endif()
endif()
target_compile_definitions(test_maple_npu_bench PUBLIC
    DISABLE_ABI_CHECK=1
    CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
    CMAKE_XCLBIN_PREFIX="${CMAKE_XCLBIN_PREFIX}"
    __FLM_VERSION__=\"${FLM_VERSION}\"
    __NPU_VERSION__=\"${NPU_VERSION}\"
)
target_link_directories(test_maple_npu_bench PUBLIC
    ${FLM_ENGINE_LIB_DIR}
    ${CMAKE_SOURCE_DIR}/lib
    ${FFMPEG_LIBRARY_DIRS}
)
target_link_libraries(test_maple_npu_bench PUBLIC
    q4_npu_eXpress
    llama_npu
    qwen2_npu
    qwen2vl_npu
    qwen3_npu
    qwen3vl_npu
    qwen3_5vl_npu
    qwen3_5_omni_npu
    qwen3_6_moe_npu
    gemma_npu
    gemma_text_npu
    gemma4e_npu
    gpt_oss_npu
    whisper_npu
    gemma_embedding
    lfm2_npu
    phi4_npu
    nanbeige_npu
    dequant
    gemm
    lm_head
    mha
    ${FFMPEG_LIBRARIES}
    Boost::program_options
    CURL::libcurl
    PkgConfig::FFTW3
    PkgConfig::FFTW3F
    PkgConfig::FFTW3L
    tokenizers_cpp
)
if(NOT WIN32)
    target_compile_options(test_maple_npu_bench PUBLIC -mavx -mavx2 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512bf16 -mavx512vnni -mfma -O3)
    if(NOT FLM_PORTABLE_BUILD)
        target_compile_definitions(test_maple_npu_bench PUBLIC FASTFLOWLM_USE_READLINE=1)
        target_link_libraries(test_maple_npu_bench PUBLIC PkgConfig::readline PkgConfig::ncurses)
    endif()
    target_link_libraries(test_maple_npu_bench PUBLIC dl aiebu xrt_coreutil vulkan Threads::Threads)
    set_target_properties(test_maple_npu_bench PROPERTIES
        BUILD_RPATH "${FLM_ENGINE_LIB_DIR}")
endif()

# Maple Capabilities & Benchmark Suite
set(TEST_MAPLE_CAPABILITIES_SRCS ${SOURCES} test/test_maple_capabilities.cpp)
list(FILTER TEST_MAPLE_CAPABILITIES_SRCS EXCLUDE REGEX ".*/src/main\\.cpp$")
add_executable(test_maple_capabilities ${TEST_MAPLE_CAPABILITIES_SRCS} ${HEADERS})
target_include_directories(test_maple_capabilities PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/runner
    ${CMAKE_SOURCE_DIR}/server
    ${CMAKE_SOURCE_DIR}/pull
    ${FFMPEG_INCLUDE_DIRS}
)
if(NOT FLM_USE_HRX)
    if(NOT WIN32 AND XRT_FOUND)
        target_include_directories(test_maple_capabilities PUBLIC ${XRT_INCLUDE_DIRS})
    else()
        target_include_directories(test_maple_capabilities PUBLIC ${XRT_INCLUDE_DIR})
    endif()
endif()
target_compile_definitions(test_maple_capabilities PUBLIC
    DISABLE_ABI_CHECK=1
    CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
    CMAKE_XCLBIN_PREFIX="${CMAKE_XCLBIN_PREFIX}"
    __FLM_VERSION__=\"${FLM_VERSION}\"
    __NPU_VERSION__=\"${NPU_VERSION}\"
)
target_link_directories(test_maple_capabilities PUBLIC
    ${FLM_ENGINE_LIB_DIR}
    ${CMAKE_SOURCE_DIR}/lib
    ${FFMPEG_LIBRARY_DIRS}
)
target_link_libraries(test_maple_capabilities PUBLIC
    q4_npu_eXpress
    llama_npu
    qwen2_npu
    qwen2vl_npu
    qwen3_npu
    qwen3vl_npu
    qwen3_5vl_npu
    qwen3_5_omni_npu
    qwen3_6_moe_npu
    gemma_npu
    gemma_text_npu
    gemma4e_npu
    gpt_oss_npu
    whisper_npu
    gemma_embedding
    lfm2_npu
    phi4_npu
    nanbeige_npu
    dequant
    gemm
    lm_head
    mha
    ${FFMPEG_LIBRARIES}
    Boost::program_options
    CURL::libcurl
    PkgConfig::FFTW3
    PkgConfig::FFTW3F
    PkgConfig::FFTW3L
    tokenizers_cpp
)
if(NOT WIN32)
    target_compile_options(test_maple_capabilities PUBLIC -mavx -mavx2 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512bf16 -mavx512vnni -mfma -O3)
    if(NOT FLM_PORTABLE_BUILD)
        target_compile_definitions(test_maple_capabilities PUBLIC FASTFLOWLM_USE_READLINE=1)
        target_link_libraries(test_maple_capabilities PUBLIC PkgConfig::readline PkgConfig::ncurses)
    endif()
    target_link_libraries(test_maple_capabilities PUBLIC dl aiebu xrt_coreutil vulkan Threads::Threads)
    set_target_properties(test_maple_capabilities PROPERTIES
        BUILD_RPATH "${FLM_ENGINE_LIB_DIR}")
endif()

# Maple Agentic & High-Context Benchmark Suite
set(TEST_AGENTIC_BENCH_SRCS ${SOURCES} test/test_agentic_benchmark.cpp)
list(FILTER TEST_AGENTIC_BENCH_SRCS EXCLUDE REGEX ".*/src/main\\.cpp$")
add_executable(test_agentic_benchmark ${TEST_AGENTIC_BENCH_SRCS} ${HEADERS})
target_include_directories(test_agentic_benchmark PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/runner
    ${CMAKE_SOURCE_DIR}/server
    ${CMAKE_SOURCE_DIR}/pull
    ${FFMPEG_INCLUDE_DIRS}
)
if(NOT FLM_USE_HRX)
    if(NOT WIN32 AND XRT_FOUND)
        target_include_directories(test_agentic_benchmark PUBLIC ${XRT_INCLUDE_DIRS})
    else()
        target_include_directories(test_agentic_benchmark PUBLIC ${XRT_INCLUDE_DIR})
    endif()
endif()
target_compile_definitions(test_agentic_benchmark PUBLIC
    DISABLE_ABI_CHECK=1
    CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
    CMAKE_XCLBIN_PREFIX="${CMAKE_XCLBIN_PREFIX}"
    __FLM_VERSION__=\"${FLM_VERSION}\"
    __NPU_VERSION__=\"${NPU_VERSION}\"
)
target_link_directories(test_agentic_benchmark PUBLIC
    ${FLM_ENGINE_LIB_DIR}
    ${CMAKE_SOURCE_DIR}/lib
    ${FFMPEG_LIBRARY_DIRS}
)
target_link_libraries(test_agentic_benchmark PUBLIC
    q4_npu_eXpress
    llama_npu
    qwen2_npu
    qwen2vl_npu
    qwen3_npu
    qwen3vl_npu
    qwen3_5vl_npu
    qwen3_5_omni_npu
    qwen3_6_moe_npu
    gemma_npu
    gemma_text_npu
    gemma4e_npu
    gpt_oss_npu
    whisper_npu
    gemma_embedding
    lfm2_npu
    phi4_npu
    nanbeige_npu
    dequant
    gemm
    lm_head
    mha
    ${FFMPEG_LIBRARIES}
    Boost::program_options
    CURL::libcurl
    PkgConfig::FFTW3
    PkgConfig::FFTW3F
    PkgConfig::FFTW3L
    tokenizers_cpp
)
if(NOT WIN32)
    target_compile_options(test_agentic_benchmark PUBLIC -mavx -mavx2 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512bf16 -mavx512vnni -mfma -O3)
    if(NOT FLM_PORTABLE_BUILD)
        target_compile_definitions(test_agentic_benchmark PUBLIC FASTFLOWLM_USE_READLINE=1)
        target_link_libraries(test_agentic_benchmark PUBLIC PkgConfig::readline PkgConfig::ncurses)
    endif()
    target_link_libraries(test_agentic_benchmark PUBLIC dl aiebu xrt_coreutil vulkan Threads::Threads)
    set_target_properties(test_agentic_benchmark PROPERTIES
        BUILD_RPATH "${FLM_ENGINE_LIB_DIR}")
endif()

# Maple-20B vs Qwen3.6 MoE Comparative Benchmark Suite
set(TEST_MAPLE_VS_QWEN36_SRCS ${SOURCES} test/test_maple_vs_qwen36.cpp)
list(FILTER TEST_MAPLE_VS_QWEN36_SRCS EXCLUDE REGEX ".*/src/main\\.cpp$")
add_executable(test_maple_vs_qwen36 ${TEST_MAPLE_VS_QWEN36_SRCS} ${HEADERS})
target_include_directories(test_maple_vs_qwen36 PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/runner
    ${CMAKE_SOURCE_DIR}/server
    ${CMAKE_SOURCE_DIR}/pull
    ${FFMPEG_INCLUDE_DIRS}
)
if(NOT FLM_USE_HRX)
    if(NOT WIN32 AND XRT_FOUND)
        target_include_directories(test_maple_vs_qwen36 PUBLIC ${XRT_INCLUDE_DIRS})
    else()
        target_include_directories(test_maple_vs_qwen36 PUBLIC ${XRT_INCLUDE_DIR})
    endif()
endif()
target_compile_definitions(test_maple_vs_qwen36 PUBLIC
    DISABLE_ABI_CHECK=1
    CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
    CMAKE_XCLBIN_PREFIX="${CMAKE_XCLBIN_PREFIX}"
    __FLM_VERSION__=\"${FLM_VERSION}\"
    __NPU_VERSION__=\"${NPU_VERSION}\"
)
target_link_directories(test_maple_vs_qwen36 PUBLIC
    ${FLM_ENGINE_LIB_DIR}
    ${CMAKE_SOURCE_DIR}/lib
    ${FFMPEG_LIBRARY_DIRS}
)
target_link_libraries(test_maple_vs_qwen36 PUBLIC
    q4_npu_eXpress
    llama_npu
    qwen2_npu
    qwen2vl_npu
    qwen3_npu
    qwen3vl_npu
    qwen3_5vl_npu
    qwen3_5_omni_npu
    qwen3_6_moe_npu
    gemma_npu
    gemma_text_npu
    gemma4e_npu
    gpt_oss_npu
    whisper_npu
    gemma_embedding
    lfm2_npu
    phi4_npu
    nanbeige_npu
    dequant
    gemm
    lm_head
    mha
    ${FFMPEG_LIBRARIES}
    Boost::program_options
    CURL::libcurl
    PkgConfig::FFTW3
    PkgConfig::FFTW3F
    PkgConfig::FFTW3L
    tokenizers_cpp
)
if(NOT WIN32)
    target_compile_options(test_maple_vs_qwen36 PUBLIC -mavx -mavx2 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512bf16 -mavx512vnni -mfma -O3)
    if(NOT FLM_PORTABLE_BUILD)
        target_compile_definitions(test_maple_vs_qwen36 PUBLIC FASTFLOWLM_USE_READLINE=1)
        target_link_libraries(test_maple_vs_qwen36 PUBLIC PkgConfig::readline PkgConfig::ncurses)
    endif()
    target_link_libraries(test_maple_vs_qwen36 PUBLIC dl aiebu xrt_coreutil vulkan Threads::Threads)
    set_target_properties(test_maple_vs_qwen36 PROPERTIES
        BUILD_RPATH "${FLM_ENGINE_LIB_DIR}")
endif()
