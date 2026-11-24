cmake_minimum_required(VERSION 3.20)
project(hyphy3 VERSION 3.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Build-speed options -----------------------------------------------------------
# All targets share the same large header stack (nlohmann/json, Eigen, core and
# analysis headers), so full builds parse it once per translation unit.
#
# - ccache: if found on PATH, identical translation units are served from
#   cache. Harmless when absent.
find_program(HYPHY3_CCACHE_PROGRAM ccache)
if(HYPHY3_CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${HYPHY3_CCACHE_PROGRAM}")
endif()
#
# - HYPHY3_FAST_COMPILE=ON: compiles at -O1 for faster iteration builds.
#   Release artifacts must be built without it.
option(HYPHY3_FAST_COMPILE "Trade optimization for compile speed (dev builds)" OFF)
if(HYPHY3_FAST_COMPILE)
    string(REGEX REPLACE "-O3" "-O1" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "-O3" "-O1" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()
#
# - HYPHY3_PCH (default ON): precompile the stable part of the shared header
#   stack once (src/pch.hpp) and reuse it for every target. Set
#   -DHYPHY3_PCH=OFF to disable.
option(HYPHY3_PCH "Shared precompiled header for all executable targets" ON)
#
# - HYPHY3_BUILD_TESTS (default ON): build the test executables and register the
#   ctest suite. Needs tests/ and benchmarks/data. The Dockerfile passes
#   -DHYPHY3_BUILD_TESTS=OFF so the image build does not need them.
option(HYPHY3_BUILD_TESTS "Build the test suite" ON)
#
# - HYPHY3_STANDALONE_SYMLINKS (default OFF): the hyphy3 driver already
#   dispatches on argv[0], so the hyphy3-* CLI names work as symlinks to the
#   driver. Enabling this skips ~20 duplicate app-TU compilations (~40% of the
#   build) and shrinks installs by not embedding the analysis code in 18
#   separate binaries. Build-tree targets hyphy3-fel etc. are not created when
#   ON; the names appear at install time.
option(HYPHY3_STANDALONE_SYMLINKS "Install hyphy3-* names as symlinks to hyphy3 instead of separate executables" OFF)

# Header-only Eigen 3.4.0 (SOURCE_SUBDIR points to a non-existent subdir so
# FetchContent_MakeAvailable populates the headers without running Eigen's
# legacy CMakeLists.txt or triggering CMP0169 warnings).
include(FetchContent)
FetchContent_Declare(
    Eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR _cmake_header_only_noop
)
FetchContent_MakeAvailable(Eigen3)

# Core interface library
add_library(hyphy3_core INTERFACE)
add_library(hyphy_core ALIAS hyphy3_core)

target_include_directories(hyphy3_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${eigen3_SOURCE_DIR}
)

# Build metadata for the JSON output software block (consumed by
# include/hyphy/core/json_output.hpp). git_commit is captured at configure
# time; falls back to "unknown" outside a git checkout. Pass
# -DHYPHY3_GIT_COMMIT=<value> to pin it: the commit becomes a compile
# definition of every target, so a value that changes per commit defeats ccache
# (CI pins it to "ci").
if(NOT DEFINED HYPHY3_GIT_COMMIT)
    execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE HYPHY3_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
if(NOT HYPHY3_GIT_COMMIT)
    set(HYPHY3_GIT_COMMIT "unknown")
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(HYPHY3_BUILD_TYPE_VALUE "unknown")
else()
    set(HYPHY3_BUILD_TYPE_VALUE "${CMAKE_BUILD_TYPE}")
endif()
target_compile_definitions(hyphy3_core INTERFACE
    HYPHY3_VERSION="${PROJECT_VERSION}"
    HYPHY3_GIT_COMMIT="${HYPHY3_GIT_COMMIT}"
    HYPHY3_BUILD_TYPE="${HYPHY3_BUILD_TYPE_VALUE}"
)

# OpenMP support (skipped under Emscripten WASM cross-compilation)
if(NOT EMSCRIPTEN)
    find_package(OpenMP)
    if(NOT OpenMP_CXX_FOUND)
        foreach(_omp_prefix "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
            if(EXISTS "${_omp_prefix}")
                set(OpenMP_ROOT "${_omp_prefix}")
                find_package(OpenMP)
                if(OpenMP_CXX_FOUND)
                    break()
                endif()
            endif()
        endforeach()
    endif()
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP CXX found: ${OpenMP_CXX_VERSION}")
        target_link_libraries(hyphy3_core INTERFACE OpenMP::OpenMP_CXX)
    endif()
endif()

# Apple Accelerate (AMX / vDSP) hardware BLAS backend for batched 61xP GEMM Inside-Outside
if(APPLE AND NOT EMSCRIPTEN)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        message(STATUS "Apple Accelerate framework found: ${ACCELERATE_FRAMEWORK}")
        target_link_libraries(hyphy3_core INTERFACE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(hyphy3_core INTERFACE EIGEN_USE_BLAS ACCELERATE_NEW_LAPACK)
    endif()
endif()

if(EMSCRIPTEN)
    message(STATUS "Configuring HyPhy 3 WALRUS for WebAssembly (Emscripten SIMD128)")
    set(HYPHY3_BUILD_TESTS OFF CACHE BOOL "Disable tests under Emscripten" FORCE)
    set(HYPHY3_PCH OFF CACHE BOOL "Disable PCH under Emscripten" FORCE)
    target_compile_options(hyphy3_core INTERFACE -msimd128 -fexceptions)
endif()

# HTML results viewer (viz/fel-viewer), embedded into the binaries at build time so hyphy3 can
# write self-contained reports with no Node, Python or companion files (cmake/EmbedViewer.cmake)
set(HYPHY3_VIEWER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/viz/fel-viewer)
set(HYPHY3_OHMIC_VIEWER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/viz/ohmic-viewer)
set(HYPHY3_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(HYPHY3_VIEWER_HEADER ${HYPHY3_GENERATED_DIR}/hyphy/report/viewer_assets.hpp)
set(HYPHY3_VIEWER_STAMP ${HYPHY3_GENERATED_DIR}/viewer_assets.stamp)
file(GLOB_RECURSE HYPHY3_VIEWER_SOURCES CONFIGURE_DEPENDS
    ${HYPHY3_VIEWER_DIR}/src/*.js
    ${HYPHY3_VIEWER_DIR}/src/*.css
    ${HYPHY3_OHMIC_VIEWER_DIR}/src/*.js
    ${HYPHY3_OHMIC_VIEWER_DIR}/src/*.css
)
add_custom_command(
    OUTPUT ${HYPHY3_VIEWER_STAMP}
    BYPRODUCTS ${HYPHY3_VIEWER_HEADER}
    COMMAND ${CMAKE_COMMAND}
        -DVIEWER_DIR=${HYPHY3_VIEWER_DIR}
        -DOHMIC_VIEWER_DIR=${HYPHY3_OHMIC_VIEWER_DIR}
        -DOUTPUT=${HYPHY3_VIEWER_HEADER}
        -DSTAMP=${HYPHY3_VIEWER_STAMP}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedViewer.cmake
    DEPENDS ${HYPHY3_VIEWER_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedViewer.cmake
    COMMENT "Embedding HTML results viewer"
    VERBATIM
)
add_custom_target(hyphy3_viewer_assets DEPENDS ${HYPHY3_VIEWER_STAMP})

add_library(hyphy3_report INTERFACE)
target_include_directories(hyphy3_report INTERFACE ${HYPHY3_GENERATED_DIR})
target_link_libraries(hyphy3_report INTERFACE hyphy3_core)
add_dependencies(hyphy3_report hyphy3_viewer_assets)

# Also expose generated viewer_assets.hpp on hyphy3_core so any target including
# hyphy/core/cli_common.hpp -> hyphy/report/html_report.hpp finds it cleanly.
target_include_directories(hyphy3_core INTERFACE ${HYPHY3_GENERATED_DIR})
add_dependencies(hyphy3_core hyphy3_viewer_assets)

# Out-of-line JSON emitters (src/analyses/*.cpp). The emitter bodies live in
# this one static library instead of the analysis headers so that editing an
# emitter recompiles a single TU rather than all ~45 consumers.
add_library(hyphy3_emitters STATIC
    src/analyses/absrel.cpp
    src/analyses/busted.cpp
    src/analyses/busted_impl.cpp
    src/analyses/characterize_impl.cpp
    src/analyses/contrast_fel.cpp
    src/analyses/directional_fel.cpp
    src/analyses/fel.cpp
    src/analyses/fubar.cpp
    src/analyses/gard.cpp
    src/analyses/joint_align_impl.cpp
    src/analyses/meme.cpp
    src/analyses/mss.cpp
    src/analyses/ohmic.cpp
    src/analyses/prime.cpp
    src/analyses/relax.cpp
    src/analyses/slac.cpp
    src/analyses/tree_engine.cpp
)
target_link_libraries(hyphy3_emitters PUBLIC hyphy3_core)
# The nanobind module links this static lib into a MODULE: it needs PIC.
set_target_properties(hyphy3_emitters PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Test targets
if(HYPHY3_BUILD_TESTS)
    enable_testing()
    # Lets tests locate benchmarks/ and tests/data/ regardless of working directory
    add_compile_definitions(HYPHY3_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    # R4: three merged doctest binaries replace ~29 per-file executables.
    # tests/bin_*.cpp #include the individual test .cpps; the first included
    # file's DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN supplies main() and doctest
    # registers every TEST_CASE. ctest selects per-file cases with doctest's
    # -sf (source-file) filter, so the ctest name list is unchanged.
    #
    # All test targets link hyphy3_emitters (not just hyphy3_core): R3 moved
    # the analysis engine and emitter bodies into that static lib, and the
    # emitters PUBLIC-link propagates core. Static-lib semantics mean unneeded
    # objects are never pulled in.
    #
    # Five tests are standalone programs — own main(), plain asserts, no
    # doctest TEST_CASEs — so they cannot be textually merged and keep their
    # own executables: test_cpp_autograd, test_dfel, test_mss,
    # test_joint_align, test_characterize.
    add_executable(hyphy_test_core_bin tests/bin_core.cpp)
    target_link_libraries(hyphy_test_core_bin PRIVATE hyphy3_emitters hyphy3_report)

    add_executable(hyphy_test_optimizer_bin
        tests/bin_optimizer.cpp
        src/analyses/expcm_optimizer.cpp
        src/analyses/persite_optimizer.cpp)
    target_link_libraries(hyphy_test_optimizer_bin PRIVATE hyphy3_emitters)

    add_executable(hyphy_test_analyses_bin tests/bin_analyses.cpp)
    target_link_libraries(hyphy_test_analyses_bin PRIVATE hyphy3_emitters)

    function(hyphy3_add_standalone target name src)
        add_executable(${target} tests/${src})
        target_link_libraries(${target} PRIVATE hyphy3_emitters)
        add_test(NAME ${name}
            COMMAND ${target}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endfunction()

    hyphy3_add_standalone(hyphy_cpp_autograd_test  CppAutogradTest    test_cpp_autograd.cpp)
    hyphy3_add_standalone(hyphy_dfel_test          DirectionalFELTest test_dfel.cpp)
    hyphy3_add_standalone(hyphy_mss_test           MSSTest            test_mss.cpp)
    hyphy3_add_standalone(hyphy_joint_align_test   JointAlignTest     test_joint_align.cpp)
    hyphy3_add_standalone(hyphy_characterize_test  CharacterizeTest   test_characterize.cpp)

    function(hyphy3_add_doctest name bin src)
        add_test(NAME ${name}
            COMMAND ${bin} -sf=*${src}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endfunction()

    hyphy3_add_doctest(CoreTests                hyphy_test_core_bin     test_core.cpp)
    hyphy3_add_doctest(TreeEngineTest           hyphy_test_core_bin     test_tree_engine.cpp)
    hyphy3_add_doctest(ReportTest               hyphy_test_core_bin     test_report.cpp)
    hyphy3_add_doctest(ExpcmTest                hyphy_test_optimizer_bin test_expcm.cpp)
    hyphy3_add_doctest(ExpcmOptimizerTest       hyphy_test_optimizer_bin test_expcm_optimizer.cpp)
    hyphy3_add_doctest(GradientValidationTest   hyphy_test_optimizer_bin test_gradient_validation.cpp)
    hyphy3_add_doctest(OptimizerQuickTest       hyphy_test_optimizer_bin test_optimizer_quick.cpp)
    hyphy3_add_doctest(Phase4PersiteTest        hyphy_test_optimizer_bin test_phase4_persite.cpp)
    hyphy3_add_doctest(Phase4OptimizerTest      hyphy_test_optimizer_bin test_phase4_optimizer.cpp)
    hyphy3_add_doctest(Phase4ValidationTest     hyphy_test_optimizer_bin test_phase4_validation.cpp)
    hyphy3_add_doctest(FELTests                 hyphy_test_analyses_bin  test_fel.cpp)
    hyphy3_add_doctest(CD2ParityTest            hyphy_test_analyses_bin  test_fel_cd2_parity.cpp)
    hyphy3_add_doctest(MultiDatasetParityTest   hyphy_test_analyses_bin  test_fel_multi_dataset.cpp)
    hyphy3_add_doctest(MEMEParityTest           hyphy_test_analyses_bin  test_meme.cpp)
    hyphy3_add_doctest(BUSTEDTest               hyphy_test_analyses_bin  test_busted.cpp)
    hyphy3_add_doctest(BUSTEDMHTest             hyphy_test_analyses_bin  test_busted_mh.cpp)
    hyphy3_add_doctest(ABSRELTest               hyphy_test_analyses_bin  test_absrel.cpp)
    hyphy3_add_doctest(RELAXTest                hyphy_test_analyses_bin  test_relax.cpp)
    hyphy3_add_doctest(ContrastFELTest          hyphy_test_analyses_bin  test_contrast_fel.cpp)
    hyphy3_add_doctest(FUBARTest                hyphy_test_analyses_bin  test_fubar.cpp)
    hyphy3_add_doctest(PRIMETest                hyphy_test_analyses_bin  test_prime.cpp)
    hyphy3_add_doctest(GARDTest                 hyphy_test_analyses_bin  test_gard.cpp)
    hyphy3_add_doctest(SLACTest                 hyphy_test_analyses_bin  test_slac.cpp)
    hyphy3_add_doctest(OhmicTest                hyphy_test_analyses_bin  test_ohmic.cpp)

    add_test(NAME HTMLReportEndToEndTest
        COMMAND ${CMAKE_COMMAND}
            -DHYPHY3=$<TARGET_FILE:hyphy3>
            -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/data
            -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/html_report_test
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_html_report.cmake
    )

    # Slow tests (a minute or more each on CI) carry the "deep" label: CI runs them nightly,
    # on manual dispatch and on PRs labelled full-ci; every other run uses `ctest -LE deep`.
    # Label any new test that takes a minute or more. The Python tests (PythonAutogradTest,
    # JSONSchemaTest) only exist when nanobind is installed, which only the deep tier does.
    set_tests_properties(
        MultiDatasetParityTest MEMEParityTest BUSTEDTest BUSTEDMHTest ABSRELTest
        RELAXTest ContrastFELTest PRIMETest MSSTest JointAlignTest
        PROPERTIES LABELS deep
    )

    # Viewer unit tests (node:test, no packages). Node expands the glob itself; a bare
    # directory argument is resolved as a module by Node 22 and fails.
    find_program(NODE_EXECUTABLE node)
    if(NODE_EXECUTABLE)
        add_test(NAME ViewerUnitTests
            COMMAND ${NODE_EXECUTABLE} --test ${CMAKE_CURRENT_SOURCE_DIR}/viz/fel-viewer/tests/*.test.mjs)
        add_test(NAME OhmicViewerUnitTests
            COMMAND ${NODE_EXECUTABLE} --test ${CMAKE_CURRENT_SOURCE_DIR}/viz/ohmic-viewer/tests/*.test.mjs)
    endif()
endif()


# Unified HyPhy 3 CLI Driver (hyphy3 fel ..., hyphy3 meme ..., hyphy3 busted ..., hyphy3 absrel ..., hyphy3 relax ..., hyphy3 fubar ..., hyphy3 prime ..., hyphy3 gard ..., hyphy3 slac ..., hyphy3 dfel ..., hyphy3 mss ..., hyphy3 ohmic ..., hyphy3 joint-align ..., hyphy3 expcm ...)
add_executable(hyphy3
    src/apps/hyphy3.cpp
    src/apps/hyphy_fel.cpp
    src/apps/hyphy_contrast_fel.cpp
    src/apps/hyphy_meme.cpp
    src/apps/hyphy_busted.cpp
    src/apps/hyphy_absrel.cpp
    src/apps/hyphy_relax.cpp
    src/apps/hyphy_fubar.cpp
    src/apps/hyphy_prime.cpp
    src/apps/hyphy_gard.cpp
    src/apps/hyphy_slac.cpp
    src/apps/hyphy_dfel.cpp
    src/apps/hyphy_mss.cpp
    src/apps/hyphy_ohmic.cpp
    src/apps/hyphy_tree.cpp
    src/apps/hyphy_joint_align.cpp
    src/apps/hyphy_characterize.cpp
    src/apps/hyphy_expcm.cpp
    src/apps/hyphy_report.cpp
    src/analyses/expcm_optimizer.cpp
    src/analyses/persite_optimizer.cpp
)
target_compile_definitions(hyphy3 PRIVATE HYPHY3_COMBINED_DRIVER)
target_link_libraries(hyphy3 PRIVATE hyphy3_report)

# Regression gates G1 (golden CLI output vs committed snapshots, plus v3
# schema validation of the raw output) and G3 (driver dispatch / argv[0]
# alias surface). Only python3 is required — no package dependencies.
if(HYPHY3_BUILD_TESTS)
    find_program(HYPHY3_PYTHON3 NAMES python3 python)
    if(HYPHY3_PYTHON3)
        add_test(NAME GoldenOutputTest
            COMMAND "${HYPHY3_PYTHON3}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/golden_diff.py"
                --bin "$<TARGET_FILE:hyphy3>"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        add_test(NAME GoldenOutputDeepTest
            COMMAND "${HYPHY3_PYTHON3}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/golden_diff.py"
                --bin "$<TARGET_FILE:hyphy3>" --tier slow
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        set_tests_properties(GoldenOutputDeepTest PROPERTIES LABELS deep)
        add_test(NAME CLISurfaceTest
            COMMAND "${HYPHY3_PYTHON3}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli_surface.py"
                "$<TARGET_FILE:hyphy3>"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
    endif()
endif()

# Standalone CLI Applications
# With HYPHY3_STANDALONE_SYMLINKS these targets are not built at all: every
# standalone main() is just a try/catch wrapper around the same run_*() entry
# point the hyphy3 driver exposes, and the driver dispatches on argv[0]. The
# hyphy3-* names are then created as symlinks at install time (below).
if(NOT HYPHY3_STANDALONE_SYMLINKS)
add_executable(hyphy3-fel src/apps/hyphy_fel.cpp)
target_link_libraries(hyphy3-fel PRIVATE hyphy3_report)

add_executable(hyphy3-contrast-fel src/apps/hyphy_contrast_fel.cpp)
target_link_libraries(hyphy3-contrast-fel PRIVATE hyphy3_report)

add_executable(hyphy3-meme src/apps/hyphy_meme.cpp)
target_link_libraries(hyphy3-meme PRIVATE hyphy3_report)

add_executable(hyphy3-busted src/apps/hyphy_busted.cpp)
target_link_libraries(hyphy3-busted PRIVATE hyphy3_report)

add_executable(hyphy3-absrel src/apps/hyphy_absrel.cpp)
target_link_libraries(hyphy3-absrel PRIVATE hyphy3_report)

add_executable(hyphy3-relax src/apps/hyphy_relax.cpp)
target_link_libraries(hyphy3-relax PRIVATE hyphy3_report)

add_executable(hyphy3-fubar src/apps/hyphy_fubar.cpp)
target_link_libraries(hyphy3-fubar PRIVATE hyphy3_report)

add_executable(hyphy3-prime src/apps/hyphy_prime.cpp)
target_link_libraries(hyphy3-prime PRIVATE hyphy3_report)

add_executable(hyphy3-gard src/apps/hyphy_gard.cpp)
target_link_libraries(hyphy3-gard PRIVATE hyphy3_report)

add_executable(hyphy3-slac src/apps/hyphy_slac.cpp)
target_link_libraries(hyphy3-slac PRIVATE hyphy3_report)

add_executable(hyphy3-dfel src/apps/hyphy_dfel.cpp)
target_link_libraries(hyphy3-dfel PRIVATE hyphy3_report)

add_executable(hyphy3-mss src/apps/hyphy_mss.cpp)
target_link_libraries(hyphy3-mss PRIVATE hyphy3_report)

add_executable(hyphy3-ohmic src/apps/hyphy_ohmic.cpp)
target_link_libraries(hyphy3-ohmic PRIVATE hyphy3_report)

add_executable(hyphy3-tree src/apps/hyphy_tree.cpp)
target_link_libraries(hyphy3-tree PRIVATE hyphy3_core)

add_executable(hyphy3-joint-align src/apps/hyphy_joint_align.cpp)
target_link_libraries(hyphy3-joint-align PRIVATE hyphy3_core)

add_executable(hyphy3-characterize src/apps/hyphy_characterize.cpp)
target_link_libraries(hyphy3-characterize PRIVATE hyphy3_core)

add_executable(hyphy3-walrus src/apps/hyphy_characterize.cpp)
target_link_libraries(hyphy3-walrus PRIVATE hyphy3_core)

add_executable(hyphy3-expcm
    src/apps/hyphy_expcm.cpp
    src/apps/hyphy_expcm_main.cpp
    src/analyses/expcm_optimizer.cpp
    src/analyses/persite_optimizer.cpp
)
target_link_libraries(hyphy3-expcm PRIVATE hyphy3_core)

# Backward-compatible aliases
add_executable(hyphy-fel ALIAS hyphy3-fel)
add_executable(hyphy-contrast-fel ALIAS hyphy3-contrast-fel)
add_executable(hyphy-meme ALIAS hyphy3-meme)
add_executable(hyphy-busted ALIAS hyphy3-busted)
add_executable(hyphy-absrel ALIAS hyphy3-absrel)
add_executable(hyphy-relax ALIAS hyphy3-relax)
add_executable(hyphy-fubar ALIAS hyphy3-fubar)
add_executable(hyphy-prime ALIAS hyphy3-prime)
add_executable(hyphy-gard ALIAS hyphy3-gard)
add_executable(hyphy-slac ALIAS hyphy3-slac)
add_executable(hyphy-dfel ALIAS hyphy3-dfel)
add_executable(hyphy-mss ALIAS hyphy3-mss)
add_executable(hyphy-ohmic ALIAS hyphy3-ohmic)
add_executable(hyphy-tree ALIAS hyphy3-tree)
add_executable(hyphy-joint-align ALIAS hyphy3-joint-align)
add_executable(hyphy-characterize ALIAS hyphy3-characterize)
add_executable(hyphy-walrus ALIAS hyphy3-walrus)
add_executable(hyphy-expcm ALIAS hyphy3-expcm)
endif()

include(GNUInstallDirs)
if(HYPHY3_STANDALONE_SYMLINKS)
    set(_hyphy3_cli_targets hyphy3)
else()
    set(_hyphy3_cli_targets
        hyphy3 hyphy3-characterize hyphy3-walrus hyphy3-busted hyphy3-fel
        hyphy3-absrel hyphy3-meme hyphy3-relax hyphy3-slac hyphy3-fubar
        hyphy3-gard hyphy3-contrast-fel hyphy3-dfel hyphy3-prime hyphy3-mss
        hyphy3-ohmic hyphy3-tree hyphy3-joint-align hyphy3-expcm
    )
    if(TARGET hyphy3-report)
        list(APPEND _hyphy3_cli_targets hyphy3-report)
    endif()
endif()
install(TARGETS ${_hyphy3_cli_targets}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(HYPHY3_STANDALONE_SYMLINKS)
    # Same names the standalone executables would have installed. Created
    # relative so the links keep working if the install tree is relocated
    # (conda prefix relocation, moved DESTDIR staging, etc.). Declared after
    # install(TARGETS) so the driver and bindir already exist; install()
    # commands run in declaration order.
    foreach(_alias IN ITEMS
        fel contrast-fel meme busted absrel relax fubar prime gard slac dfel
        mss ohmic tree joint-align characterize walrus expcm
    )
        install(CODE "
            file(MAKE_DIRECTORY
                \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}\")
            execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
                hyphy3
                \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/hyphy3-${_alias}\")
        ")
    endforeach()
endif()

# WebAssembly (Emscripten) One-Button WALRUS Engine Target
if(EMSCRIPTEN)
    add_executable(hyphy3_wasm src/apps/hyphy_wasm.cpp)
    target_link_libraries(hyphy3_wasm PRIVATE hyphy3_core hyphy3_emitters)
    set_target_properties(hyphy3_wasm PROPERTIES
        OUTPUT_NAME "hyphy3_walrus"
        SUFFIX ".js"
        LINK_FLAGS "-O3 -msimd128 -fexceptions -sMODULARIZE=1 -sEXPORT_NAME=createWalrusModule -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=67108864 -sMAXIMUM_MEMORY=2147483648 -sSTACK_SIZE=8388608 -sENVIRONMENT=web,worker,node -sFILESYSTEM=1 -sEXPORTED_FUNCTIONS=['_walrus_stage0_run','_walrus_stage1_run','_walrus_stage2_run','_walrus_stage3_run','_walrus_ping_run','_walrus_get_last_json','_walrus_get_last_html','_walrus_get_last_error','_walrus_render_html_from_json','_malloc','_free'] -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','UTF8ToString','stringToNewUTF8','FS']"
    )
else()
    add_custom_target(wasm
        COMMAND /bin/bash "${CMAKE_SOURCE_DIR}/scripts/build_wasm.sh"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Cross-compiling HyPhy-3 WALRUS engine to WebAssembly (hyphy3_walrus.js + hyphy3_walrus.wasm)"
        USES_TERMINAL
    )
endif()

# Python & nanobind bindings
if(NOT EMSCRIPTEN)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module)
if(Python_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR
        RESULT_VARIABLE nanobind_STATUS
    )
    if(nanobind_STATUS EQUAL 0 AND EXISTS "${nanobind_DIR}")
        list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
        find_package(nanobind CONFIG)
        if(nanobind_FOUND)
            message(STATUS "Building Python bindings with nanobind")
            nanobind_add_module(_hyphy3 NB_STATIC
                src/python/bindings.cpp
                src/python/bindings_core.cpp
                src/python/bindings_fel.cpp
                src/python/bindings_meme.cpp
                src/python/bindings_busted.cpp
                src/python/bindings_absrel.cpp
                src/python/bindings_relax.cpp
                src/python/bindings_fubar.cpp
                src/python/bindings_prime.cpp
                src/python/bindings_gard.cpp
                src/python/bindings_slac.cpp
                src/python/bindings_dfel.cpp
                src/python/bindings_ohmic.cpp
            )
            target_link_libraries(_hyphy3 PRIVATE hyphy3_core hyphy3_emitters)
            set_target_properties(_hyphy3 PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python"
            )
            # Wheel/sdist installs (scikit-build-core): extension lands at the
            # wheel root so `import _hyphy3` works after `pip install .`.
            if(DEFINED SKBUILD)
                install(TARGETS _hyphy3 LIBRARY DESTINATION .)
            else()
                install(TARGETS _hyphy3 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
            endif()

            if(HYPHY3_BUILD_TESTS)
                add_test(NAME PythonAutogradTest
                    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_python_autograd.py"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                )
                set_tests_properties(PythonAutogradTest PROPERTIES
                    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python;KMP_DUPLICATE_LIB_OK=TRUE"
                )

                add_test(NAME JSONSchemaTest
                    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_json_schema.py"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                )
                set_tests_properties(JSONSchemaTest PROPERTIES
                    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python;KMP_DUPLICATE_LIB_OK=TRUE"
                )
            endif()
        endif()
    endif()
endif()
endif()


# Consumer wiring ----------------------------------------------------------------
# Every executable links hyphy3_emitters (out-of-line emitter definitions) and,
# when HYPHY3_PCH is on, reuses the shared precompiled header compiled by
# hyphy3_pch instead of re-parsing nlohmann/json + Eigen + the ubiquitous core
# headers. Disable PCH with -DHYPHY3_PCH=OFF if it causes toolchain issues.
if(HYPHY3_PCH)
    add_library(hyphy3_pch STATIC src/pch.cpp)
    target_link_libraries(hyphy3_pch PUBLIC hyphy3_core)
    target_precompile_headers(hyphy3_pch PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pch.hpp>")

    # hyphy3_emitters gets its OWN PCH of the same stable stack rather than
    # REUSE_FROM hyphy3_pch: emitters compile with -fPIC and hyphy3_pch does
    # not, and GCC silently ignores a PCH built with mismatched flags. The
    # extra PCH costs one parse of the stack and speeds all ~14 emitter TUs.
    target_precompile_headers(hyphy3_emitters
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/pch.hpp")
endif()

get_property(_hyphy3_all_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
             PROPERTY BUILDSYSTEM_TARGETS)
foreach(_tgt IN LISTS _hyphy3_all_targets)
    if(_tgt STREQUAL "hyphy3_pch" OR _tgt STREQUAL "hyphy3_emitters")
        continue()
    endif()
    get_target_property(_tgt_type ${_tgt} TYPE)
    if(_tgt_type STREQUAL "EXECUTABLE")
        target_link_libraries(${_tgt} PRIVATE hyphy3_emitters)
        if(HYPHY3_PCH)
            target_link_libraries(${_tgt} PRIVATE hyphy3_pch)
            target_precompile_headers(${_tgt} REUSE_FROM hyphy3_pch)
        endif()
    endif()
endforeach()
