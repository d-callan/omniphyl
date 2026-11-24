// R4: merged doctest binary for core/non-analysis tests.
// Each test .cpp is included verbatim; the first one's
// DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN supplies main(), later identical
// macro definitions are benign and doctest.h include guards dedupe.
// ctest selects per-file cases with: -sf=*test_<name>.cpp
//
// test_cpp_autograd.cpp is NOT merged here: it is a standalone program with
// its own main() and plain asserts (no TEST_CASEs). It keeps its own
// executable in CMakeLists.txt.

#include "test_core.cpp"
#include "test_tree_engine.cpp"
#include "test_report.cpp"
