// R4: merged doctest binary for per-analysis tests.
// ctest selects per-file cases with: -sf=*test_<name>.cpp
//
// Merging is textual, so every included file shares one scope. Several files
// define identically-named file-scope helpers — resolve_test_path (6 files)
// and compute_ranks/pearson_correlation/spearman_correlation (3 files) —
// which collide when merged. Each include is therefore wrapped in #define
// renames that give those names a per-file suffix; the test sources stay
// unchanged. A uniform wrapper around every include keeps the rule simple:
// any shared file-scope name gets a private spelling per file.
//
// test_mss.cpp, test_dfel.cpp, test_joint_align.cpp and test_characterize.cpp
// are NOT merged here: they are standalone programs with their own main()
// and plain asserts (no TEST_CASEs). They keep their own executables in
// CMakeLists.txt.

#define resolve_test_path      FEL_resolve_test_path
#define compute_ranks          FEL_compute_ranks
#define pearson_correlation    FEL_pearson_correlation
#define spearman_correlation   FEL_spearman_correlation
#include "test_fel.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation

#define resolve_test_path      CD2_resolve_test_path
#define compute_ranks          CD2_compute_ranks
#define pearson_correlation    CD2_pearson_correlation
#define spearman_correlation   CD2_spearman_correlation
#include "test_fel_cd2_parity.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation

#define resolve_test_path      MDS_resolve_test_path
#define compute_ranks          MDS_compute_ranks
#define pearson_correlation    MDS_pearson_correlation
#define spearman_correlation   MDS_spearman_correlation
#include "test_fel_multi_dataset.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation

#define resolve_test_path      MEM_resolve_test_path
#define compute_ranks          MEM_compute_ranks
#define pearson_correlation    MEM_pearson_correlation
#define spearman_correlation   MEM_spearman_correlation
#include "test_meme.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation

#define resolve_test_path      BUS_resolve_test_path
#define compute_ranks          BUS_compute_ranks
#define pearson_correlation    BUS_pearson_correlation
#define spearman_correlation   BUS_spearman_correlation
#include "test_busted.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation

#define resolve_test_path      BMH_resolve_test_path
#define compute_ranks          BMH_compute_ranks
#define pearson_correlation    BMH_pearson_correlation
#define spearman_correlation   BMH_spearman_correlation
#include "test_busted_mh.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation

#define resolve_test_path      ABS_resolve_test_path
#define compute_ranks          ABS_compute_ranks
#define pearson_correlation    ABS_pearson_correlation
#define spearman_correlation   ABS_spearman_correlation
#include "test_absrel.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation

#define resolve_test_path      RLX_resolve_test_path
#define compute_ranks          RLX_compute_ranks
#define pearson_correlation    RLX_pearson_correlation
#define spearman_correlation   RLX_spearman_correlation
#include "test_relax.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation

#define resolve_test_path      CFL_resolve_test_path
#define compute_ranks          CFL_compute_ranks
#define pearson_correlation    CFL_pearson_correlation
#define spearman_correlation   CFL_spearman_correlation
#include "test_contrast_fel.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation

#define resolve_test_path      FUB_resolve_test_path
#define compute_ranks          FUB_compute_ranks
#define pearson_correlation    FUB_pearson_correlation
#define spearman_correlation   FUB_spearman_correlation
#include "test_fubar.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation

#define resolve_test_path      PRM_resolve_test_path
#define compute_ranks          PRM_compute_ranks
#define pearson_correlation    PRM_pearson_correlation
#define spearman_correlation   PRM_spearman_correlation
#include "test_prime.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation

#define resolve_test_path      GRD_resolve_test_path
#define compute_ranks          GRD_compute_ranks
#define pearson_correlation    GRD_pearson_correlation
#define spearman_correlation   GRD_spearman_correlation
#include "test_gard.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation

#define resolve_test_path      SLC_resolve_test_path
#define compute_ranks          SLC_compute_ranks
#define pearson_correlation    SLC_pearson_correlation
#define spearman_correlation   SLC_spearman_correlation
#include "test_slac.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation

#define resolve_test_path      OHM_resolve_test_path
#define compute_ranks          OHM_compute_ranks
#define pearson_correlation    OHM_pearson_correlation
#define spearman_correlation   OHM_spearman_correlation
#include "test_ohmic.cpp"
#undef resolve_test_path
#undef compute_ranks
#undef pearson_correlation
#undef spearman_correlation
