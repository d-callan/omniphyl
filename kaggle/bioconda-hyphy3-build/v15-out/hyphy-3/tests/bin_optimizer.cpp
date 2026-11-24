// R4: merged doctest binary for optimizer/phase4 tests.
// The target also compiles src/analyses/expcm_optimizer.cpp and
// persite_optimizer.cpp once, shared by all cases below.
// ctest selects per-file cases with: -sf=*test_<name>.cpp
//
// Merging is textual, so every included file shares one scope. Six of the
// files below declare the same file-scope constants (ALIGNMENT_FILE,
// TREE_FILE, PREFS_FILE, and the BETA/OMEGA/KAPPA reference values — some
// with DIFFERENT values, so they cannot be hoisted into a shared header).
// Each include is therefore wrapped in #define renames that give every
// shared name a per-file suffix; the test sources stay unchanged.

#define ALIGNMENT_FILE   XPC_ALIGNMENT_FILE
#define TREE_FILE        XPC_TREE_FILE
#define PREFS_FILE       XPC_PREFS_FILE
#define REFERENCE_FILE   XPC_REFERENCE_FILE
#define BETA_REF         XPC_BETA_REF
#define OMEGA_REF        XPC_OMEGA_REF
#define KAPPA_REF        XPC_KAPPA_REF
#define LNL_REF          XPC_LNL_REF
#define BETA_PHASE3      XPC_BETA_PHASE3
#define OMEGA_PHASE3     XPC_OMEGA_PHASE3
#define KAPPA_PHASE3     XPC_KAPPA_PHASE3
#include "test_expcm.cpp"
#undef ALIGNMENT_FILE
#undef TREE_FILE
#undef PREFS_FILE
#undef REFERENCE_FILE
#undef BETA_REF
#undef OMEGA_REF
#undef KAPPA_REF
#undef LNL_REF
#undef BETA_PHASE3
#undef OMEGA_PHASE3
#undef KAPPA_PHASE3

#define ALIGNMENT_FILE   XOP_ALIGNMENT_FILE
#define TREE_FILE        XOP_TREE_FILE
#define PREFS_FILE       XOP_PREFS_FILE
#define REFERENCE_FILE   XOP_REFERENCE_FILE
#define BETA_REF         XOP_BETA_REF
#define OMEGA_REF        XOP_OMEGA_REF
#define KAPPA_REF        XOP_KAPPA_REF
#define LNL_REF          XOP_LNL_REF
#define BETA_PHASE3      XOP_BETA_PHASE3
#define OMEGA_PHASE3     XOP_OMEGA_PHASE3
#define KAPPA_PHASE3     XOP_KAPPA_PHASE3
#include "test_expcm_optimizer.cpp"
#undef ALIGNMENT_FILE
#undef TREE_FILE
#undef PREFS_FILE
#undef REFERENCE_FILE
#undef BETA_REF
#undef OMEGA_REF
#undef KAPPA_REF
#undef LNL_REF
#undef BETA_PHASE3
#undef OMEGA_PHASE3
#undef KAPPA_PHASE3

#define ALIGNMENT_FILE   GRV_ALIGNMENT_FILE
#define TREE_FILE        GRV_TREE_FILE
#define PREFS_FILE       GRV_PREFS_FILE
#define REFERENCE_FILE   GRV_REFERENCE_FILE
#define BETA_REF         GRV_BETA_REF
#define OMEGA_REF        GRV_OMEGA_REF
#define KAPPA_REF        GRV_KAPPA_REF
#define LNL_REF          GRV_LNL_REF
#define BETA_PHASE3      GRV_BETA_PHASE3
#define OMEGA_PHASE3     GRV_OMEGA_PHASE3
#define KAPPA_PHASE3     GRV_KAPPA_PHASE3
#include "test_gradient_validation.cpp"
#undef ALIGNMENT_FILE
#undef TREE_FILE
#undef PREFS_FILE
#undef REFERENCE_FILE
#undef BETA_REF
#undef OMEGA_REF
#undef KAPPA_REF
#undef LNL_REF
#undef BETA_PHASE3
#undef OMEGA_PHASE3
#undef KAPPA_PHASE3

#define ALIGNMENT_FILE   OQK_ALIGNMENT_FILE
#define TREE_FILE        OQK_TREE_FILE
#define PREFS_FILE       OQK_PREFS_FILE
#define REFERENCE_FILE   OQK_REFERENCE_FILE
#define BETA_REF         OQK_BETA_REF
#define OMEGA_REF        OQK_OMEGA_REF
#define KAPPA_REF        OQK_KAPPA_REF
#define LNL_REF          OQK_LNL_REF
#define BETA_PHASE3      OQK_BETA_PHASE3
#define OMEGA_PHASE3     OQK_OMEGA_PHASE3
#define KAPPA_PHASE3     OQK_KAPPA_PHASE3
#include "test_optimizer_quick.cpp"
#undef ALIGNMENT_FILE
#undef TREE_FILE
#undef PREFS_FILE
#undef REFERENCE_FILE
#undef BETA_REF
#undef OMEGA_REF
#undef KAPPA_REF
#undef LNL_REF
#undef BETA_PHASE3
#undef OMEGA_PHASE3
#undef KAPPA_PHASE3

#define ALIGNMENT_FILE   P4P_ALIGNMENT_FILE
#define TREE_FILE        P4P_TREE_FILE
#define PREFS_FILE       P4P_PREFS_FILE
#define REFERENCE_FILE   P4P_REFERENCE_FILE
#define BETA_REF         P4P_BETA_REF
#define OMEGA_REF        P4P_OMEGA_REF
#define KAPPA_REF        P4P_KAPPA_REF
#define LNL_REF          P4P_LNL_REF
#define BETA_PHASE3      P4P_BETA_PHASE3
#define OMEGA_PHASE3     P4P_OMEGA_PHASE3
#define KAPPA_PHASE3     P4P_KAPPA_PHASE3
#include "test_phase4_persite.cpp"
#undef ALIGNMENT_FILE
#undef TREE_FILE
#undef PREFS_FILE
#undef REFERENCE_FILE
#undef BETA_REF
#undef OMEGA_REF
#undef KAPPA_REF
#undef LNL_REF
#undef BETA_PHASE3
#undef OMEGA_PHASE3
#undef KAPPA_PHASE3

#define ALIGNMENT_FILE   P4O_ALIGNMENT_FILE
#define TREE_FILE        P4O_TREE_FILE
#define PREFS_FILE       P4O_PREFS_FILE
#define REFERENCE_FILE   P4O_REFERENCE_FILE
#define BETA_REF         P4O_BETA_REF
#define OMEGA_REF        P4O_OMEGA_REF
#define KAPPA_REF        P4O_KAPPA_REF
#define LNL_REF          P4O_LNL_REF
#define BETA_PHASE3      P4O_BETA_PHASE3
#define OMEGA_PHASE3     P4O_OMEGA_PHASE3
#define KAPPA_PHASE3     P4O_KAPPA_PHASE3
#include "test_phase4_optimizer.cpp"
#undef ALIGNMENT_FILE
#undef TREE_FILE
#undef PREFS_FILE
#undef REFERENCE_FILE
#undef BETA_REF
#undef OMEGA_REF
#undef KAPPA_REF
#undef LNL_REF
#undef BETA_PHASE3
#undef OMEGA_PHASE3
#undef KAPPA_PHASE3

#define ALIGNMENT_FILE   P4V_ALIGNMENT_FILE
#define TREE_FILE        P4V_TREE_FILE
#define PREFS_FILE       P4V_PREFS_FILE
#define REFERENCE_FILE   P4V_REFERENCE_FILE
#define BETA_REF         P4V_BETA_REF
#define OMEGA_REF        P4V_OMEGA_REF
#define KAPPA_REF        P4V_KAPPA_REF
#define LNL_REF          P4V_LNL_REF
#define BETA_PHASE3      P4V_BETA_PHASE3
#define OMEGA_PHASE3     P4V_OMEGA_PHASE3
#define KAPPA_PHASE3     P4V_KAPPA_PHASE3
#include "test_phase4_validation.cpp"
#undef ALIGNMENT_FILE
#undef TREE_FILE
#undef PREFS_FILE
#undef REFERENCE_FILE
#undef BETA_REF
#undef OMEGA_REF
#undef KAPPA_REF
#undef LNL_REF
#undef BETA_PHASE3
#undef OMEGA_PHASE3
#undef KAPPA_PHASE3
