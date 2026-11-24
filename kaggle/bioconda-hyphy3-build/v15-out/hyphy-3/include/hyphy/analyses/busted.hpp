#pragma once

#include "hyphy/core/types.hpp"
#include "hyphy/core/genetic_code.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/rate_matrix.hpp"
#include "hyphy/core/likelihood.hpp"
#include "hyphy/opt/optimizer.hpp"
#include "hyphy/core/json_output.hpp"
#include "hyphy/core/stats.hpp"
#include "hyphy/analyses/busted_types.hpp"
#include "hyphy/analyses/mss.hpp"
#include "hyphy/analyses/fel.hpp"
#include "hyphy/core/progress_bar.hpp"
#include "hyphy/opt/nelder_mead.hpp"
#include "hyphy/opt/squarem.hpp"
#include "nlohmann/json.hpp"
#include "hyphy/core/provenance.hpp"

#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace hyphy::analyses {

using namespace hyphy::core;
using namespace hyphy::opt;

class BUSTEDAnalyzer {
public:
    Tree tree;
    Alignment aln;
    std::shared_ptr<const GeneticCode> code;
    mutable MG94Parameters base_params;
    mutable std::vector<Scalar> syn_pair_rates_storage; // Optional S*S flat array for BUSTED-MSS

    // Relative branch lengths tau_b (normalized from MG94 baseline)
    mutable std::vector<Scalar> branch_tau;
    mutable std::vector<std::array<Scalar, 3>> bridge_tau;
    mutable std::vector<bool> is_test_branch;
    // 0 = Test, 1 = Background/Comparison, 2 = Unclassified/Nuisance
    mutable std::vector<int> branch_partition;
    std::vector<size_t> leaf_to_taxon;
    mutable size_t num_test_branches = 0;
    mutable size_t num_background_branches = 0;
    mutable size_t num_unclassified_branches = 0;

    // Precomputed off-diagonal saturation replacement kernel P_sink(i, j) = pi_j / (1 - pi_i) for i != j, 0 for i == j
    Matrix P_sink;
    Vector inv_one_minus_pi;

    // GTR baseline results
    bool has_gtr_fit = false;
    Scalar gtr_log_l = 0.0;
    Scalar gtr_aicc = 0.0;
    GTRParameters gtr_rates;
    std::unordered_map<std::string, Scalar> gtr_branch_lengths;

    // Global MG94 baseline results
    Scalar mg94_log_l = 0.0;
    Scalar mg94_omega = 1.0;
    Scalar mg94_background_omega = -1.0;
    Scalar mg94_tree_length = 0.0;
    mutable size_t num_mss_free_params = 0;

    BUSTEDAnalyzer(
        Tree t,
        Alignment a,
        MG94Parameters base_p = MG94Parameters{}
    );

    void select_branch_partitions(
        const std::string& test_spec,
        const std::string& comp_spec = "",
        bool auto_detect_tags = false
    ) const;

    void select_test_branches(const std::string& spec) const;

    void build_sink_kernel();

    // O(S) evaluation of V^T * P_sink * D using the rank-1 off-diagonal identity
    inline Scalar eval_sink_bilinear(const Vector& V, const Vector& D) const {
        Scalar pi_dot_D = aln.codon_frequencies_f3x4.dot(D);
        int S = static_cast<int>(V.size());
        Scalar acc = 0.0;
        for (int i = 0; i < S; ++i) {
            Scalar p_sink_D_i = (pi_dot_D - aln.codon_frequencies_f3x4(i) * D(i)) * inv_one_minus_pi(i);
            acc += V(i) * std::max(0.0, p_sink_D_i);
        }
        return acc;
    }

    static BUSTEDAnalyzer create_and_fit(
        Tree raw_tree,
        Alignment aln,
        const std::vector<bool>* is_test_branch_ptr = nullptr
    );

    void prepare_baseline_lengths();

    Scalar bridge_resolution_tau(
        size_t b_idx,
        int r,
        Scalar s,
        const std::vector<Scalar>* custom_branch_lengths = nullptr
    ) const;

    std::vector<std::array<Matrix, 3>> build_one_set_bridge_mix(
        Scalar s,
        Scalar alpha_m,
        const std::vector<MG94Matrix>& mg,
        const MG94Matrix& mg_bg,
        const std::vector<Scalar>& weights,
        Scalar eps_sink,
        Scalar bio_scale,
        const std::vector<Scalar>* custom_branch_lengths = nullptr
    ) const;

    std::vector<std::array<Matrix, 3>> build_two_set_bridge_mix(
        Scalar s,
        Scalar alpha_m,
        const std::vector<MG94Matrix>& mg_t,
        const std::vector<MG94Matrix>& mg_b,
        const std::vector<MG94Matrix>& mg_u,
        const MG94Matrix& mg_unclass,
        const std::vector<Scalar>& weights_test,
        const std::vector<Scalar>& weights_bg,
        const std::vector<Scalar>* weights_unclass,
        bool has_unclass_dist,
        Scalar eps_sink_test,
        Scalar eps_sink_bg,
        Scalar bio_scale_t,
        Scalar bio_scale_b,
        const std::vector<Scalar>* custom_branch_lengths = nullptr
    ) const;

    // Evaluates tree log-likelihood for arbitrary K rate categories, optional SRV (M classes), optional saturation error sink (eps), and custom branch lengths
    Scalar evaluate_log_l(
        Scalar s,
        const std::vector<Scalar>& omegas,
        const std::vector<Scalar>& weights,
        const std::vector<Scalar>& syn_rates = {1.0},
        const std::vector<Scalar>& syn_weights = {1.0},
        const std::vector<Scalar>* custom_branch_lengths = nullptr,
        std::vector<Scalar>* out_pattern_ll = nullptr,
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        Scalar eps_sink = 0.0,
        const MG94Parameters* custom_base_params = nullptr
    ) const;

    // Expectation Step: Computes new mixture weights (and optional saturation sink weight eps_sink) via batched inside-outside GEMM traversal
    std::vector<Scalar> compute_expected_weights(
        Scalar s,
        const std::vector<Scalar>& omegas,
        const std::vector<Scalar>& weights,
        const std::vector<Scalar>& syn_rates = {1.0},
        const std::vector<Scalar>& syn_weights = {1.0},
        const std::vector<Scalar>* custom_branch_lengths = nullptr,
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        Scalar eps_sink = 0.0,
        Scalar* out_new_eps_sink = nullptr,
        Scalar max_eps_sink = 0.01,
        const MG94Parameters* custom_base_params = nullptr
    ) const;

    // Expectation Step: Computes new synonymous rate weights via batched GEMM site posteriors (with log-underflow scaling)
    std::vector<Scalar> compute_expected_syn_weights(
        Scalar s,
        const std::vector<Scalar>& omegas,
        const std::vector<Scalar>& weights,
        const std::vector<Scalar>& syn_rates,
        const std::vector<Scalar>& syn_weights,
        const std::vector<Scalar>* custom_branch_lengths = nullptr,
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        Scalar eps_sink = 0.0,
        const MG94Parameters* custom_base_params = nullptr
    ) const;

    // Evaluates log-likelihood and exact analytical branch gradients d ln L / d t_b
    // under the BUSTED mixture model using batched Inside-Outside GEMM adjoints (with optional SRV and error sink)
    std::pair<Scalar, std::vector<Scalar>> compute_busted_branch_gradients(
        const std::vector<Scalar>& branch_lengths,
        const std::vector<Scalar>& omegas,
        const std::vector<Scalar>& weights,
        const std::vector<Scalar>& syn_rates = {1.0},
        const std::vector<Scalar>& syn_weights = {1.0},
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        Scalar eps_sink = 0.0,
        const MG94Parameters* custom_base_params = nullptr
    ) const;

    // High-performance joint branch length refinement via exact Inside-Outside adjoint gradients and L-BFGS in log-branch-length space
    std::vector<Scalar> refine_branch_lengths(
        const std::vector<Scalar>& init_bl,
        const std::vector<Scalar>& omegas,
        const std::vector<Scalar>& weights,
        const std::vector<Scalar>& syn_rates = {1.0},
        const std::vector<Scalar>& syn_weights = {1.0},
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        int max_iters = 25,
        Scalar eps_sink = 0.0,
        const MG94Parameters* custom_base_params = nullptr
    ) const;

    // Fits BUSTED model for specified number of rate categories K >= 1, optional SRV (M >= 1), and optional saturation error sink
    BUSTEDFit fit_model(
        bool is_constrained,
        size_t K = 3,
        bool refine_branches = true,
        bool srv = false,
        size_t num_syn_rates = 3,
        const std::string& multiple_hits = "None",
        ProgressBar* pbar = nullptr,
        const std::string& model_desc = "",
        bool use_error_sink = false,
        Scalar max_eps_sink = 0.01,
        const BUSTEDFit* warm_start_fit = nullptr
    ) const;

    // Computes branch-site Error Sink Bayes Factors in a single O(patterns * branches) Inside-Outside pass
    void compute_error_sink_diagnostics(
        const BUSTEDFit& fit,
        Scalar bf_threshold,
        std::vector<Scalar>& out_site_max_bf,
        std::vector<BUSTEDErrorEvent>& out_flagged_events
    ) const;

    // Fast 4-Gate Surprisal & Score Automatic Model Selector (BUSTED-Auto)
    // Diagnoses Error Sink (O(1) saturation + spatial adjacency), Multi-Hits (O(t_b) quarantined tangent score),
    // Synonymous Rate Variation (Neyman-Scott overdispersion + SQUAREM probe), and Omega classes K in a single baseline pass.
    BUSTEDAutoDiagnostics diagnose_and_select_model(BUSTEDSettings& settings) const;

    // Step-up model selection across K in {1, ..., max_k} using AICc on the Null model
    std::pair<size_t, std::vector<BUSTEDFit>> select_optimal_k(
        size_t max_k = 3,
        bool refine_branches = true,
        bool srv = false,
        size_t num_syn_rates = 3,
        const std::string& multiple_hits = "None",
        ProgressBar* pbar = nullptr,
        bool use_error_sink = false,
        Scalar max_eps_sink = 0.01
    ) const;

    // Two-set likelihood evaluation for BUSTED with separate Test (partition 0), Background/Comparison (partition 1), and optional Unclassified/Nuisance (partition 2) K-class distributions
    Scalar evaluate_two_set_log_l(
        Scalar s,
        const std::vector<Scalar>& omegas_test,
        const std::vector<Scalar>& weights_test,
        const std::vector<Scalar>& omegas_bg,
        const std::vector<Scalar>& weights_bg,
        const std::vector<Scalar>& syn_rates = {1.0},
        const std::vector<Scalar>& syn_weights = {1.0},
        const std::vector<Scalar>* custom_branch_lengths = nullptr,
        std::vector<Scalar>* out_pattern_ll = nullptr,
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        Scalar eps_sink_test = 0.0,
        Scalar eps_sink_bg = 0.0,
        const std::vector<Scalar>* omegas_unclass = nullptr,
        const std::vector<Scalar>* weights_unclass = nullptr,
        const MG94Parameters* custom_base_params = nullptr
    ) const;

    // Two-set Expectation Step: simultaneously updates Test (partition 0), Background (partition 1), and optional Unclassified (partition 2) mixture weights in one Inside-Outside pass
    std::pair<std::vector<Scalar>, std::vector<Scalar>> compute_expected_two_set_weights(
        Scalar s,
        const std::vector<Scalar>& omegas_test,
        const std::vector<Scalar>& weights_test,
        const std::vector<Scalar>& omegas_bg,
        const std::vector<Scalar>& weights_bg,
        const std::vector<Scalar>& syn_rates = {1.0},
        const std::vector<Scalar>& syn_weights = {1.0},
        const std::vector<Scalar>* custom_branch_lengths = nullptr,
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        Scalar eps_sink_test = 0.0,
        Scalar eps_sink_bg = 0.0,
        Scalar* out_new_eps_test = nullptr,
        Scalar* out_new_eps_bg = nullptr,
        Scalar max_eps_sink = 0.01,
        const std::vector<Scalar>* omegas_unclass = nullptr,
        const std::vector<Scalar>* weights_unclass = nullptr,
        std::vector<Scalar>* out_new_weights_unclass = nullptr,
        const MG94Parameters* custom_base_params = nullptr
    ) const;

    // Two-set synonymous rate E-step (with log-underflow scaling)
    std::vector<Scalar> compute_expected_two_set_syn_weights(
        Scalar s,
        const std::vector<Scalar>& omegas_test,
        const std::vector<Scalar>& weights_test,
        const std::vector<Scalar>& omegas_bg,
        const std::vector<Scalar>& weights_bg,
        const std::vector<Scalar>& syn_rates,
        const std::vector<Scalar>& syn_weights,
        const std::vector<Scalar>* custom_branch_lengths = nullptr,
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        Scalar eps_sink_test = 0.0,
        Scalar eps_sink_bg = 0.0,
        const std::vector<Scalar>* omegas_unclass = nullptr,
        const std::vector<Scalar>* weights_unclass = nullptr,
        const MG94Parameters* custom_base_params = nullptr
    ) const;

    // Two-set analytical branch gradients
    std::pair<Scalar, std::vector<Scalar>> compute_two_set_branch_gradients(
        const std::vector<Scalar>& branch_lengths,
        const std::vector<Scalar>& omegas_test,
        const std::vector<Scalar>& weights_test,
        const std::vector<Scalar>& omegas_bg,
        const std::vector<Scalar>& weights_bg,
        const std::vector<Scalar>& syn_rates = {1.0},
        const std::vector<Scalar>& syn_weights = {1.0},
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        Scalar eps_sink_test = 0.0,
        Scalar eps_sink_bg = 0.0,
        const std::vector<Scalar>* omegas_unclass = nullptr,
        const std::vector<Scalar>* weights_unclass = nullptr,
        const MG94Parameters* custom_base_params = nullptr
    ) const;

    std::vector<Scalar> refine_two_set_branch_lengths(
        const std::vector<Scalar>& init_bl,
        const std::vector<Scalar>& omegas_test,
        const std::vector<Scalar>& weights_test,
        const std::vector<Scalar>& omegas_bg,
        const std::vector<Scalar>& weights_bg,
        const std::vector<Scalar>& syn_rates = {1.0},
        const std::vector<Scalar>& syn_weights = {1.0},
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        int max_iters = 25,
        Scalar eps_sink_test = 0.0,
        Scalar eps_sink_bg = 0.0,
        const std::vector<Scalar>* omegas_unclass = nullptr,
        const std::vector<Scalar>* weights_unclass = nullptr,
        const MG94Parameters* custom_base_params = nullptr
    ) const;

    // Fits two-set (or three-set when num_unclassified_branches > 0) BUSTED / BUSTED-PH model
    // with independent K-class omega distributions on Test, Background, and (if present) Unclassified branches.
    BUSTEDFit fit_two_set_model(
        bool constrain_test,
        bool constrain_bg,
        size_t K = 3,
        bool refine_branches = true,
        bool srv = false,
        size_t num_syn_rates = 3,
        const std::string& multiple_hits = "None",
        ProgressBar* pbar = nullptr,
        const std::string& model_desc = "",
        bool use_error_sink = false,
        Scalar max_eps_sink = 0.01,
        const BUSTEDRateDistribution* init_test_dist = nullptr,
        const BUSTEDRateDistribution* init_bg_dist = nullptr,
        const BUSTEDRateDistribution* init_unclass_dist = nullptr,
        const std::vector<Scalar>* init_branch_lengths = nullptr,
        const MG94Parameters* init_base_params = nullptr,
        Scalar init_delta = 0.0,
        Scalar init_psi = 0.0
    ) const;

    // Fits coupled-k RELAX two-set model:
    //   Reference (Background, partition 1): K-class distribution (omegas_b[c], weights[c])
    //   Test (Foreground, partition 0):      K-class distribution (omegas_t[c] = (omegas_b[c])^k, weights[c])
    //   Unclassified (partition 2, if any):  Unconstrained K-class distribution (omegas_u[c], weights_u[c])
    // Warm-started from shared_null_fit (the exact k = 1 RELAX Null model), guaranteeing lnL_RELAX >= lnL_shared.
    BUSTEDFit fit_relax_coupled_model(
        const BUSTEDFit& shared_null_fit,
        size_t K = 3,
        bool refine_branches = true,
        bool srv = false,
        size_t num_syn_rates = 3,
        const std::string& multiple_hits = "None",
        ProgressBar* pbar = nullptr,
        const std::string& model_desc = "",
        bool use_error_sink = false,
        Scalar max_eps_sink = 0.01,
        Scalar* out_relax_k = nullptr
    ) const;

    // Fits Shared Distribution model (D_T == D_B on Test U Background; independent K-class on Unclassified if present)
    BUSTEDFit fit_shared_distribution_model(
        size_t K,
        const BUSTEDSettings& settings,
        ProgressBar* pbar,
        const std::string& model_desc,
        const BUSTEDFit* warm_unc = nullptr
    ) const;

    // Computes per-branch, per-site posterior probabilities and EBFs for omega_K > 1 (F-023)
    // and per-branch 2H / 3H / 2H+3H evidence ratios when multiple_hits != "None" (F-060)
    // in a single O(patterns * branches) Inside-Outside pass.
    void compute_branch_site_diagnostics(
        const BUSTEDFit& fit,
        const BUSTEDSettings& settings,
        BUSTEDResult& res
    ) const;

    // Run complete BUSTED or BUSTED-PH analysis with options
    BUSTEDResult run(
        BUSTEDSettings settings = {},
        bool show_progress = false,
        bool force_progress = false
    ) const;

    // Masks branch-site error events flagged by BUSTED-E (BF >= error_sink_bf_threshold)
    // as "---" in all descendant leaf sequences of the flagged branch at that codon site,
    // matching HyPhy 2.5's error-filter.bf workflow.
    Alignment filter_alignment_by_error_sink(const BUSTEDResult& res) const;

    // Writes the BUSTED-E error-filtered codon alignment to a FASTA file
    void write_filtered_alignment(const std::string& filepath, const BUSTEDResult& res) const;

nlohmann::json to_legacy_json(const BUSTEDResult& res) const;

nlohmann::json to_modern_json(const BUSTEDResult& res, const Provenance& prov = {}) const;
    nlohmann::json to_json(
        const BUSTEDResult& res,
        JSONFormat format = JSONFormat::ModernV3,
        const Provenance& prov = {}
    ) const;

    void save_json(
        const std::string& filepath,
        const BUSTEDResult& res,
        JSONFormat format = JSONFormat::ModernV3,
        const Provenance& prov = {}
    ) const;

};

} // namespace hyphy::analyses
