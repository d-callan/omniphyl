#pragma once

#include "hyphy/analyses/joint_align.hpp"
#include "hyphy/analyses/directional_fel.hpp"
#include "hyphy/analyses/prime.hpp"
#include "hyphy/core/console.hpp"
#include "hyphy/core/stats.hpp"
#include "hyphy/core/json_output.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hyphy::analyses {

using namespace hyphy::core;

// ============================================================================
// One-Button Evolutionary Characterization ("Characterize / Auto / Omni")
//   From raw unaligned coding sequences (or an existing MSA) to complete
//   alignment-marginalized, reticulation-aware, compositionally audited
//   gene-, branch-, clade-, site-, and biochemical property-level selection
//   inference via a single shared Inside-Outside sufficiency tensor,
//   two-stage active-set continuous polish, and hierarchical Simes/BH FDR.
// ============================================================================

struct CharacterizeSettings {
    Scalar p_value_threshold = 0.10;
    Scalar fdr_threshold = 0.10;
    Scalar active_set_screen_p = 0.25; // Decision 1: Two-stage active-set continuous polish gate
    bool from_unaligned = false;
    bool enable_coupled_spr = true;
    bool enable_segment_routing = true;
    bool enable_coevolution = false;   // Tier 3 (Opt-in): Epistatic co-substitution pairs (--coevolution)
    bool emit_legacy_jsons = false;    // Decision 7: Optional Vision/Datamonkey legacy JSON bundle
    bool quiet = false;
    std::string genetic_code_name = "Universal";
    std::string foreground_tag = "";   // Decision 5: Optional user foreground tag (auto-discovered if empty)
};

struct SpatialDomainSegment {
    size_t start_codon = 0; // 0-indexed
    size_t end_codon = 0;   // 0-indexed inclusive
    std::string domain_type = "Purifying Core"; // "Purifying Core", "Permissive / Surface", "Hypervariable / IDR"
    Scalar mean_alpha = 1.0;
    Scalar mean_beta = 0.1;
    Scalar mean_omega = 0.1;
    Scalar mean_gap_fraction = 0.0;
    Scalar mean_n_eff = 1.0;
};

struct SynonymousConstraintIsland {
    size_t start_codon = 0; // 0-indexed
    size_t end_codon = 0;   // 0-indexed inclusive
    Scalar mean_alpha = 0.1;
    Scalar gene_mean_alpha = 1.0;
    Scalar suppression_ratio = 0.1;
    Scalar z_score = 0.0;
    Scalar p_value = 1.0;
};

struct MutationalContextAlert {
    bool triggered = false;
    size_t num_selected_sites_evaluated = 0;
    size_t cpg_driven_selected_sites = 0;
    size_t apobec_driven_selected_sites = 0;
    Scalar cpg_enrichment_fold = 1.0;
    Scalar cpg_fisher_p = 1.0;
    std::vector<size_t> flagged_cpg_sites;    // 1-indexed site numbers
    std::vector<size_t> flagged_apobec_sites; // 1-indexed site numbers
    std::string alert_summary;
};

struct CoevolutionSitePair {
    size_t site_i = 0; // 0-indexed
    size_t site_j = 0; // 0-indexed
    Scalar shared_branches = 0.0;
    Scalar expected_shared = 0.0;
    Scalar cosine_similarity = 0.0;
    Scalar p_value = 1.0;
    Scalar bh_q_value = 1.0;
    std::string top_co_mutating_branches;
};

struct RecombinationBridgeSegment {
    int32_t node_u = -1;
    int32_t node_v = -1;
    std::string clade_u_name;
    std::string target_v_name;
    Scalar bridge_conductance = 0.0;
    size_t peak_start_codon = 0;
    size_t peak_end_codon = 0;
    Scalar segment_delta_log_l = 0.0;
    Scalar p_value = 1.0;
    bool routed_to_spr_tree = false; // Decision 3: True if segment p <= 0.01 routed to local SPR tree T_{u->v}
};

struct CladeMacroRegimeSummary {
    // Tier 2 Triggered: Dual-level autonomous macro-regime partition (or user {Foreground} contrast)
    bool triggered = false; // True if user supplied {Foreground} tags OR autonomous split achieves Delta BIC >= 10
    std::string partition_source = "None (Homogeneous Tree-Wide Regime)";
    std::string test_regime_label = "Focal Clade";
    std::string reference_regime_label = "Background";
    size_t num_test_branches = 0;
    size_t num_reference_branches = 0;
    Scalar clade_relax_k = 1.0;
    Scalar clade_relax_lrt = 0.0;
    Scalar clade_relax_delta_bic = 0.0;
    Scalar clade_relax_p_value = 1.0;
    std::string clade_relax_call = "Baseline"; // "Intensified", "Relaxed", or "Baseline"
    size_t contrast_fel_differing_sites = 0;
};

struct BranchEvolutionaryProfile {
    int32_t node_id = -1;
    std::string branch_name;
    std::string model_tag;
    bool is_leaf = false;
    Scalar branch_length_subs = 0.0;
    std::string compositional_regime = "Neutral";
    std::array<Scalar, 3> directional_weights = {0.0, 1.0, 0.0}; // Decision 4: Softmax weights [w_AT, w_Neu, w_GC]
    Scalar gc_cp3 = 0.50;
    std::string macro_regime = "Reference"; // Decision 5: "Test" vs "Reference"

    // aBSREL-style branch episodic diversifying selection (with Decision 1 active-set continuous polish)
    Scalar omega_bg = 0.5;
    Scalar omega_plus = 1.0;
    Scalar weight_plus = 0.0;
    Scalar absrel_lrt = 0.0;
    Scalar absrel_p_value = 1.0;
    Scalar absrel_holm_p = 1.0;
    Scalar absrel_bh_q = 1.0;

    // Autonomous RELAX selection intensity parameter K_b (omega_b = omega_gene^{K_b})
    Scalar relax_k = 1.0;
    Scalar relax_lrt = 0.0;
    Scalar relax_p_value = 1.0;
    Scalar relax_bh_q = 1.0;
    std::string relax_call = "Baseline"; // "Intensified", "Relaxed", or "Baseline"

    // Tier 1 (#2): Lineage-Specific Loss-of-Constraint / Pseudogenization (Core-Invasion) Detector
    Scalar core_nonsyn_Expected = 0.0;
    Scalar core_nonsyn_Observed = 0.0;
    Scalar lof_bayes_factor = 0.0;
    Scalar lof_p_value = 1.0;
    bool lof_flagged = false;
    std::vector<size_t> core_invaded_sites; // 1-indexed core sites mutated on this branch

    size_t episodic_site_count = 0; // Sites with Empirical Bayes EBF >= 10 on this branch
};

struct UnifiedSiteAtlasEntry {
    size_t site_index = 0; // 0-indexed (1-indexed in display)
    Scalar column_homology_posterior = 1.0;
    Scalar gap_fraction = 0.0;
    bool in_indel_window = false;
    size_t ensemble_paths_evaluated = 1; // Decision 2: K=3 POA path realizations if PP_c < 0.99
    std::string active_topology_label = "Backbone"; // Decision 3: "Backbone" or "SPR(u->v)"
    std::string spatial_domain = "Purifying Core";  // Tier 1 (#3): 1D spatial segment label
    bool in_synonymous_island = false;              // Tier 1 (#3): Inside a synonymous constraint island

    // Pervasive selection (Alignment-Marginalized FEL & Compositional-Drift-Adjusted dFEL)
    Scalar alpha = 1.0;
    Scalar beta_pervasive = 1.0;
    Scalar fel_lrt = 0.0;
    Scalar fel_p_value = 1.0;
    Scalar fel_bh_q = 1.0;
    Scalar dfel_lrt = 0.0;
    Scalar dfel_p_value = 1.0;
    Scalar dfel_bh_q = 1.0;

    // Episodic selection (Exact Inside-Outside MEME + Active-Set Continuous Polish + EBF)
    Scalar beta_minus = 0.5;
    Scalar beta_plus = 1.0;
    Scalar weight_plus = 0.0;
    Scalar meme_lrt = 0.0;
    Scalar meme_p_value = 1.0;
    Scalar meme_bh_q = 1.0;
    bool meme_polished = false;
    size_t num_episodic_branches = 0;
    std::vector<std::string> top_episodic_branches;
    Scalar max_branch_ebf = 0.0;

    // Decision 5 (Tier 2 Triggered): Clade / Macro-Regime Differential Selection (Contrast-FEL)
    Scalar beta_reference = 1.0;
    Scalar beta_test = 1.0;
    Scalar contrast_fel_lrt = 0.0;
    Scalar contrast_fel_p_value = 1.0;
    Scalar contrast_fel_bh_q = 1.0;

    // Tier 1 (#1): Model-Free Amino-Acid Exchangeability, Toggle & Convergence Classifier
    // (Replaces locked-in 5-property scales with empirical state-space concentration & homoplasy)
    Scalar expected_nonsyn_subs = 0.0;       // M_s = sum_b E[N_{s,b} | D]
    Scalar effective_aa_alphabet = 1.0;      // N_eff(s) = exp(H(pi_s^AA))
    Scalar expected_neutral_n_eff = 1.0;     // E[N_eff | M_s] under neutral codon branching
    Scalar pair_concentration = 0.0;         // Top unordered AA pair fraction of nonsyn transitions
    size_t max_convergent_transitions = 0;   // Max independent branches transitioning into the same target AA
    char dominant_target_aa = '-';
    Scalar exchangeability_stat = 0.0;       // Concentration / convergence surprise statistic
    Scalar exchangeability_p_value = 1.0;
    Scalar exchangeability_bh_q = 1.0;
    std::string exchangeability_mode = "Baseline"; // "Toggle (A<->B)", "Convergent (->A)", "Focused Exchange", "Broadly Diversifying"
    std::string key_property_driver;         // Summary string (e.g., "V<->I Toggle (6.2 subs, p=0.002)" or "Convergent ->K (4 br, p=0.004)")

    // Tier 2 (#4): Mutational context annotations on selected sites
    bool cpg_hypermutable_context = false;
    bool apobec_motif_context = false;

    // Decision 6: Hierarchical Simes-adjusted omnibus p-value & BH q-value + conflict-free label
    Scalar simes_omnibus_p = 1.0;
    Scalar simes_omnibus_q = 1.0;
    std::string classification = "Neutral";
    bool rigid_alignment_artifact_eliminated = false;
};

struct GeneWideSelectionSummary {
    // Decision 2: BUSTED-E (Alignment-Marginalized 3-class ω mixture + Dedicated Error-Sink Class ω_err)
    std::array<Scalar, 3> omega_classes = {0.1, 0.8, 1.5};
    std::array<Scalar, 3> omega_weights = {0.7, 0.25, 0.05};
    Scalar error_sink_omega = 50.0;
    Scalar error_sink_weight = 0.0;
    Scalar unconstrained_log_l = 0.0;
    Scalar null_log_l = 0.0;
    Scalar sink_null_log_l_gain = 0.0; // Null lnL with the error sink minus without it
    Scalar busted_lrt = 0.0;
    Scalar busted_p_value = 1.0;
    bool has_gene_wide_positive_selection = false;
};

struct CharacterizeResult {
    // Metadata
    std::string genetic_code_name = "Universal";
    std::string input_mode = "aligned"; // "unaligned" or "aligned"
    std::string tree_source = "de_novo_inferred"; // "de_novo_inferred", "user_supplied", or "embedded_in_alignment"
    Scalar p_value_threshold = 0.10;
    Scalar fdr_threshold = 0.10;

    // Stage I: Co-Estimated Alignment, Tree, Network & Objective Quality
    JointAlignResult joint_align;
    JointAlignEngine::AlignmentObjectiveEvaluation alignment_objective;
    std::vector<RecombinationBridgeSegment> recombination_segments;

    // Stage II: Compositional Drift & Auto-Rooting Audit
    CompositionTestResult composition_test;
    bool directional_drift_active = false;
    std::string chosen_root_branch;
    Scalar chosen_root_weight = 0.0;

    // Stage III: Unified Gene, Clade, Branch, Domain & Site Selection Atlas
    GeneWideSelectionSummary gene_wide;
    CladeMacroRegimeSummary clade_macro_regime;
    std::vector<SpatialDomainSegment> spatial_domains;
    std::vector<SynonymousConstraintIsland> synonymous_islands;
    MutationalContextAlert mutational_context_alert;
    std::vector<CoevolutionSitePair> coevolution_pairs;
    std::vector<BranchEvolutionaryProfile> branch_profiles;
    std::vector<UnifiedSiteAtlasEntry> site_atlas;
    std::vector<int> node_ancestral_codons; // Empirical Bayes MAP sense codon [nid * C + c]

    // Counts
    size_t sites_pervasive_diversifying = 0;
    size_t sites_pervasive_purifying = 0;
    size_t sites_episodic_diversifying = 0;
    size_t sites_regime_differential = 0;
    size_t sites_property_constrained_or_shifting = 0; // Toggle / Convergent / Focused Exchange sites
    size_t sites_simes_fdr_significant = 0;
    size_t sites_simes_diversifying = 0;
    size_t sites_simes_purifying = 0;
    size_t branches_episodic_diversifying = 0;
    size_t branches_intensified = 0;
    size_t branches_relaxed = 0;
    size_t branches_loss_of_constraint = 0;

    double runtime_seconds = 0.0;
};

class CharacterizeEngine {
public:
    CharacterizeSettings settings;

    explicit CharacterizeEngine(CharacterizeSettings s = CharacterizeSettings{});

    // Benjamini-Hochberg step-up FDR q-values from a vector of raw p-values
    static std::vector<Scalar> compute_bh_q_values(const std::vector<Scalar>& p_vals);

    // Simes (1986) combined p-value across a family of K test p-values at a single site
    static Scalar compute_simes_p_value(std::vector<Scalar> p_family);

    CharacterizeResult run_from_unaligned_file(
        const std::string& filepath,
        std::shared_ptr<const GeneticCode> gcode = nullptr
    );

    CharacterizeResult run_from_unaligned(
        const std::vector<Sequence>& unaligned_seqs,
        std::shared_ptr<const GeneticCode> gcode = nullptr,
        std::optional<Tree> input_tree = std::nullopt
    );

    CharacterizeResult run_from_alignment(
        const Alignment& input_aln,
        std::optional<Tree> input_tree = std::nullopt
    );

    static std::string to_json(const CharacterizeResult& res);

    // Decision 7: Optional legacy JSON bundle emission (.FEL.json, .MEME.json, .BUSTED.json)
    static void write_legacy_json_bundle(const CharacterizeResult& res, const std::string& base_prefix);

    // Decision 7: Unified, self-contained interactive 6-stage multi-tab SPA (.CHARACTERIZE.html)
    static std::string render_standalone_html_from_json_string(
        const std::string& raw_json_payload,
        const std::string& dataset_label
    );

    static std::string render_standalone_html(const CharacterizeResult& res, const std::string& dataset_label);

private:
    CharacterizeResult run_unified_tensor_characterization(JointAlignResult&& ja_res);
};

} // namespace hyphy::analyses

