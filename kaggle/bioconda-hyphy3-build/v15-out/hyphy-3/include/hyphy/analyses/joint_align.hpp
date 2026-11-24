#pragma once

#include "hyphy/core/types.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/rate_matrix.hpp"
#include "hyphy/core/likelihood.hpp"
#include "hyphy/core/ohmic_network.hpp"
#include "hyphy/core/wheatstone_likelihood.hpp"
#include "hyphy/analyses/tree_engine.hpp"
#include "hyphy/analyses/fel.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <iostream>
#include <iomanip>

namespace hyphy::analyses {

using namespace hyphy::core;

// =============================================================================
// Joint Ohmic Alignment + Phylogeny + Selection Engine (3-Stage Prototype)
//   Stage 1: Phylogenetic Codon Indel Transducer & Window Realignment (Inside-Outside EM)
//   Stage 2: Coupled Ohmic SPR / Port-Slide with Boundary Indel Transducer Updates
//   Stage 3: Alignment-Marginalized FEL (Posterior Homology-Weighted Selection LRT)
// =============================================================================

struct IndelWindow {
    size_t start_codon = 0; // inclusive (0-indexed)
    size_t end_codon = 0;   // exclusive (0-indexed)
    size_t num_gapped_taxa = 0;
    size_t total_gap_codons = 0;
    Scalar mean_gap_fraction = 0.0;
};

struct TaxonWindowPlacement {
    size_t taxon_idx = 0;
    std::vector<int8_t> aligned_codons; // length W = end_codon - start_codon (-1 for gap)
    std::string raw_Subseq;             // length 3 * W
    Scalar log_posterior = 0.0;
    Scalar posterior_prob = 1.0;
    bool changed_from_input = false;
};

struct WindowRealignmentSummary {
    IndelWindow window;
    Scalar initial_window_log_l = 0.0;
    Scalar polished_window_log_l = 0.0;
    Scalar delta_log_l = 0.0;
    size_t num_taxa_realigned = 0;
    size_t num_codons_shifted = 0;
    std::vector<Scalar> column_homology_posteriors; // length W: expected posterior confidence per column
};

struct CoupledMoveEvent {
    std::string move_type; // "Coupled_SPR" or "Coupled_PortSlide"
    int32_t clade_node_id = -1;
    int32_t target_node_id = -1;
    Scalar delta_sub_log_l = 0.0;
    Scalar delta_joint_log_l = 0.0;
    size_t realigned_clade_taxa = 0;
};

struct MarginalizedSiteResult {
    size_t site_index = 0; // 0-indexed codon site
    bool in_indel_window = false;
    Scalar column_homology_posterior = 1.0; // Posterior alignment consistency in [0, 1]
    Scalar gap_fraction = 0.0;

    // Input (rigid MSA) FEL statistics
    Scalar rigid_alpha = 0.0;
    Scalar rigid_beta = 0.0;
    Scalar rigid_lrt = 0.0;
    Scalar rigid_p_value = 1.0;

    // MAP Joint Realigned FEL statistics
    Scalar map_alpha = 0.0;
    Scalar map_beta = 0.0;
    Scalar map_lrt = 0.0;
    Scalar map_p_value = 1.0;

    // Stage 3: Alignment-Marginalized FEL statistics (integrated over candidate gap placements)
    Scalar marginalized_alpha = 0.0;
    Scalar marginalized_beta = 0.0;
    Scalar marginalized_lrt = 0.0;
    Scalar marginalized_p_value = 1.0;

    bool artifact_eliminated = false; // True if rigid_p <= threshold && marginalized_p > threshold
    bool selection_rescued = false;   // True if rigid_p > threshold && marginalized_p <= threshold
};

struct TaxonAdmissionEntry {
    std::string name;
    std::string status = "Admitted"; // "Admitted", "Rescued", "Quarantined"
    std::string reason = "Clean in-frame CDS";
    int frame_offset = 0;            // 0, 1, or 2
    bool reverse_complemented = false;
    size_t raw_nt_length = 0;
    size_t codon_length = 0;
    size_t internal_stops_frame0 = 0;
    Scalar max_homology_z = 0.0;
    Scalar max_aa_identity = 0.0;
};

struct InputAdmissionReport {
    size_t total_input_sequences = 0;
    size_t admitted_sequences = 0;
    size_t rescued_frame_or_strand = 0;
    size_t quarantined_sequences = 0;
    Scalar median_pairwise_z = 0.0;
    Scalar median_aa_identity = 0.0;
    Scalar msa_expansion_ratio = 1.0;
    Scalar high_confidence_column_fraction = 1.0;
    Matrix admitted_dp_similarity;
    std::vector<TaxonAdmissionEntry> entries;
};

struct JointAlignSettings {
    // Stage 0 (Input Triage & Homology Admission Control) settings
    bool enforce_admission_control = true;
    size_t min_sense_codons = 15;        // reject/quarantine tiny fragments (< 15 codons)
    Scalar min_homology_z = 3.5;         // minimum pairwise homology Z-score to join main component
    Scalar min_cohort_median_z = 4.5;    // dataset-wide minimum median pairwise homology Z-score
    Scalar max_msa_expansion_ratio = 2.0;// maximum allowed C_total / median_L before aborting as staggered junk

    // Stage 1 settings
    Scalar transducer_gap_open = 2.85;   // Phase 2 progressive transducer affine gap-open penalty (in nats)
    Scalar transducer_gap_extend = 0.65; // Phase 2 progressive transducer affine gap-extend penalty (in nats)
    Scalar transducer_laplace_floor = 0.22; // Phase 2 homology Bayes-factor Laplace floor \epsilon_0
    Scalar gap_open_penalty = 3.2;       // Phase 3 window realignment log-penalty for opening a codon deletion block
    Scalar gap_extend_penalty = 0.85;    // Phase 3 window realignment log-penalty per additional deleted codon
    size_t window_flank_codons = 2;      // conserved anchor codons on each side of a gappy region
    size_t max_window_codons = 18;       // maximum width of a local realignment window
    size_t em_iterations = 2;            // Inside-Outside cavity realignment EM sweeps

    // Subquadratic scaling settings for large alignments (N > 64 taxa or L > 200 codons)
    size_t subquadratic_taxon_threshold = 64;   // switch to O(N sqrt(N)) landmark + k-mer k-NN graph when N > threshold
    size_t subquadratic_length_threshold = 200; // switch to O(B * L) seed-chain anchor-banded profile DP when min(L1, L2) > threshold

    // Stage 2 settings
    bool enable_coupled_topology_search = true;
    size_t max_coupled_spr_candidates = 8;

    // Stage 3 settings
    bool run_marginalized_fel = true;
    Scalar p_value_threshold = 0.10;
    bool quiet = false;
};

struct TransducerTaxonResult {
    std::vector<int8_t> map_codons;          // length W
    std::vector<uint64_t> map_masks;         // length W: resolved IUPAC bitmasks
    std::string map_raw_subseq;              // length 3*W
    Scalar input_log_score = 0.0;
    Scalar map_log_score = 0.0;
    Scalar log_partition_Z = 0.0;
    Scalar map_posterior_prob = 1.0;
    std::vector<Scalar> col_match_posterior; // length W: posterior prob that column w is a homologous Match
    bool changed = false;
    size_t codons_shifted = 0;

    // Top K candidate placements with their exact Forward-Backward posterior weights
    struct CandidatePlacement {
        std::vector<int8_t> codons;
        Scalar log_score = -1e20;
        Scalar posterior_weight = 0.0;
    };
    std::vector<CandidatePlacement> top_candidates;
};

struct JointAlignResult {
    InputAdmissionReport admission_report;
    Alignment initial_alignment;
    Alignment map_alignment;
    Tree initial_tree;
    Tree optimized_tree;

    Scalar initial_gtr_log_l = 0.0;
    Scalar initial_mg94_log_l = 0.0;
    Scalar stage1_mg94_log_l = 0.0;
    Scalar stage2_mg94_log_l = 0.0;

    std::vector<WindowRealignmentSummary> windows;
    std::vector<std::vector<TransducerTaxonResult>> window_taxon_results;
    std::vector<CoupledMoveEvent> coupled_moves;
    std::vector<MarginalizedSiteResult> site_results;

    // Posterior column confidence across the entire alignment (size = num_codons)
    std::vector<Scalar> site_homology_posteriors;

    // Co-estimated Ohmic electrical network, Extended Newick (with #H<k> Wheatstone bridges & NHX support),
    // Stockholm quality-annotated MSA (#=GC PP_cons), and Partial Order Alignment (POA) graphs (GFAv1 & DOT)
    OhmicNetwork ohmic_network;
    std::string extended_newick;
    std::string stockholm_alignment;
    std::string poa_gfa;
    std::string poa_dot;

    size_t total_taxa_realigned = 0;
    size_t total_codons_shifted = 0;
    size_t total_coupled_moves_accepted = 0;
    size_t false_positive_sites_eliminated = 0;
    size_t true_positive_sites_rescued = 0;
    double runtime_seconds = 0.0;
};

class JointAlignEngine {
public:
    JointAlignSettings settings;

    explicit JointAlignEngine(JointAlignSettings s = JointAlignSettings{});

    // =========================================================================
    // Empirical Two-Stratum (Core + Loop) Physicochemical & Multi-Hit Codon Model
    //   Shared across Stage 1 Phase 2 (progressive profile alignment) and
    //   Phase 3 (iterative Inside-Outside window refinement).
    // =========================================================================
    static Scalar aa_phys_dist(char a1, char a2);

    struct EmpRegimeParams {
        Scalar omega_hat = 0.25;
        Scalar theta_tv  = 0.35;
        Scalar theta_ct  = 1.00;
        Scalar lambda_aa = 1.80;
        Scalar delta_2nt = 0.15;
        Scalar psi_3nt   = 0.04;
        Scalar obs_mean_daa = 0.25;
    };

    struct EmpRegimeCounts {
        Scalar sum_ag = 1.0, sum_ct = 1.0, sum_tv = 1.0;
        Scalar sum_syn1 = 1.0, sum_ns1 = 1.0, sum_ns1_daa = 0.25;
        Scalar sum_1nt = 1.0, sum_2nt = 0.5, sum_3nt = 0.1;

        void init_from_neutral(Scalar mu0_daa) {
            sum_ns1_daa = mu0_daa;
        }

        void add_codon_pair(int8_t c1, int8_t c2, const GeneticCode& code) {
            if (c1 < 0 || c2 < 0 || c1 == c2) return;
            const auto& diff = code.diff_matrix[c1][c2];
            Scalar d_aa = diff.synonymous ? 0.0 : aa_phys_dist(code.sense_codon_aa[c1], code.sense_codon_aa[c2]);
            if (diff.num_diffs == 1) {
                sum_1nt += 1.0;
                int b1 = diff.diff_from[0], b2 = diff.diff_to[0];
                bool p1 = (b1 == 0 || b1 == 2), p2 = (b2 == 0 || b2 == 2);
                if (p1 && p2) sum_ag += 1.0;
                else if (!p1 && !p2) sum_ct += 1.0;
                else sum_tv += 1.0;
                if (diff.synonymous) {
                    sum_syn1 += 1.0;
                } else {
                    sum_ns1 += 1.0;
                    sum_ns1_daa += d_aa;
                }
            } else if (diff.num_diffs == 2) {
                sum_2nt += 1.0;
            } else if (diff.num_diffs == 3) {
                sum_3nt += 1.0;
            }
        }

        EmpRegimeParams finalize(
            Scalar mu0_daa, Scalar var0_daa,
            Scalar syn_to_ns_opp_ratio,
            Scalar opp_1_to_2, Scalar opp_1_to_3
        ) const {
            EmpRegimeParams p;
            p.theta_ct  = std::clamp<Scalar>(sum_ct / sum_ag, 0.40, 2.50);
            p.theta_tv  = std::clamp<Scalar>(sum_tv / (sum_ag + sum_ct), 0.10, 0.85);
            Scalar raw_omega = (sum_ns1 / sum_syn1) * syn_to_ns_opp_ratio;
            p.omega_hat = std::clamp<Scalar>(raw_omega, 0.06, 1.50);
            p.obs_mean_daa = sum_ns1_daa / std::max<Scalar>(1.0, sum_ns1);
            Scalar raw_lambda = (mu0_daa - p.obs_mean_daa) / std::max<Scalar>(1e-4, var0_daa);
            p.lambda_aa = std::clamp<Scalar>(raw_lambda, 0.0, 3.50);
            p.delta_2nt = std::clamp<Scalar>((sum_2nt / sum_1nt) * opp_1_to_2, 0.06, 0.55);
            p.psi_3nt   = std::clamp<Scalar>((sum_3nt / sum_1nt) * opp_1_to_3, 0.01, 0.25);
            return p;
        }
    };

    struct EmpiricalSpectralGenerator {
        Matrix U_pi, V_pi;
        Vector eigenvalues;
        Scalar scale_factor = 1.0;

        void build(const EmpRegimeParams& st, const Eigen::Matrix<Scalar, 3, 4>& pf,
                   const Vector& pi, const GeneticCode& code) {
            int nS = code.num_sense_codons;
            Vector sqrt_pi(nS), inv_sqrt_pi(nS);
            for (int i = 0; i < nS; ++i) {
                sqrt_pi(i) = std::sqrt(std::max<Scalar>(pi(i), 1e-12));
                inv_sqrt_pi(i) = 1.0 / sqrt_pi(i);
            }
            Scalar r_nuc[4][4] = {
                {0.0, st.theta_tv, 1.0, st.theta_tv},
                {st.theta_tv, 0.0, st.theta_tv, st.theta_ct},
                {1.0, st.theta_tv, 0.0, st.theta_tv},
                {st.theta_tv, st.theta_ct, st.theta_tv, 0.0}
            };
            Scalar sum_aa_w = 0.0;
            int count_ns1 = 0;
            for (int i = 0; i < nS; ++i) {
                for (int j = i + 1; j < nS; ++j) {
                    const auto& diff = code.diff_matrix[i][j];
                    if (diff.num_diffs == 1 && !diff.synonymous) {
                        sum_aa_w += std::exp(-st.lambda_aa * aa_phys_dist(code.sense_codon_aa[i], code.sense_codon_aa[j]));
                        count_ns1++;
                    }
                }
            }
            Scalar mean_aa_w = (count_ns1 > 0) ? (sum_aa_w / static_cast<Scalar>(count_ns1)) : 1.0;

            Matrix Q = Matrix::Zero(nS, nS);
            for (int i = 0; i < nS; ++i) {
                for (int j = 0; j < nS; ++j) {
                    if (i == j) continue;
                    const auto& diff = code.diff_matrix[i][j];
                    Scalar d_aa = diff.synonymous ? 0.0 : aa_phys_dist(code.sense_codon_aa[i], code.sense_codon_aa[j]);
                    Scalar aa_mod = diff.synonymous ? 1.0 : (st.omega_hat * std::exp(-st.lambda_aa * d_aa) / mean_aa_w);
                    if (diff.num_diffs == 1) {
                        Q(i, j) = aa_mod * r_nuc[diff.diff_from[0]][diff.diff_to[0]] * pf(diff.diff_pos[0], diff.diff_to[0]);
                    } else if (diff.num_diffs == 2) {
                        Scalar nr = r_nuc[diff.diff_from[0]][diff.diff_to[0]] * r_nuc[diff.diff_from[1]][diff.diff_to[1]];
                        Scalar tf = pf(diff.diff_pos[0], diff.diff_to[0]) * pf(diff.diff_pos[1], diff.diff_to[1]);
                        Scalar mh = diff.synonymous ? 1.0 : st.delta_2nt;
                        Q(i, j) = aa_mod * mh * nr * tf;
                    } else if (diff.num_diffs == 3) {
                        Scalar nr = r_nuc[diff.diff_from[0]][diff.diff_to[0]] * r_nuc[diff.diff_from[1]][diff.diff_to[1]] * r_nuc[diff.diff_from[2]][diff.diff_to[2]];
                        Scalar tf = pf(diff.diff_pos[0], diff.diff_to[0]) * pf(diff.diff_pos[1], diff.diff_to[1]) * pf(diff.diff_pos[2], diff.diff_to[2]);
                        Scalar mh = diff.synonymous ? 1.0 : st.psi_3nt;
                        Q(i, j) = aa_mod * mh * nr * tf;
                    }
                }
                Q(i, i) = -Q.row(i).sum();
            }
            Scalar avg_rate = 0.0;
            for (int i = 0; i < nS; ++i) avg_rate -= pi(i) * Q(i, i);
            scale_factor = std::max<Scalar>(1e-12, avg_rate);

            Matrix S_mat = sqrt_pi.asDiagonal() * Q * inv_sqrt_pi.asDiagonal();
            S_mat = 0.5 * (S_mat + S_mat.transpose());
            Eigen::SelfAdjointEigenSolver<Matrix> solver(S_mat);
            eigenvalues = solver.eigenvalues().cwiseMin(0.0);
            Matrix eigvecs = solver.eigenvectors();
            U_pi = inv_sqrt_pi.asDiagonal() * eigvecs;
            V_pi = eigvecs.transpose() * sqrt_pi.asDiagonal();
        }

        Matrix transition_matrix(Scalar branch_length) const {
            Scalar conv = 3.0 / scale_factor;
            Vector exp_l = (eigenvalues * (branch_length * conv)).array().exp();
            return ((U_pi * exp_l.asDiagonal()) * V_pi).cwiseMax(0.0);
        }
    };

    struct EmpiricalTwoStratumModel {
        EmpRegimeParams emp_core, emp_loop;
        Scalar w_loop = 0.30;
        Scalar rho_loop_to_core = 4.0;
        EmpiricalSpectralGenerator q_core, q_loop;
        Vector pi;

        Matrix transition_matrix(Scalar bl) const {
            Scalar t = std::max<Scalar>(1e-4, bl);
            return (1.0 - w_loop) * q_core.transition_matrix(t)
                 + w_loop * q_loop.transition_matrix(rho_loop_to_core * t);
        }

        static EmpiricalTwoStratumModel calibrate(
            const std::vector<std::vector<int8_t>>& taxon_codons,
            const Eigen::Matrix<Scalar, 3, 4>& pos_freqs,
            const Vector& pi_codon,
            const GeneticCode& gcode,
            size_t subquadratic_taxon_threshold = 64
        ) {
            EmpiricalTwoStratumModel m;
            m.pi = pi_codon;
            int S = gcode.num_sense_codons;
            size_t N = taxon_codons.size();

            Scalar sum_d0 = 0.0, sum_d0_sq = 0.0;
            Scalar n_ns1_opp = 0.0, n_syn1_opp = 0.0;
            Scalar n_1nt_opp = 0.0, n_2nt_opp = 0.0, n_3nt_opp = 0.0;
            for (int i = 0; i < S; ++i) {
                for (int j = i + 1; j < S; ++j) {
                    const auto& diff = gcode.diff_matrix[i][j];
                    if (diff.num_diffs == 1) {
                        n_1nt_opp += 1.0;
                        if (diff.synonymous) {
                            n_syn1_opp += 1.0;
                        } else {
                            Scalar d = aa_phys_dist(gcode.sense_codon_aa[i], gcode.sense_codon_aa[j]);
                            sum_d0 += d;
                            sum_d0_sq += d * d;
                            n_ns1_opp += 1.0;
                        }
                    } else if (diff.num_diffs == 2) {
                        n_2nt_opp += 1.0;
                    } else if (diff.num_diffs == 3) {
                        n_3nt_opp += 1.0;
                    }
                }
            }
            Scalar mu0_daa = (n_ns1_opp > 0) ? (sum_d0 / n_ns1_opp) : 0.32;
            Scalar var0_daa = (n_ns1_opp > 0) ? std::max<Scalar>(1e-4, sum_d0_sq / n_ns1_opp - mu0_daa * mu0_daa) : 0.035;
            Scalar syn_to_ns_opp_ratio = (n_ns1_opp > 0) ? (n_syn1_opp / n_ns1_opp) : (0.25 / 0.75);
            Scalar opp_1_to_2 = (n_2nt_opp > 0) ? (n_1nt_opp / n_2nt_opp) : 0.35;
            Scalar opp_1_to_3 = (n_3nt_opp > 0) ? (n_1nt_opp / n_3nt_opp) : 0.32;

            EmpRegimeCounts cnt_all, cnt_core, cnt_loop;
            cnt_all.init_from_neutral(mu0_daa);
            cnt_core.init_from_neutral(mu0_daa);
            cnt_loop.init_from_neutral(mu0_daa);
            Scalar sum_r_core = 0.0, sum_r_loop = 0.0;
            size_t n_core = 0, n_loop = 0;

            const bool use_subquadratic_cal = (N > subquadratic_taxon_threshold);
            const size_t cal_stride = use_subquadratic_cal ? std::max<size_t>(1, N / 6) : 1;

            for (size_t u = 0; u < N; ++u) {
                const auto& su = taxon_codons[u];
                for (size_t v = u + 1; v < N; ++v) {
                    if (use_subquadratic_cal && (v - u) > 2 && ((v - u) % cal_stride != 0)) continue;
                    const auto& sv = taxon_codons[v];
                    if (su.empty() || sv.empty()) continue;
                    size_t Lmin = std::min(su.size(), sv.size());
                    size_t Lmax = std::max(su.size(), sv.size());
                    if (Lmin < 5 || Lmax > Lmin * 2) continue;

                    // Align (su, sv) via banded affine codon DP so internal indels never shift
                    // post-indel codons out of register during Stage-0 rate matrix calibration.
                    const auto& s_short = (su.size() <= sv.size()) ? su : sv;
                    const auto& s_long  = (su.size() <= sv.size()) ? sv : su;
                    int Ls = static_cast<int>(s_short.size());
                    int Ll = static_cast<int>(s_long.size());
                    int dL = Ll - Ls;
                    int B0 = std::max(24, dL + 8);
                    int band_w = 2 * B0 + dL + 3;
                    constexpr Scalar kNegInf = -1e9;
                    constexpr Scalar cal_g_open = 1.10;
                    constexpr Scalar cal_g_ext  = 0.30;

                    std::vector<Scalar> prev_M(Ll + 1, kNegInf), curr_M(Ll + 1, kNegInf);
                    std::vector<Scalar> prev_D(Ll + 1, kNegInf), curr_D(Ll + 1, kNegInf);
                    std::vector<uint8_t> bt_mat((Ls + 1) * band_w, 0); // 0=M, 1=D(up), 2=I(left)
                    auto b_idx = [&](int r, int c) -> size_t {
                        int off = c - (r - B0);
                        return static_cast<size_t>(r * band_w + std::clamp(off, 0, band_w - 1));
                    };

                    int init_max = std::min(Ll, dL + B0);
                    for (int c = 0; c <= init_max; ++c) {
                        prev_M[c] = -std::min<Scalar>(cal_g_open + cal_g_ext * c, 3.80);
                        bt_mat[b_idx(0, c)] = 2;
                    }
                    prev_M[0] = 0.0;

                    for (int r = 1; r <= Ls; ++r) {
                        int c_min = std::max(1, r - B0);
                        int c_max = std::min(Ll, r + dL + B0);
                        int prev_c_max = std::min(Ll, (r - 1) + dL + B0);
                        std::fill(curr_M.begin() + std::max(0, c_min - 1), curr_M.begin() + c_max + 1, kNegInf);
                        std::fill(curr_D.begin() + std::max(0, c_min - 1), curr_D.begin() + c_max + 1, kNegInf);
                        if (r <= B0) {
                            Scalar pen = -std::min<Scalar>(cal_g_open + cal_g_ext * r, 3.80);
                            curr_D[0] = pen;
                            curr_M[0] = pen;
                            bt_mat[b_idx(r, 0)] = 1;
                        }
                        int8_t ci = s_short[r - 1];
                        Scalar ins_sc = kNegInf;
                        for (int c = c_min; c <= c_max; ++c) {
                            int8_t cj = s_long[c - 1];
                            Scalar d_sc = (c <= prev_c_max)
                                        ? std::max(prev_M[c] - cal_g_open - cal_g_ext, prev_D[c] - cal_g_ext)
                                        : kNegInf;
                            curr_D[c] = d_sc;
                            bool left_in_band = (c > c_min) || (c == 1 && r <= B0);
                            Scalar i_op = left_in_band ? (curr_M[c - 1] - cal_g_open - cal_g_ext) : kNegInf;
                            Scalar i_ex = (c > c_min) ? (ins_sc - cal_g_ext) : kNegInf;
                            ins_sc = std::max(i_op, i_ex);
                            Scalar sim = 0.0;
                            if (ci >= 0 && cj >= 0) {
                                if (ci == cj) sim = 1.00;
                                else if (gcode.sense_codon_aa[ci] == gcode.sense_codon_aa[cj]) sim = 0.90;
                                else if (gcode.diff_matrix[ci][cj].num_diffs == 1) sim = 0.30;
                            }
                            Scalar m_sc = prev_M[c - 1] + sim;
                            Scalar best_cell = m_sc;
                            uint8_t best_bt = 0;
                            if (d_sc > best_cell) {
                                best_cell = d_sc;
                                best_bt = 1;
                            }
                            if (ins_sc > best_cell) {
                                best_cell = ins_sc;
                                best_bt = 2;
                            }
                            curr_M[c] = best_cell;
                            bt_mat[b_idx(r, c)] = best_bt;
                        }
                        std::swap(prev_M, curr_M);
                        std::swap(prev_D, curr_D);
                    }

                    int best_c = Ll;
                    Scalar best_sc = prev_M[Ll];
                    for (int c = std::max(1, Ls - B0); c <= Ll; ++c) {
                        Scalar sc = prev_M[c] - std::min<Scalar>(cal_g_open + cal_g_ext * (Ll - c), 3.80);
                        if (sc > best_sc) { best_sc = sc; best_c = c; }
                    }

                    std::vector<std::pair<int8_t, int8_t>> matched_pairs;
                    matched_pairs.reserve(Lmin);
                    int cr = Ls, cc = best_c;
                    int aa_matches = 0;
                    while (cr > 0 && cc > 0) {
                        uint8_t mv = bt_mat[b_idx(cr, cc)];
                        if (mv == 0) {
                            int8_t c1 = s_short[cr - 1], c2 = s_long[cc - 1];
                            if (c1 >= 0 && c2 >= 0) {
                                matched_pairs.push_back({c1, c2});
                                if (c1 == c2 || gcode.sense_codon_aa[c1] == gcode.sense_codon_aa[c2]) aa_matches++;
                            }
                            cr--; cc--;
                        } else if (mv == 1) {
                            cr--;
                        } else {
                            cc--;
                        }
                    }
                    if (matched_pairs.size() < Lmin / 2) continue;
                    Scalar pair_aa_id = static_cast<Scalar>(aa_matches) / static_cast<Scalar>(matched_pairs.size());
                    if (pair_aa_id < 0.35) continue;
                    std::reverse(matched_pairs.begin(), matched_pairs.end());

                    struct AlignedSite {
                        int8_t c1, c2;
                        Scalar d_nt;
                        Scalar d_aa;
                    };
                    std::vector<AlignedSite> sites;
                    sites.reserve(matched_pairs.size());
                    Scalar sum_dnt = 0.0, sum_daa = 0.0;
                    for (const auto& [c1, c2] : matched_pairs) {
                        Scalar d_nt = 0.0, d_aa = 0.0;
                        if (c1 != c2) {
                            const auto& diff = gcode.diff_matrix[c1][c2];
                            d_nt = static_cast<Scalar>(diff.num_diffs) / 3.0;
                            if (!diff.synonymous) {
                                d_aa = aa_phys_dist(gcode.sense_codon_aa[c1], gcode.sense_codon_aa[c2]);
                            }
                        }
                        sites.push_back({c1, c2, d_nt, d_aa});
                        sum_dnt += d_nt;
                        sum_daa += d_aa;
                        cnt_all.add_codon_pair(c1, c2, gcode);
                    }
                    if (sites.size() < 5 || sum_dnt <= 1e-9) continue;
                    Scalar mean_dnt = sum_dnt / static_cast<Scalar>(sites.size());
                    Scalar mean_daa = sum_daa / static_cast<Scalar>(sites.size());

                    int M = static_cast<int>(sites.size());
                    for (int k = 0; k < M; ++k) {
                        int w0 = std::max(0, k - 2);
                        int w1 = std::min(M - 1, k + 2);
                        Scalar win_dnt = 0.0, win_daa = 0.0;
                        for (int w = w0; w <= w1; ++w) {
                            win_dnt += sites[w].d_nt;
                            win_daa += sites[w].d_aa;
                        }
                        win_dnt /= static_cast<Scalar>(w1 - w0 + 1);
                        win_daa /= static_cast<Scalar>(w1 - w0 + 1);
                        bool is_loop = (mean_daa > 1e-9) ? (win_daa > mean_daa) : (win_dnt > mean_dnt);
                        Scalar r_site = (mean_daa > 1e-9)
                            ? 0.5 * (sites[k].d_nt / mean_dnt + sites[k].d_aa / mean_daa)
                            : (sites[k].d_nt / mean_dnt);
                        if (!is_loop) {
                            n_core++;
                            sum_r_core += r_site;
                            cnt_core.add_codon_pair(sites[k].c1, sites[k].c2, gcode);
                        } else {
                            n_loop++;
                            sum_r_loop += r_site;
                            cnt_loop.add_codon_pair(sites[k].c1, sites[k].c2, gcode);
                        }
                    }
                }
            }

            EmpRegimeParams emp_all = cnt_all.finalize(mu0_daa, var0_daa, syn_to_ns_opp_ratio, opp_1_to_2, opp_1_to_3);
            m.emp_core = cnt_core.finalize(mu0_daa, var0_daa, syn_to_ns_opp_ratio, opp_1_to_2, opp_1_to_3);
            m.emp_loop = cnt_loop.finalize(mu0_daa, var0_daa, syn_to_ns_opp_ratio, opp_1_to_2, opp_1_to_3);
            m.emp_core.theta_ct = emp_all.theta_ct;
            m.emp_core.theta_tv = emp_all.theta_tv;
            m.emp_loop.theta_ct = emp_all.theta_ct;
            m.emp_loop.theta_tv = emp_all.theta_tv;

            if (n_core > 0 && n_loop > 0) {
                Scalar w_l = static_cast<Scalar>(n_loop) / static_cast<Scalar>(n_core + n_loop);
                Scalar r_c = sum_r_core / static_cast<Scalar>(n_core);
                Scalar r_l = sum_r_loop / static_cast<Scalar>(n_loop);
                m.w_loop = std::clamp<Scalar>(w_l, 0.10, 0.60);
                m.rho_loop_to_core = std::clamp<Scalar>(r_l / std::max<Scalar>(0.28, r_c), 1.5, 11.0);
            }

            m.q_core.build(m.emp_core, pos_freqs, pi_codon, gcode);
            m.q_loop.build(m.emp_loop, pos_freqs, pi_codon, gcode);
            return m;
        }

        static EmpiricalTwoStratumModel from_alignment(const Alignment& aln) {
            const auto& gcode = *(aln.code ? aln.code : GeneticCode::universal());
            std::vector<std::vector<int8_t>> tc(aln.num_taxa);
            for (size_t t = 0; t < aln.num_taxa; ++t) {
                tc[t].reserve(aln.num_codons);
                for (int8_t c : aln.sequences[t].codons) {
                    if (c >= 0) tc[t].push_back(c);
                }
            }
            return calibrate(tc, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, gcode);
        }
    };

    // =========================================================================
    // Helper: Detect Indel-Ambiguous Codon Windows
    // =========================================================================
    static inline bool is_gap_codon(const Sequence& seq, size_t c) {
        if (3 * c + 3 <= seq.raw_seq.size()) {
            return seq.raw_seq[3 * c] == '-' && seq.raw_seq[3 * c + 1] == '-' && seq.raw_seq[3 * c + 2] == '-';
        }
        return seq.codons[c] < 0;
    }

    static std::vector<IndelWindow> detect_indel_windows(
        const Alignment& aln,
        size_t flank_codons = 2,
        size_t max_window_len = 18
    );

    // =========================================================================
    // Helper: Compute Exact Inside-Outside Leaf Cavity Profiles for a Window
    //   For each codon column c in [start_codon, end_codon) and each leaf taxon t,
    //   computes the exact evolved ancestral distribution P_t(s | Data_{-t, c})
    //   at leaf t's tip (including evolution along branch t) in a single O(N * W) pass!
    // =========================================================================
    static void compute_window_leaf_cavities(
        const Tree& tree,
        const Alignment& aln,
        const MG94Matrix& mg94,
        const std::vector<Matrix>& P_branches,
        size_t start_codon,
        size_t end_codon,
        std::vector<std::vector<Vector>>& out_leaf_cavity, // [col_offset][taxon_idx] -> Vector(S)
        const Matrix* foster_omega = nullptr,
        Scalar conv_factor = 1.0,
        const EmpiricalTwoStratumModel* two_stratum = nullptr
    );

    // =========================================================================
    // Stage 1: Codon Affine-Indel Pair-HMM Transducer (Forward-Backward + Viterbi)
    //   Given a taxon's unaligned in-frame codons (c_1, ..., c_m) of length m < W
    //   inside a window of width W, evaluates ALL valid order-preserving codon
    //   placements into W columns under:
    //     log P(placement) = sum_{w=0}^{W-1} [ emission_log_p(w) + indel_prior_log_p ]
    //   Computes:
    //     1. Exact MAP placement (Viterbi)
    //     2. Exact Forward-Backward posterior probability over placements
    //     3. Per-column posterior occupancy/homology confidence
    // =========================================================================
    static TransducerTaxonResult solve_codon_indel_transducer_for_taxon(
        const Sequence& seq,
        size_t start_codon,
        size_t end_codon,
        size_t taxon_idx,
        const std::vector<std::vector<Vector>>& leaf_cavity, // [w][taxon_idx]
        const JointAlignSettings& settings
    );

    // =========================================================================
    // Stage 1 Driver: Inside-Outside Window Realignment EM
    // =========================================================================
    static std::vector<WindowRealignmentSummary> run_stage1_window_realignment(
        Alignment& aln,
        const Tree& tree,
        const MG94Parameters& mg_params,
        const JointAlignSettings& settings,
        std::vector<Scalar>& out_site_homology_posteriors,
        std::vector<std::vector<TransducerTaxonResult>>& out_window_taxon_results
    );

    // =========================================================================
    // Stage 2: Coupled Ohmic SPR / Port-Slide with Boundary Indel Transducer Updates
    //   When testing an Ohmic topological move (pruning clade C_u -> regrafting at v
    //   or sliding entry port u -> x), clades that contain gapped sequences in indel
    //   windows can be penalized by their old guide-tree alignment. Here we couple
    //   each candidate topological move with a fast local boundary realignment of C_u!
    // =========================================================================
    static std::vector<CoupledMoveEvent> run_stage2_coupled_topology_search(
        Tree& tree,
        Alignment& aln,
        const MG94Parameters& mg_params,
        const std::vector<WindowRealignmentSummary>& windows,
        const JointAlignSettings& settings
    );

    // =========================================================================
    // Stage 3: Alignment-Marginalized FEL (Selection Inference Integrated Over
    //   Posterior Indel Placements & Homology Confidence)
    // =========================================================================
    static std::vector<MarginalizedSiteResult> run_stage3_marginalized_fel(
        const Tree& tree,
        const Alignment& initial_aln,
        const Alignment& map_aln,
        const MG94Parameters& mg_params,
        const std::vector<WindowRealignmentSummary>& windows,
        const std::vector<std::vector<TransducerTaxonResult>>& window_taxon_results,
        const std::vector<Scalar>& site_homology_posteriors,
        const JointAlignSettings& settings,
        size_t& out_artifacts_eliminated,
        size_t& out_selection_rescued
    );

    // =========================================================================
    // Helper: Format Stockholm 1.0 Alignment with Per-Column (#=GC PP_cons) &
    //   Per-Residue (#=GR <taxon> PP) Posterior Homology Confidence Tracks
    // =========================================================================
    static char posterior_to_pp_char(Scalar p);

    static std::string format_stockholm_alignment(
        const Alignment& aln,
        const std::vector<Scalar>& site_posteriors,
        const std::vector<WindowRealignmentSummary>& windows,
        const std::vector<std::vector<TransducerTaxonResult>>& window_taxon_results,
        const std::string& extended_newick
    );

    // =========================================================================
    // Helper: Build Partial Order Alignment (POA) Codon Graph in GFA v1.0 & DOT
    //   - Conserved anchor regions form linear backbone segments OR column nodes
    //   - Indel-ambiguous windows expand into parallel alternative indel bubble
    //     paths weighted by their exact transducer posterior probabilities!
    // =========================================================================
    static std::pair<std::string, std::string> build_poa_gfa_and_dot(
        const Alignment& aln,
        const std::vector<Scalar>& site_posteriors,
        const std::vector<WindowRealignmentSummary>& windows,
        const std::vector<std::vector<TransducerTaxonResult>>& window_taxon_results,
        const std::vector<MarginalizedSiteResult>& site_results
    );

    // =========================================================================
    // Full 3-Stage Execution Pipeline
    // =========================================================================
    JointAlignResult run(const Alignment& input_aln, std::optional<Tree> starting_tree = std::nullopt);

    // =========================================================================
    // Phase A & B: Full De Novo Joint Inference from Completely Unaligned
    //   Homologous Coding Sequences (Unaligned FASTA/Vectors -> Ohmic Scaffold
    //   -> Bottom-Up Phylogenetic MG94 Log-Odds Codon Transducer -> Stages 1-3)
    // =========================================================================
    static std::vector<Sequence> strip_gaps_to_unaligned(const Alignment& aln);

    // =========================================================================
    // Subquadratic O(N L) Dayhoff-6 & Amino-Acid k-mer Bit-Sketch Preprocessing
    // and Farthest-Point Landmark + k-NN Graph Construction for Large Alignments
    // =========================================================================
    struct LandmarkKmerGraph {
        size_t num_taxa = 0;
        size_t num_landmarks = 0;
        std::vector<size_t> landmarks;                  // K landmark indices in [0, N)
        std::vector<bool> is_landmark;                  // size N
        std::vector<std::array<Scalar, 20>> aa_freqs;   // size N: amino-acid composition vector f_u(A)
        std::vector<std::vector<size_t>> sparse_adj;    // size N: sorted neighbor indices v > u in E_sparse

        static int aa_to_index20(char aa) {
            switch (aa) {
                case 'A': return 0;  case 'R': return 1;  case 'N': return 2;  case 'D': return 3;
                case 'C': return 4;  case 'Q': return 5;  case 'E': return 6;  case 'G': return 7;
                case 'H': return 8;  case 'I': return 9;  case 'L': return 10; case 'K': return 11;
                case 'M': return 12; case 'F': return 13; case 'P': return 14; case 'S': return 15;
                case 'T': return 16; case 'W': return 17; case 'Y': return 18; case 'V': return 19;
                default:  return -1;
            }
        }

        static int aa_to_dayhoff6(char aa) {
            switch (aa) {
                case 'A': case 'G': case 'P': case 'S': case 'T': return 0;
                case 'D': case 'E': case 'N': case 'Q':           return 1;
                case 'H': case 'K': case 'R':                     return 2;
                case 'I': case 'L': case 'M': case 'V':           return 3;
                case 'F': case 'W': case 'Y':                     return 4;
                case 'C':                                         return 5;
                default:                                          return -1;
            }
        }

        static Scalar sketch_jaccard(
            const std::array<uint64_t, 128>& a,
            const std::array<uint64_t, 128>& b
        ) {
            size_t inter = 0, uni = 0;
            for (size_t w = 0; w < 128; ++w) {
                inter += static_cast<size_t>(__builtin_popcountll(a[w] & b[w]));
                uni   += static_cast<size_t>(__builtin_popcountll(a[w] | b[w]));
            }
            return (uni > 0) ? (static_cast<Scalar>(inter) / static_cast<Scalar>(uni)) : 0.0;
        }

        Scalar expected_aa_identity(size_t u, size_t v) const {
            Scalar dot = 0.0;
            for (int a = 0; a < 20; ++a) {
                dot += aa_freqs[u][a] * aa_freqs[v][a];
            }
            return std::clamp<Scalar>(dot, 0.04, 0.25);
        }

        static LandmarkKmerGraph build(
            const std::vector<std::vector<int8_t>>& taxon_codons,
            const GeneticCode& gcode,
            size_t knn_degree = 8
        ) {
            LandmarkKmerGraph g;
            size_t N = taxon_codons.size();
            g.num_taxa = N;
            g.is_landmark.assign(N, false);
            g.aa_freqs.resize(N);
            g.sparse_adj.resize(N);
            if (N == 0) return g;

            std::vector<std::array<uint64_t, 128>> sketches(N);
            std::vector<size_t> lengths(N, 0);
            for (size_t u = 0; u < N; ++u) {
                sketches[u].fill(0ULL);
                g.aa_freqs[u].fill(0.0);
                const auto& seq = taxon_codons[u];
                lengths[u] = seq.size();
                std::vector<int> aa_idx;
                std::vector<int> d6_idx;
                aa_idx.reserve(seq.size());
                d6_idx.reserve(seq.size());
                size_t valid_aa = 0;
                for (int8_t c : seq) {
                    if (c >= 0 && c < gcode.num_sense_codons) {
                        char aa = gcode.sense_codon_aa[c];
                        int ai = aa_to_index20(aa);
                        int di = aa_to_dayhoff6(aa);
                        aa_idx.push_back(ai);
                        d6_idx.push_back(di);
                        if (ai >= 0) {
                            g.aa_freqs[u][static_cast<size_t>(ai)] += 1.0;
                            valid_aa++;
                        }
                    } else {
                        aa_idx.push_back(-1);
                        d6_idx.push_back(-1);
                    }
                }
                if (valid_aa > 0) {
                    for (int a = 0; a < 20; ++a) {
                        g.aa_freqs[u][static_cast<size_t>(a)] /= static_cast<Scalar>(valid_aa);
                    }
                } else {
                    g.aa_freqs[u].fill(0.05);
                }

                // Words 0..63: exact AA 3-mer bits
                for (size_t i = 0; i + 2 < aa_idx.size(); ++i) {
                    if (aa_idx[i] >= 0 && aa_idx[i + 1] >= 0 && aa_idx[i + 2] >= 0) {
                        uint32_t code = static_cast<uint32_t>(aa_idx[i] * 400 + aa_idx[i + 1] * 20 + aa_idx[i + 2]);
                        uint32_t h = (code * 2654435761u) >> 20; // 0..4095
                        sketches[u][h >> 6] |= (1ULL << (h & 63));
                    }
                }
                // Words 64..127: Dayhoff-6 physicochemical 5-mer bits
                for (size_t i = 0; i + 4 < d6_idx.size(); ++i) {
                    if (d6_idx[i] >= 0 && d6_idx[i + 1] >= 0 && d6_idx[i + 2] >= 0 &&
                        d6_idx[i + 3] >= 0 && d6_idx[i + 4] >= 0) {
                        uint32_t code = static_cast<uint32_t>(
                            d6_idx[i] * 1296 + d6_idx[i + 1] * 216 + d6_idx[i + 2] * 36 + d6_idx[i + 3] * 6 + d6_idx[i + 4]
                        );
                        uint32_t h = (code * 2654435761u) >> 20; // 0..4095
                        sketches[u][64 + (h >> 6)] |= (1ULL << (h & 63));
                    }
                }
            }

            // Select K = min(N, max(32, 2 * ceil(sqrt(N)))) landmarks via farthest-point traversal
            size_t K = std::min<size_t>(N, std::max<size_t>(32, 2 * static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(N))))));
            std::vector<size_t> sorted_lens = lengths;
            std::sort(sorted_lens.begin(), sorted_lens.end());
            size_t med_len = sorted_lens[N / 2];

            size_t first_lm = 0;
            size_t best_len_diff = std::numeric_limits<size_t>::max();
            for (size_t u = 0; u < N; ++u) {
                size_t diff = (lengths[u] >= med_len) ? (lengths[u] - med_len) : (med_len - lengths[u]);
                if (diff < best_len_diff) {
                    best_len_diff = diff;
                    first_lm = u;
                }
            }

            g.landmarks.push_back(first_lm);
            g.is_landmark[first_lm] = true;
            std::vector<Scalar> max_sim_to_lm(N, 0.0);
            for (size_t u = 0; u < N; ++u) {
                max_sim_to_lm[u] = sketch_jaccard(sketches[u], sketches[first_lm]);
            }

            while (g.landmarks.size() < K) {
                size_t next_lm = N;
                Scalar min_max_sim = 2.0;
                for (size_t u = 0; u < N; ++u) {
                    if (g.is_landmark[u] || lengths[u] == 0) continue;
                    if (max_sim_to_lm[u] < min_max_sim) {
                        min_max_sim = max_sim_to_lm[u];
                        next_lm = u;
                    }
                }
                if (next_lm >= N) break;
                g.landmarks.push_back(next_lm);
                g.is_landmark[next_lm] = true;
                for (size_t u = 0; u < N; ++u) {
                    if (!g.is_landmark[u]) {
                        max_sim_to_lm[u] = std::max(max_sim_to_lm[u], sketch_jaccard(sketches[u], sketches[next_lm]));
                    }
                }
            }
            g.num_landmarks = g.landmarks.size();

            // Build symmetric sparse adjacency list (all taxon-landmark edges + top knn_degree k-mer neighbors per taxon)
            std::vector<std::vector<size_t>> knn_lists(N);
            #pragma omp parallel for schedule(static)
            for (size_t u = 0; u < N; ++u) {
                std::vector<std::pair<Scalar, size_t>> cand;
                cand.reserve(N);
                for (size_t v = 0; v < N; ++v) {
                    if (u == v) continue;
                    cand.push_back({sketch_jaccard(sketches[u], sketches[v]), v});
                }
                size_t k_take = std::min(knn_degree, cand.size());
                std::partial_sort(cand.begin(), cand.begin() + static_cast<ptrdiff_t>(k_take), cand.end(),
                    [](const auto& a, const auto& b) {
                        if (a.first != b.first) return a.first > b.first;
                        return a.second < b.second;
                    });
                knn_lists[u].reserve(k_take);
                for (size_t k = 0; k < k_take; ++k) {
                    knn_lists[u].push_back(cand[k].second);
                }
            }

            for (size_t u = 0; u < N; ++u) {
                if (g.is_landmark[u]) {
                    for (size_t v = u + 1; v < N; ++v) {
                        g.sparse_adj[u].push_back(v);
                    }
                } else {
                    std::vector<size_t> nbrs;
                    for (size_t lm : g.landmarks) {
                        if (lm > u) nbrs.push_back(lm);
                    }
                    for (size_t v : knn_lists[u]) {
                        if (v > u) nbrs.push_back(v);
                    }
                    for (size_t v = u + 1; v < N; ++v) {
                        for (size_t w : knn_lists[v]) {
                            if (w == u) { nbrs.push_back(v); break; }
                        }
                    }
                    std::sort(nbrs.begin(), nbrs.end());
                    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
                    g.sparse_adj[u] = std::move(nbrs);
                }
            }
            return g;
        }
    };

    static std::pair<Scalar, Scalar> run_semiglobal_banded_dp_pair(
        const std::vector<int8_t>& s_a,
        const std::vector<int8_t>& s_b,
        const GeneticCode& gcode,
        std::vector<Scalar>& prev_M,
        std::vector<Scalar>& curr_M,
        std::vector<Scalar>& prev_D,
        std::vector<Scalar>& curr_D,
        std::vector<size_t>& prev_id_M,
        std::vector<size_t>& curr_id_M,
        std::vector<size_t>& prev_id_D,
        std::vector<size_t>& curr_id_D
    );

    struct PairwiseCodonTracks {
        Scalar dp_similarity = 0.0;
        Scalar aa_identity = 0.0;
        Scalar d_tot = 0.0;
        Scalar d_syn = 0.0;
        Scalar d_ns = 0.0;
        Scalar d_tn93 = 0.0;
    };

    static PairwiseCodonTracks run_semiglobal_multitrack_dp_pair(
        const std::vector<int8_t>& s_a,
        const std::vector<int8_t>& s_b,
        const std::string& raw_a,
        const std::string& raw_b,
        const GeneticCode& gcode,
        const Vector4& pi
    );

    static std::pair<Alignment, Tree> build_de_novo_codon_alignment_and_tree(
        const std::vector<Sequence>& unaligned_seqs,
        std::shared_ptr<const GeneticCode> gcode = nullptr,
        const JointAlignSettings& settings = JointAlignSettings{},
        const Matrix* precomputed_dp_similarity = nullptr
    );

    // =========================================================================
    // 4-Gate Input Triage & Homology Admission Filter:
    //   Gate 1: Lexical & Alphabet Gate (rejects protein FASTA, empty/short/high-N sequences)
    //   Gate 2: 6-Frame & Strand ORF Gate (rescues +1/+2 frame shifts & reverse-complements,
    //           quarantines pseudogenes/introns with internal stop codons, aborts if >25% fail)
    //   Gate 3: Pairwise Log-Odds Homology Z-Score & Connected-Component Gate
    //           (quarantines non-homologous rogue taxa or extreme length outliers,
    //            aborts if cohort median Z < min_cohort_median_z or component < 3 taxa)
    //   Gate 4: Post-Transducer MSA Expansion & Core Occupancy Gate (enforced after Phase B)
    // =========================================================================
    static std::string reverse_complement_dna(const std::string& seq);

    static std::pair<std::vector<Sequence>, InputAdmissionReport> triage_and_admit_unaligned_sequences(
        const std::vector<Sequence>& raw_seqs,
        std::shared_ptr<const GeneticCode> gcode = nullptr,
        const JointAlignSettings& settings = JointAlignSettings{}
    );

    static std::vector<Sequence> load_unaligned_fasta(
        const std::string& filepath,
        std::shared_ptr<const GeneticCode> gcode = nullptr
    );

    JointAlignResult run_from_unaligned(
        const std::vector<Sequence>& unaligned_seqs,
        std::shared_ptr<const GeneticCode> gcode = nullptr,
        std::optional<Tree> starting_tree = std::nullopt
    );

    JointAlignResult run_from_unaligned_file(
        const std::string& filepath,
        std::shared_ptr<const GeneticCode> gcode = nullptr
    );

    // =========================================================================
    // Objective Alignment Evaluation Suite (Options 1, 3 & 5):
    //   1. Option 1 (Joint Generative Log-Likelihood):
    //        ln P(S, A | T) = ln L_MG94(A | T) + ln P_TreeIndel(G(A) | T, lam_open, eps_ext)
    //      where ancestral presence/absence is reconstructed via Sankoff DP on T and scored
    //      by an affine birth-death indel process along every branch of T.
    //   2. Option 3 (Leave-One-Taxon-Out Cross-Validation Predictive Log-Likelihood):
    //        sum_{t=1}^N ln P(S_t | A_{-t}, T)
    //      integrated over a normalized Pair-HMM Forward partition function against the
    //      exact Inside-Outside held-out leaf cavity profile P(s | Data_{-t, c}).
    //   3. Option 5 (Posterior Model Weights & Per-Column Homology Confidence):
    //      Computes normalized Bayesian model weights across candidate alignments under both
    //      Option 1 (Joint) and Option 3 (LOTO CV), plus mean/min column homology posteriors.
    // =========================================================================
    struct AlignmentObjectiveEvaluation {
        std::string label;
        size_t num_taxa = 0;
        size_t num_codons = 0;
        size_t total_gap_codons = 0;
        Scalar mg94_sub_log_l = 0.0;
        size_t branch_indel_events = 0;
        size_t branch_indel_codons = 0;
        Scalar branch_indel_log_l = 0.0;
        Scalar joint_log_l = 0.0;          // Option 1: mg94_sub_log_l + branch_indel_log_l
        Scalar loto_cv_log_l = 0.0;        // Option 3: sum_t ln P(S_t | A_{-t}, T)
        Scalar joint_posterior_weight = 0.0; // Option 5: exp(joint - max_joint) / Z
        Scalar loto_posterior_weight = 0.0;  // Option 5: exp(loto - max_loto) / Z
        Scalar mean_column_posterior = 1.0;  // Option 5: mean column homology posterior
        Scalar min_column_posterior = 1.0;
    };

    static std::vector<AlignmentObjectiveEvaluation> evaluate_candidate_alignments(
        const std::vector<std::pair<std::string, Alignment>>& candidates,
        std::optional<Tree> ref_tree = std::nullopt,
        const JointAlignSettings& settings = JointAlignSettings{}
    );
};

} // namespace hyphy::analyses
