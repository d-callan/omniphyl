#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "hyphy/core/types.hpp"
#include "hyphy/core/genetic_code.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/rate_matrix.hpp"
#include "hyphy/analyses/slac.hpp"

#include <iostream>
#include <fstream>
#include <cmath>

using namespace hyphy::core;
using namespace hyphy::analyses;

inline std::string resolve_test_path(const std::string& path) {
    if (std::ifstream(path).good()) return path;
    if (std::ifstream("../" + path).good()) return "../" + path;
    if (std::ifstream("../../" + path).good()) return "../../" + path;
    return path;
}

TEST_CASE("SLAC: Extended Binomial Tail Calculation") {
    // Edge cases
    CHECK(extended_bin_tail(10.0, 0.0, 5.0) == 0.0);
    CHECK(extended_bin_tail(10.0, 1.0, 5.0) == 1.0);
    CHECK(extended_bin_tail(10.0, 0.5, -1.0) == 0.0);
    CHECK(extended_bin_tail(10.0, 0.5, 10.0) == 1.0);
    CHECK(extended_bin_tail(10.0, 0.5, 15.0) == 1.0);

    // Exact parity with adh.nex reference site 1 (S=1, N=0, total=1, p=0.2891330861395575)
    double p_tail_site1 = extended_bin_tail(1.0, 0.2891330861395575, 1.0);
    CHECK(std::abs(p_tail_site1 - 1.0) < 1e-12);

    double p_tail_site1_neg = 1.0 - extended_bin_tail(1.0, 0.2891330861395575, 0.0);
    CHECK(std::abs(p_tail_site1_neg - 0.2891330861395575) < 1e-12);

    // Site 2 (S=5, N=4, total=9, p=0.2894278487088991)
    double p_pos2 = extended_bin_tail(9.0, 0.2894278487088991, 5.0);
    CHECK(std::abs(p_pos2 - 0.978863222760064) < 1e-6);

    double p_neg2 = 1.0 - extended_bin_tail(9.0, 0.2894278487088991, 4.0);
    CHECK(std::abs(p_neg2 - 0.086375411156225) < 1e-6);
}

TEST_CASE("SLAC: Pairwise Codon Differences & Expected Sites Metric") {
    auto code = GeneticCode::universal();
    CHECK(code->num_sense_codons == 61);

    Vector4 equal_freqs = Vector4::Constant(0.25);
    GTRParameters gtr; // Default equal rates = 1.0

    SLACCodonMetric metric = SLACCodonMetric::compute(*code, equal_freqs, gtr);

    // ATG (Methionine) has no synonymous single-nucleotide mutations under Universal code
    int8_t atg_idx = code->triplet_to_sense('A', 'T', 'G');
    REQUIRE(atg_idx >= 0);
    CHECK(std::abs(metric.SS[atg_idx] - 0.0) < 1e-9);
    CHECK(std::abs(metric.NS[atg_idx] - 3.0) < 1e-9);

    // TGG (Tryptophan) has no synonymous single-nucleotide mutations (TGA=stop, TAG=stop)
    int8_t tgg_idx = code->triplet_to_sense('T', 'G', 'G');
    REQUIRE(tgg_idx >= 0);
    CHECK(std::abs(metric.SS[tgg_idx] - 0.0) < 1e-9);

    // Diagonal elements of EPS/EPN equal SS/NS
    for (int s = 0; s < code->num_sense_codons; ++s) {
        CHECK(std::abs(metric.EPS(s, s) - metric.SS[s]) < 1e-12);
        CHECK(std::abs(metric.EPN(s, s) - metric.NS[s]) < 1e-12);
        CHECK(metric.OPS(s, s) == 0.0);
        CHECK(metric.OPN(s, s) == 0.0);
    }

    // Symmetry check across all pairs
    for (int s1 = 0; s1 < code->num_sense_codons; ++s1) {
        for (int s2 = s1 + 1; s2 < code->num_sense_codons; ++s2) {
            CHECK(metric.EPS(s1, s2) == metric.EPS(s2, s1));
            CHECK(metric.EPN(s1, s2) == metric.EPN(s2, s1));
            CHECK(metric.OPS(s1, s2) == metric.OPS(s2, s1));
            CHECK(metric.OPN(s1, s2) == metric.OPN(s2, s1));
        }
    }

    // 1-step synonymous change: e.g. AAA (Lys) -> AAG (Lys)
    int8_t aaa_idx = code->triplet_to_sense('A', 'A', 'A');
    int8_t aag_idx = code->triplet_to_sense('A', 'A', 'G');
    REQUIRE(aaa_idx >= 0);
    REQUIRE(aag_idx >= 0);
    CHECK(metric.OPS(aaa_idx, aag_idx) == 1.0);
    CHECK(metric.OPN(aaa_idx, aag_idx) == 0.0);

    // 1-step non-synonymous change: e.g. AAA (Lys) -> AAC (Asn)
    int8_t aac_idx = code->triplet_to_sense('A', 'A', 'C');
    REQUIRE(aac_idx >= 0);
    CHECK(metric.OPS(aaa_idx, aac_idx) == 0.0);
    CHECK(metric.OPN(aaa_idx, aac_idx) == 1.0);
}

TEST_CASE("SLAC: Joint ML Ancestral State Reconstruction & Substitution Counting on CD2") {
    std::string aln_path = resolve_test_path("benchmarks/data/cd2.fna");
    std::string tree_path = resolve_test_path("benchmarks/data/cd2.nwk");

    Alignment aln = Alignment::from_fasta(aln_path);
    Tree tree = Tree::from_newick_file(tree_path);

    SLACSettings settings;
    settings.samples = 0; // Skip sampling for fast test
    settings.p_value_threshold = 0.1;
    settings.full_model = false; // Quick fit

    SLACAnalyzer analyzer = SLACAnalyzer::create_and_fit(tree, aln, settings);
    SLACResult res = analyzer.run();

    CHECK(res.total_sites == 187);
    CHECK(res.resolved_sites.size() == 187);
    CHECK(res.averaged_sites.size() == 187);
    CHECK(res.num_tested_branches == tree.num_nodes() - 1);
    CHECK(res.total_tested_tree_length > 0.0);

    // Check invariants at each site
    for (size_t s = 0; s < res.total_sites; ++s) {
        const auto& r = res.resolved_sites[s];
        CHECK(r.ES >= 0.0);
        CHECK(r.EN >= 0.0);
        CHECK(r.S >= 0.0);
        CHECK(r.N >= 0.0);
        CHECK(r.P_S >= 0.0);
        CHECK(r.P_S <= 1.0);
        CHECK(r.p_positive >= 0.0);
        CHECK(r.p_positive <= 1.0);
        CHECK(r.p_negative >= 0.0);
        CHECK(r.p_negative <= 1.0);
        CHECK(r.total_branch_length > 0.0);
    }

    // Verify Legacy JSON Export
    nlohmann::json legacy = res.to_legacy_json(aln_path, tree.to_newick());
    CHECK(legacy.contains("MLE"));
    CHECK(legacy["MLE"].contains("headers"));
    CHECK(legacy["MLE"]["headers"].size() == 11);
    CHECK(legacy["MLE"].contains("content"));
    CHECK(legacy["MLE"]["content"]["0"]["by-site"]["RESOLVED"].size() == 187);
    CHECK(legacy["MLE"]["content"]["0"]["by-site"]["AVERAGED"].size() == 187);
    CHECK(legacy.contains("branch attributes"));
    CHECK(legacy["branch attributes"].contains("0"));
    CHECK(legacy.contains("fits"));
    CHECK(legacy["fits"].contains("Global MG94xREV"));
    CHECK(legacy["fits"].contains("Nucleotide GTR"));

    // Verify Modern v3 JSON Export
    nlohmann::json modern = res.to_modern_json(aln_path, tree.to_newick());
    CHECK(modern["schema_version"] == "3.0.0");
    CHECK(modern["analysis"]["id"] == "slac");
    CHECK(modern["dataset"]["codon_sites"] == 187);
    CHECK(modern["site_results"]["data"]["site"].size() == 187);
    CHECK(modern["site_results"]["data"]["ES"].size() == 187);
    CHECK(modern["site_results"]["data"]["S"].size() == 187);
    CHECK(modern["site_results"]["data"]["N"].size() == 187);
    CHECK(modern["site_results"]["data"]["p_positive"].size() == 187);
    CHECK(modern["site_results"]["data"]["classification"].size() == 187);
}

TEST_CASE("SLAC: Stochastic Ancestral State Sampling & Quantiles") {
    std::string aln_path = resolve_test_path("benchmarks/data/cd2.fna");
    std::string tree_path = resolve_test_path("benchmarks/data/cd2.nwk");

    Alignment aln = Alignment::from_fasta(aln_path);
    Tree tree = Tree::from_newick_file(tree_path);

    SLACSettings settings;
    settings.samples = 50; // 50 samples
    settings.p_value_threshold = 0.1;
    settings.full_model = false;

    SLACAnalyzer analyzer = SLACAnalyzer::create_and_fit(tree, aln, settings);
    SLACResult res = analyzer.run();

    CHECK(res.total_sites == 187);
    for (size_t s = 0; s < res.total_sites; ++s) {
        const auto& r = res.resolved_sites[s];
        REQUIRE(r.quantiles_median.size() == 11);
        REQUIRE(r.quantiles_2_5.size() == 11);
        REQUIRE(r.quantiles_97_5.size() == 11);

        // Quantile monotonicity: Q_2.5 <= Q_median <= Q_97.5 for S, N, ES, EN
        CHECK(r.quantiles_2_5[0] <= r.quantiles_median[0] + 1e-9); // ES
        CHECK(r.quantiles_median[0] <= r.quantiles_97_5[0] + 1e-9);

        CHECK(r.quantiles_2_5[1] <= r.quantiles_median[1] + 1e-9); // EN
        CHECK(r.quantiles_median[1] <= r.quantiles_97_5[1] + 1e-9);

        CHECK(r.quantiles_2_5[2] <= r.quantiles_median[2] + 1e-9); // S
        CHECK(r.quantiles_median[2] <= r.quantiles_97_5[2] + 1e-9);

        CHECK(r.quantiles_2_5[3] <= r.quantiles_median[3] + 1e-9); // N
        CHECK(r.quantiles_median[3] <= r.quantiles_97_5[3] + 1e-9);
    }

    nlohmann::json legacy = res.to_legacy_json(aln_path, tree.to_newick());
    CHECK(legacy.contains("sample-median"));
    CHECK(legacy.contains("sample-2.5"));
    CHECK(legacy.contains("sample-97.5"));
    CHECK(legacy["sample-median"]["0"]["RESOLVED"].size() == 187);
}

TEST_CASE("SLAC: Numerical Parity with Reference ADH Dataset") {
    std::string ref_json_path = resolve_test_path("tests/data/adh.nex.SLAC.json");
    if (!std::ifstream(ref_json_path).good()) {
        ref_json_path = resolve_test_path("../hyphy/tests/data/adh.nex.SLAC.json");
    }

    if (std::ifstream(ref_json_path).good()) {
        std::ifstream f(ref_json_path);
        nlohmann::json ref_data;
        f >> ref_data;

        std::string aln_path = resolve_test_path("tests/data/adh.nex");
        if (!std::ifstream(aln_path).good()) {
            aln_path = resolve_test_path("../hyphy/tests/data/adh.nex");
        }

        if (std::ifstream(aln_path).good()) {
            Alignment aln = Alignment::from_nexus(aln_path);
            Tree tree = Tree::from_newick(aln.embedded_tree_newick);

            SLACSettings settings;
            settings.samples = 0;
            settings.full_model = false;

            SLACAnalyzer analyzer = SLACAnalyzer::create_and_fit(tree, aln, settings);
            SLACResult res = analyzer.run();

            CHECK(res.total_sites == 254);

            // Compare site 0 (conserved ATG)
            const auto& r0 = res.resolved_sites[0];
            CHECK(r0.S == 0.0);
            CHECK(r0.N == 0.0);
            CHECK(std::abs(r0.ES - 0.0) < 1e-9);
            CHECK(std::abs(r0.EN - 3.0) < 1e-9);
            CHECK(r0.p_positive == 1.0);
            CHECK(r0.p_negative == 1.0);

            // Compare reference row 0
            const auto& ref_row0 = ref_data["MLE"]["content"]["0"]["by-site"]["RESOLVED"][0];
            CHECK(ref_row0[0].get<double>() == 0.0);
            CHECK(ref_row0[1].get<double>() == 3.0);
            CHECK(ref_row0[2].get<double>() == 0.0);
            CHECK(ref_row0[3].get<double>() == 0.0);
            CHECK(ref_row0[8].get<double>() == 1.0);
            CHECK(ref_row0[9].get<double>() == 1.0);
        }
    }
}
