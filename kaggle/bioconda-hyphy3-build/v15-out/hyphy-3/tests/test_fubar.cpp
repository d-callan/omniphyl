#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "hyphy/core/types.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/rate_matrix.hpp"
#include "hyphy/analyses/fubar.hpp"
#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <set>

using namespace hyphy::core;
using namespace hyphy::analyses;

inline std::string resolve_test_path(const std::string& path) {
    if (std::ifstream(path).good()) return path;
    if (std::ifstream("../" + path).good()) return "../" + path;
    if (std::ifstream("../../" + path).good()) return "../../" + path;
    return path;
}

inline double pearson_corr(const std::vector<double>& x, const std::vector<double>& y) {
    size_t n = x.size();
    if (n == 0 || n != y.size()) return 0.0;
    double mx = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double my = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double num = 0.0, den_x = 0.0, den_y = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - mx;
        double dy = y[i] - my;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    return (den_x > 0.0 && den_y > 0.0) ? (num / std::sqrt(den_x * den_y)) : 0.0;
}

TEST_CASE("FUBAR Grid Generation and Boundary Math") {
    SUBCASE("Default 20x20 Grid") {
        auto grid = FUBARAnalyzer::create_grid(20, false);
        REQUIRE(grid.size() == 400);

        // Point 0: (0.0, 0.0)
        CHECK(grid[0].first == doctest::Approx(0.0));
        CHECK(grid[0].second == doctest::Approx(0.0));

        // Grid point 273 is (1.0, 1.0) since u[13] = 1.0 and 13 * 20 + 13 = 273
        CHECK(grid[273].first == doctest::Approx(1.0));
        CHECK(grid[273].second == doctest::Approx(1.0));

        // Last point is (50.0, 50.0)
        CHECK(grid[399].first == doctest::Approx(50.0));
        CHECK(grid[399].second == doctest::Approx(50.0));
    }

    SUBCASE("Non-zero Grid") {
        auto grid = FUBARAnalyzer::create_grid(20, true);
        REQUIRE(grid.size() == 400);
        CHECK(grid[0].first >= 1e-3);
    }
}

TEST_CASE("FUBAR False Positive CI DP Convolution") {
    // 13 error probabilities from CD2 ground truth
    std::vector<Scalar> p_err = {
        0.08795, 0.09395, 0.04918, 0.04200, 0.09388, 0.08800,
        0.09650, 0.03130, 0.05490, 0.08310, 0.07630, 0.05600, 0.09580
    };

    Scalar sum_err = std::accumulate(p_err.begin(), p_err.end(), 0.0);
    CHECK(sum_err == doctest::Approx(0.9488).epsilon(0.01));

    auto ci = FUBARAnalyzer::compute_enfp_ci(p_err, 0.05);
    CHECK(ci.first == 0);
    CHECK(ci.second == 3);
}

TEST_CASE("FUBAR Parity on CD2 Dataset against HyPhy 2.5 Ground Truth") {
    std::string aln_path = resolve_test_path("benchmarks/data/cd2.fna");
    std::string tree_path = resolve_test_path("benchmarks/data/cd2.nwk");
    
    REQUIRE(std::ifstream(aln_path).good());
    REQUIRE(std::ifstream(tree_path).good());

    auto aln = Alignment::from_fasta(aln_path);
    auto tree = Tree::from_newick_file(tree_path);

    FUBARSettings settings;
    settings.grid_points = 20;
    settings.method = FUBARMethod::VariationalBayes;
    settings.concentration = 0.5;
    settings.posterior_threshold = 0.90;

    FUBARAnalyzer fubar = FUBARAnalyzer::create_and_fit(tree, aln, settings);
    FUBARResult res = fubar.run(nullptr, false);

    CHECK(res.site_results.size() == 187);

    // Expected significant positive sites from HyPhy 2.5:
    // Sites: 7, 9, 34, 55, 56, 64, 81, 113, 117, 126, 141, 149, 172 (13 sites)
    std::set<size_t> expected_pos = {7, 9, 34, 55, 56, 64, 81, 113, 117, 126, 141, 149, 172};
    std::set<size_t> actual_pos(res.positive_sites.begin(), res.positive_sites.end());

    std::cout << "[FUBAR CD2 Test]\n"
              << "  Detected positive sites: " << res.positive_sites.size() << " (expected 13)\n"
              << "  Expected false positives: " << res.expected_false_positives << " (95% CI: "
              << res.false_positives_ci_95.first << "-" << res.false_positives_ci_95.second << ")\n"
              << "  Grid execution time: " << res.time_grid_sec << "s\n"
              << "  Posterior inference time: " << res.time_posterior_sec << "s\n"
              << "  Total analysis time: " << res.time_total_sec << "s\n";

    // Check that key positive selection sites match
    size_t overlap_count = 0;
    for (size_t s : expected_pos) {
        if (actual_pos.count(s)) overlap_count++;
    }
    CHECK(overlap_count >= 12); // At least 12/13 identical

    // Check site 34 (strongest positive site: prob_pos ~ 0.95, alpha ~ 0.48, beta ~ 2.13)
    const auto& site34 = res.site_results[33];
    CHECK(site34.site == 34);
    CHECK(site34.prob_pos > 0.90);
    CHECK(site34.beta > site34.alpha);
    CHECK(site34.classification == "diversifying");

    // Check site 11 (strong purifying site: prob_neg ~ 0.94, alpha ~ 2.57, beta ~ 0.19)
    const auto& site11 = res.site_results[10];
    CHECK(site11.site == 11);
    CHECK(site11.prob_neg > 0.90);
    CHECK(site11.alpha > site11.beta);
    CHECK(site11.classification == "purifying");

    // JSON export validation
    std::string modern_json = fubar.to_modern_json(res);
    nlohmann::json parsed_modern = nlohmann::json::parse(modern_json);
    CHECK(parsed_modern["analysis"]["id"] == "fubar");
    CHECK(parsed_modern["site_results"]["data"]["site"].size() == 187);

    std::string legacy_json = fubar.to_legacy_json(res);
    nlohmann::json parsed_legacy = nlohmann::json::parse(legacy_json);
    CHECK(parsed_legacy["MLE"]["content"]["0"].size() == 187);
    CHECK(parsed_legacy["grid"].size() == 400);
}

TEST_CASE("FUBAR Collapsed Gibbs Sampler Test") {
    std::string aln_path = resolve_test_path("benchmarks/data/cd2.fna");
    std::string tree_path = resolve_test_path("benchmarks/data/cd2.nwk");

    auto aln = Alignment::from_fasta(aln_path);
    auto tree = Tree::from_newick_file(tree_path);

    FUBARSettings settings;
    settings.grid_points = 20;
    settings.method = FUBARMethod::CollapsedGibbs;
    settings.mcmc_chain_length = 5000;
    settings.mcmc_burn_in = 1000;
    settings.mcmc_samples = 50;

    FUBARAnalyzer fubar = FUBARAnalyzer::create_and_fit(tree, aln, settings);
    FUBARResult res = fubar.run(nullptr, false);

    CHECK(res.site_results.size() == 187);
    CHECK(res.posterior_grid_weights.size() == 400);

    // Grid weights should sum to 1.0
    Scalar sum_weights = std::accumulate(res.posterior_grid_weights.begin(), res.posterior_grid_weights.end(), 0.0);
    CHECK(sum_weights == doctest::Approx(1.0).epsilon(1e-4));
}
