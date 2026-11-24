#include "bindings_registry.hpp"
#include "hyphy/analyses/fubar.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace hyphy::core;
using namespace hyphy::analyses;

namespace hyphy::bindings {

void register_fubar(nb::module_ &m) {
    // FUBAR
    nb::enum_<FUBARMethod>(m, "FUBARMethod")
        .value("VariationalBayes", FUBARMethod::VariationalBayes)
        .value("CollapsedGibbs", FUBARMethod::CollapsedGibbs)
        .value("MetropolisHastings", FUBARMethod::MetropolisHastings);

    nb::class_<FUBARSettings>(m, "FUBARSettings")
        .def(nb::init<>())
        .def_rw("grid_points", &FUBARSettings::grid_points)
        .def_rw("method", &FUBARSettings::method)
        .def_rw("concentration", &FUBARSettings::concentration)
        .def_rw("posterior_threshold", &FUBARSettings::posterior_threshold)
        .def_rw("non_zero", &FUBARSettings::non_zero)
        .def_rw("tree_scaling", &FUBARSettings::tree_scaling)
        .def_rw("mcmc_chains", &FUBARSettings::mcmc_chains)
        .def_rw("mcmc_chain_length", &FUBARSettings::mcmc_chain_length)
        .def_rw("mcmc_burn_in", &FUBARSettings::mcmc_burn_in)
        .def_rw("mcmc_samples", &FUBARSettings::mcmc_samples);

    nb::class_<FUBARSiteResult>(m, "FUBARSiteResult")
        .def_ro("site", &FUBARSiteResult::site)
        .def_ro("alpha", &FUBARSiteResult::alpha)
        .def_ro("beta", &FUBARSiteResult::beta)
        .def_ro("beta_minus_alpha", &FUBARSiteResult::beta_minus_alpha)
        .def_ro("prob_neg", &FUBARSiteResult::prob_neg)
        .def_ro("prob_pos", &FUBARSiteResult::prob_pos)
        .def_ro("bayes_factor", &FUBARSiteResult::bayes_factor)
        .def_ro("classification", &FUBARSiteResult::classification)
        .def_ro("posterior_weights", &FUBARSiteResult::posterior_weights);

    nb::class_<FUBARResult>(m, "FUBARResult")
        .def_ro("site_results", &FUBARResult::site_results)
        .def_ro("grid", &FUBARResult::grid)
        .def_ro("posterior_grid_weights", &FUBARResult::posterior_grid_weights)
        .def_ro("positive_sites", &FUBARResult::positive_sites)
        .def_ro("purifying_sites", &FUBARResult::purifying_sites)
        .def_ro("expected_false_positives", &FUBARResult::expected_false_positives)
        .def_ro("false_positives_ci_95", &FUBARResult::false_positives_ci_95)
        .def_ro("weight_non_positive", &FUBARResult::weight_non_positive)
        .def_ro("time_total_sec", &FUBARResult::time_total_sec);

    nb::class_<FUBARAnalyzer>(m, "FUBARAnalyzer")
        .def(nb::init<Tree, Alignment, MG94Parameters, FUBARSettings>(),
             "tree"_a, "alignment"_a, "params"_a = MG94Parameters{}, "settings"_a = FUBARSettings{})
        .def_static("create_and_fit", [](Tree tree, Alignment aln, const FUBARSettings& settings) {
            return FUBARAnalyzer::create_and_fit(std::move(tree), std::move(aln), settings, nullptr);
        }, "tree"_a, "alignment"_a, "settings"_a = FUBARSettings{})
        .def("run", [](FUBARAnalyzer& fubar, bool verbose) {
            return fubar.run(nullptr, verbose);
        }, "verbose"_a = false)
        .def("to_modern_json", &FUBARAnalyzer::to_modern_json, "result"_a, "cli_cmd"_a = "hyphy3 fubar", "input_filepath"_a = "")
        .def("to_legacy_json", &FUBARAnalyzer::to_legacy_json, "result"_a, "input_filepath"_a = "");
}

} // namespace hyphy::bindings
