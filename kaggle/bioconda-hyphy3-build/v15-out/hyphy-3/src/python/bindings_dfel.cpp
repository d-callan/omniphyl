#include "bindings_registry.hpp"
#include "hyphy/analyses/directional_fel.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace hyphy::core;
using namespace hyphy::analyses;

namespace hyphy::bindings {

void register_dfel(nb::module_ &m) {
    // Directional FEL Results & Analyzer
    nb::class_<DirectionalFELSettings>(m, "DirectionalFELSettings")
        .def(nb::init<>())
        .def_rw("p_value_threshold", &DirectionalFELSettings::p_value_threshold)
        .def_rw("heterogeneity_threshold", &DirectionalFELSettings::heterogeneity_threshold)
        .def_rw("root_strategy", &DirectionalFELSettings::root_strategy)
        .def_rw("num_regimes", &DirectionalFELSettings::num_regimes)
        .def_rw("full_model", &DirectionalFELSettings::full_model);

    nb::class_<TaxonComposition>(m, "TaxonComposition")
        .def_ro("name", &TaxonComposition::name)
        .def_ro("total_codons", &TaxonComposition::total_codons)
        .def_ro("gc_overall", &TaxonComposition::gc_overall)
        .def_ro("gc_cp3", &TaxonComposition::gc_cp3);

    nb::class_<CompositionTestResult>(m, "CompositionTestResult")
        .def_ro("chi2_overall", &CompositionTestResult::chi2_overall)
        .def_ro("df_overall", &CompositionTestResult::df_overall)
        .def_ro("p_value_overall", &CompositionTestResult::p_value_overall)
        .def_ro("chi2_cp3", &CompositionTestResult::chi2_cp3)
        .def_ro("df_cp3", &CompositionTestResult::df_cp3)
        .def_ro("p_value_cp3", &CompositionTestResult::p_value_cp3)
        .def_ro("is_heterogeneous", &CompositionTestResult::is_heterogeneous)
        .def_ro("taxa", &CompositionTestResult::taxa);

    nb::class_<RootCandidate>(m, "RootCandidate")
        .def_ro("rank", &RootCandidate::rank)
        .def_ro("branch_name", &RootCandidate::branch_name)
        .def_ro("is_terminal", &RootCandidate::is_terminal)
        .def_ro("log_l", &RootCandidate::log_l)
        .def_ro("delta_log_l", &RootCandidate::delta_log_l)
        .def_ro("weight", &RootCandidate::weight);

    nb::class_<AutoRootResult>(m, "AutoRootResult")
        .def_ro("chosen_branch_name", &AutoRootResult::chosen_branch_name)
        .def_ro("chosen_log_l", &AutoRootResult::chosen_log_l)
        .def_ro("chosen_weight", &AutoRootResult::chosen_weight)
        .def_ro("candidates", &AutoRootResult::candidates);

    nb::class_<DirectionalRegime>(m, "DirectionalRegime")
        .def_ro("id", &DirectionalRegime::id)
        .def_ro("name", &DirectionalRegime::name)
        .def_ro("psi", &DirectionalRegime::psi)
        .def_ro("mean_gc", &DirectionalRegime::mean_gc)
        .def_ro("branch_count", &DirectionalRegime::branch_count);

    nb::class_<DirectionalSiteResult>(m, "DirectionalSiteResult")
        .def_ro("site_index", &DirectionalSiteResult::site_index)
        .def_ro("alpha", &DirectionalSiteResult::alpha)
        .def_ro("beta", &DirectionalSiteResult::beta)
        .def_ro("alpha_null", &DirectionalSiteResult::alpha_null)
        .def_ro("lrt", &DirectionalSiteResult::lrt)
        .def_ro("p_value", &DirectionalSiteResult::p_value)
        .def_ro("total_branch_length", &DirectionalSiteResult::total_branch_length);

    nb::class_<DirectionalFELAnalyzer>(m, "DirectionalFELAnalyzer")
        .def(nb::init<Tree, Alignment, MG94Parameters, DirectionalFELSettings>(),
             "tree"_a, "alignment"_a, "params"_a = MG94Parameters{}, "settings"_a = DirectionalFELSettings{})
        .def_static("run_composition_test", &DirectionalFELAnalyzer::run_composition_test, "alignment"_a, "threshold"_a = 0.05)
        .def_static("create_and_fit", [](Tree tree, Alignment aln, const DirectionalFELSettings& settings) {
            return DirectionalFELAnalyzer::create_and_fit(std::move(tree), std::move(aln), settings);
        }, "tree"_a, "alignment"_a, "settings"_a = DirectionalFELSettings{})
        .def("run", &DirectionalFELAnalyzer::run, "show_progress"_a = false, "force_progress"_a = false)
        .def("to_modern_json", [](const DirectionalFELAnalyzer& dfel, const std::string& input_filepath, const std::string& tree_string) {
            return dfel.to_modern_json(input_filepath, tree_string).dump();
        }, "input_filepath"_a = "", "tree_string"_a = "")
        .def("to_legacy_json", [](const DirectionalFELAnalyzer& dfel, const std::string& input_filepath, const std::string& tree_string) {
            return dfel.to_legacy_json(input_filepath, tree_string).dump();
        }, "input_filepath"_a = "", "tree_string"_a = "")
        .def_ro("comp_test", &DirectionalFELAnalyzer::comp_test)
        .def_ro("root_result", &DirectionalFELAnalyzer::root_result)
        .def_ro("regimes", &DirectionalFELAnalyzer::regimes)
        .def_ro("branch_regime_ids", &DirectionalFELAnalyzer::branch_regime_ids)
        .def_ro("branch_regime_weights", &DirectionalFELAnalyzer::branch_regime_weights)
        .def_ro("site_results", &DirectionalFELAnalyzer::site_results);
}

} // namespace hyphy::bindings
