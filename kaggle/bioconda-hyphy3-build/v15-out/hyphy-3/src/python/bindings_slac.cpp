#include "bindings_registry.hpp"
#include "hyphy/analyses/slac.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace hyphy::core;
using namespace hyphy::analyses;

namespace hyphy::bindings {

void register_slac(nb::module_ &m) {
    // SLAC Settings, Results & Analyzer
    nb::class_<SLACSettings>(m, "SLACSettings")
        .def(nb::init<>())
        .def_rw("p_value_threshold", &SLACSettings::p_value_threshold)
        .def_rw("samples", &SLACSettings::samples)
        .def_rw("test_branches", &SLACSettings::test_branches)
        .def_rw("test_branch_regex", &SLACSettings::test_branch_regex)
        .def_rw("test_branch_names", &SLACSettings::test_branch_names)
        .def_rw("full_model", &SLACSettings::full_model)
        .def_rw("quiet", &SLACSettings::quiet)
        .def_rw("all_sites", &SLACSettings::all_sites)
        .def_rw("random_seed", &SLACSettings::random_seed);

    nb::class_<SLACSiteResult>(m, "SLACSiteResult")
        .def_ro("site_index", &SLACSiteResult::site_index)
        .def_ro("ES", &SLACSiteResult::ES)
        .def_ro("EN", &SLACSiteResult::EN)
        .def_ro("S", &SLACSiteResult::S)
        .def_ro("N", &SLACSiteResult::N)
        .def_ro("P_S", &SLACSiteResult::P_S)
        .def_ro("dS", &SLACSiteResult::dS)
        .def_ro("dN", &SLACSiteResult::dN)
        .def_ro("dN_minus_dS", &SLACSiteResult::dN_minus_dS)
        .def_ro("p_positive", &SLACSiteResult::p_positive)
        .def_ro("p_negative", &SLACSiteResult::p_negative)
        .def_ro("total_branch_length", &SLACSiteResult::total_branch_length)
        .def_ro("has_dS", &SLACSiteResult::has_dS)
        .def_ro("has_dN", &SLACSiteResult::has_dN)
        .def_ro("has_dN_minus_dS", &SLACSiteResult::has_dN_minus_dS)
        .def_ro("quantiles_2_5", &SLACSiteResult::quantiles_2_5)
        .def_ro("quantiles_median", &SLACSiteResult::quantiles_median)
        .def_ro("quantiles_97_5", &SLACSiteResult::quantiles_97_5);

    nb::class_<SLACBranchResult>(m, "SLACBranchResult")
        .def_ro("branch_name", &SLACBranchResult::branch_name)
        .def_ro("branch_length", &SLACBranchResult::branch_length)
        .def_ro("ES", &SLACBranchResult::ES)
        .def_ro("EN", &SLACBranchResult::EN)
        .def_ro("S", &SLACBranchResult::S)
        .def_ro("N", &SLACBranchResult::N)
        .def_ro("P_S", &SLACBranchResult::P_S)
        .def_ro("dS", &SLACBranchResult::dS)
        .def_ro("dN", &SLACBranchResult::dN)
        .def_ro("dN_minus_dS", &SLACBranchResult::dN_minus_dS)
        .def_ro("p_positive", &SLACBranchResult::p_positive)
        .def_ro("p_negative", &SLACBranchResult::p_negative)
        .def_ro("has_dS", &SLACBranchResult::has_dS)
        .def_ro("has_dN", &SLACBranchResult::has_dN)
        .def_ro("has_dN_minus_dS", &SLACBranchResult::has_dN_minus_dS);

    nb::class_<SLACResult>(m, "SLACResult")
        .def_ro("settings", &SLACResult::settings)
        .def_ro("baseline_log_l", &SLACResult::baseline_log_l)
        .def_ro("baseline_aicc", &SLACResult::baseline_aicc)
        .def_ro("gtr_log_l", &SLACResult::gtr_log_l)
        .def_ro("gtr_aicc", &SLACResult::gtr_aicc)
        .def_ro("global_omega", &SLACResult::global_omega)
        .def_ro("total_tree_length", &SLACResult::total_tree_length)
        .def_ro("total_tested_tree_length", &SLACResult::total_tested_tree_length)
        .def_ro("num_tested_branches", &SLACResult::num_tested_branches)
        .def_ro("num_background_branches", &SLACResult::num_background_branches)
        .def_ro("tested_branch_names", &SLACResult::tested_branch_names)
        .def_ro("background_branch_names", &SLACResult::background_branch_names)
        .def_ro("resolved_sites", &SLACResult::resolved_sites)
        .def_ro("averaged_sites", &SLACResult::averaged_sites)
        .def_ro("resolved_branches", &SLACResult::resolved_branches)
        .def_ro("averaged_branches", &SLACResult::averaged_branches)
        .def_ro("total_sites", &SLACResult::total_sites)
        .def_ro("diversifying_sites", &SLACResult::diversifying_sites)
        .def_ro("purifying_sites", &SLACResult::purifying_sites)
        .def_ro("neutral_sites", &SLACResult::neutral_sites)
        .def_ro("runtime_seconds", &SLACResult::runtime_seconds)
        .def("to_modern_json", [](const SLACResult& res, const std::string& input_filepath, const std::string& tree_string) {
            return res.to_modern_json(input_filepath, tree_string).dump();
        }, "input_filepath"_a = "", "tree_string"_a = "")
        .def("to_legacy_json", [](const SLACResult& res, const std::string& input_filepath, const std::string& tree_string) {
            return res.to_legacy_json(input_filepath, tree_string).dump();
        }, "input_filepath"_a = "", "tree_string"_a = "");

    nb::class_<SLACAnalyzer>(m, "SLACAnalyzer")
        .def(nb::init<Tree, Alignment, MG94Parameters, SLACSettings>(),
             "tree"_a, "alignment"_a, "params"_a = MG94Parameters{}, "settings"_a = SLACSettings{})
        .def_static("create_and_fit", [](Tree tree, Alignment aln, const SLACSettings& settings) {
            return SLACAnalyzer::create_and_fit(std::move(tree), std::move(aln), settings);
        }, "tree"_a, "alignment"_a, "settings"_a = SLACSettings{})
        .def("run", [](SLACAnalyzer& slac, bool show_progress, bool force_progress) {
            return slac.run(nullptr, show_progress, force_progress);
        }, "show_progress"_a = false, "force_progress"_a = false);
}

} // namespace hyphy::bindings
