#include "bindings_registry.hpp"
#include "hyphy/analyses/prime.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace hyphy::core;
using namespace hyphy::analyses;

namespace hyphy::bindings {

void register_prime(nb::module_ &m) {
    // PRIME
    nb::class_<AminoAcidProperty>(m, "AminoAcidProperty")
        .def_ro("name", &AminoAcidProperty::name)
        .def_ro("description", &AminoAcidProperty::description)
        .def_ro("mean", &AminoAcidProperty::mean)
        .def_ro("std_dev", &AminoAcidProperty::std_dev);

    nb::class_<PropertySiteInference>(m, "PropertySiteInference")
        .def_ro("property_name", &PropertySiteInference::property_name)
        .def_ro("lambda_", &PropertySiteInference::lambda)
        .def_ro("lrt", &PropertySiteInference::lrt)
        .def_ro("p_value", &PropertySiteInference::p_value)
        .def_ro("adjusted_p_value", &PropertySiteInference::adjusted_p_value)
        .def_ro("log_l_null", &PropertySiteInference::log_l_null)
        .def_ro("classification", &PropertySiteInference::classification);

    nb::class_<PRIMESettings>(m, "PRIMESettings")
        .def(nb::init<>())
        .def_rw("p_value_threshold", &PRIMESettings::p_value_threshold)
        .def_rw("stage2_p_value", &PRIMESettings::stage2_p_value)
        .def_rw("property_set", &PRIMESettings::property_set)
        .def_rw("custom_property_file", &PRIMESettings::custom_property_file)
        .def_rw("shrinkage_lambda", &PRIMESettings::shrinkage_lambda)
        .def_rw("max_bound", &PRIMESettings::max_bound)
        .def_rw("full_model", &PRIMESettings::full_model)
        .def_rw("fit_global", &PRIMESettings::fit_global)
        .def_rw("impute_states", &PRIMESettings::impute_states)
        .def_rw("test_branches", &PRIMESettings::test_branches)
        .def_rw("test_branch_regex", &PRIMESettings::test_branch_regex)
        .def_rw("test_branch_names", &PRIMESettings::test_branch_names);

    nb::class_<PRIMESiteResult>(m, "PRIMESiteResult")
        .def_ro("site_index", &PRIMESiteResult::site_index)
        .def_ro("alpha", &PRIMESiteResult::alpha)
        .def_ro("beta", &PRIMESiteResult::beta)
        .def_ro("beta_nuisance", &PRIMESiteResult::beta_nuisance)
        .def_ro("fel_alpha", &PRIMESiteResult::fel_alpha)
        .def_ro("fel_beta", &PRIMESiteResult::fel_beta)
        .def_ro("log_l_alt", &PRIMESiteResult::log_l_alt)
        .def_ro("log_l_fel", &PRIMESiteResult::log_l_fel)
        .def_ro("omnibus_lrt", &PRIMESiteResult::omnibus_lrt)
        .def_ro("omnibus_p_value", &PRIMESiteResult::omnibus_p_value)
        .def_ro("omnibus_q_value", &PRIMESiteResult::omnibus_q_value)
        .def_ro("total_branch_length", &PRIMESiteResult::total_branch_length)
        .def_ro("properties", &PRIMESiteResult::properties)
        .def_ro("nonsyn_subs", &PRIMESiteResult::nonsyn_subs)
        .def_ro("unique_aa", &PRIMESiteResult::unique_aa)
        .def_ro("redundancy_ratio", &PRIMESiteResult::redundancy_ratio)
        .def("is_significant", &PRIMESiteResult::is_significant, "threshold"_a = 0.10);

    nb::class_<PRIMEGlobalResult>(m, "PRIMEGlobalResult")
        .def_ro("log_likelihood", &PRIMEGlobalResult::log_likelihood)
        .def_ro("aicc", &PRIMEGlobalResult::aicc)
        .def_ro("alpha", &PRIMEGlobalResult::alpha)
        .def_ro("beta", &PRIMEGlobalResult::beta)
        .def_ro("lambdas", &PRIMEGlobalResult::lambdas)
        .def_ro("property_names", &PRIMEGlobalResult::property_names)
        .def_ro("baseline_tree_length", &PRIMEGlobalResult::baseline_tree_length);

    nb::class_<PRIMEResult>(m, "PRIMEResult")
        .def_ro("settings", &PRIMEResult::settings)
        .def_ro("properties", &PRIMEResult::properties)
        .def_ro("global_fit", &PRIMEResult::global_fit)
        .def_ro("site_results", &PRIMEResult::site_results)
        .def_ro("total_sites", &PRIMEResult::total_sites)
        .def_ro("invariant_sites", &PRIMEResult::invariant_sites)
        .def_ro("significant_sites", &PRIMEResult::significant_sites)
        .def_ro("conserved_counts", &PRIMEResult::conserved_counts)
        .def_ro("changing_counts", &PRIMEResult::changing_counts)
        .def_ro("runtime_seconds", &PRIMEResult::runtime_seconds)
        .def("to_modern_json", [](const PRIMEResult& res, const Tree& tree, const Alignment& aln, const std::string& cli_cmd, const std::string& input_filepath) {
            return res.to_modern_json(tree, aln, cli_cmd, input_filepath).dump();
        }, "tree"_a, "alignment"_a, "cli_cmd"_a = "hyphy3 prime", "input_filepath"_a = "")
        .def("to_legacy_json", [](const PRIMEResult& res, const Tree& tree, const Alignment& aln, const std::string& input_filepath) {
            return res.to_legacy_json(tree, aln, input_filepath).dump();
        }, "tree"_a, "alignment"_a, "input_filepath"_a = "");

    nb::class_<PRIMEAnalyzer>(m, "PRIMEAnalyzer")
        .def(nb::init<Tree, Alignment, MG94Parameters, PRIMESettings>(),
             "tree"_a, "alignment"_a, "params"_a = MG94Parameters{}, "settings"_a = PRIMESettings{})
        .def_static("create_and_fit", [](Tree tree, Alignment aln, const PRIMESettings& settings) {
            return PRIMEAnalyzer::create_and_fit(std::move(tree), std::move(aln), settings);
        }, "tree"_a, "alignment"_a, "settings"_a = PRIMESettings{})
        .def("run", [](
            PRIMEAnalyzer& prime,
            bool show_progress,
            bool force_progress,
            std::function<void(size_t, const PRIMESiteResult&)> site_cb,
            std::function<void(size_t, size_t, size_t, const PRIMESiteResult&)> live_site_cb
        ) {
            return prime.run(nullptr, show_progress, force_progress, site_cb, live_site_cb);
        }, "show_progress"_a = false, "force_progress"_a = false,
           "site_callback"_a = nb::none(), "live_site_callback"_a = nb::none());
}

} // namespace hyphy::bindings
