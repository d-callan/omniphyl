#include "bindings_registry.hpp"
#include "hyphy/analyses/fel.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace hyphy::core;
using namespace hyphy::analyses;

namespace hyphy::bindings {

void register_fel(nb::module_ &m) {
    // FEL Settings, Results & Analyzer
    nb::class_<FELSettings>(m, "FELSettings")
        .def(nb::init<>())
        .def_rw("p_value_threshold", &FELSettings::p_value_threshold)
        .def_rw("full_model", &FELSettings::full_model)
        .def_rw("srv", &FELSettings::srv)
        .def_rw("test_branches", &FELSettings::test_branches)
        .def_rw("test_branch_regex", &FELSettings::test_branch_regex)
        .def_rw("test_branch_names", &FELSettings::test_branch_names);

    nb::class_<SubstitutionEvent>(m, "SubstitutionEvent")
        .def_ro("node_id", &SubstitutionEvent::node_id)
        .def_ro("branch_name", &SubstitutionEvent::branch_name)
        .def_ro("is_tested", &SubstitutionEvent::is_tested)
        .def_ro("branch_length", &SubstitutionEvent::branch_length)
        .def_ro("parent_codon", &SubstitutionEvent::parent_codon)
        .def_ro("child_codon", &SubstitutionEvent::child_codon)
        .def_prop_ro("parent_aa", [](const SubstitutionEvent& ev) { return std::string(1, ev.parent_aa); })
        .def_prop_ro("child_aa", [](const SubstitutionEvent& ev) { return std::string(1, ev.child_aa); })
        .def_ro("nt_distance", &SubstitutionEvent::nt_distance)
        .def_ro("is_synonymous", &SubstitutionEvent::is_synonymous)
        .def_ro("transition_prob", &SubstitutionEvent::transition_prob)
        .def_ro("surprise_bits", &SubstitutionEvent::surprise_bits);

    nb::class_<SiteSubstitutions>(m, "SiteSubstitutions")
        .def_ro("site_index", &SiteSubstitutions::site_index)
        .def_ro("root_codon", &SiteSubstitutions::root_codon)
        .def_prop_ro("root_aa", [](const SiteSubstitutions& s) { return std::string(1, s.root_aa); })
        .def_ro("events", &SiteSubstitutions::events)
        .def_ro("syn_test", &SiteSubstitutions::syn_test)
        .def_ro("nonsyn_test", &SiteSubstitutions::nonsyn_test)
        .def_ro("syn_bg", &SiteSubstitutions::syn_bg)
        .def_ro("nonsyn_bg", &SiteSubstitutions::nonsyn_bg)
        .def_ro("subs_1nt", &SiteSubstitutions::subs_1nt)
        .def_ro("subs_2nt", &SiteSubstitutions::subs_2nt)
        .def_ro("subs_3nt", &SiteSubstitutions::subs_3nt)
        .def_ro("mean_surprise_bits", &SiteSubstitutions::mean_surprise_bits)
        .def_ro("total_surprise_bits", &SiteSubstitutions::total_surprise_bits)
        .def_ro("mean_branch_length", &SiteSubstitutions::mean_branch_length)
        .def_ro("site_perplexity", &SiteSubstitutions::site_perplexity)
        .def_ro("summary_string", &SiteSubstitutions::summary_string)
        .def_ro("summary_string_partitioned", &SiteSubstitutions::summary_string_partitioned);

    nb::class_<SiteResult>(m, "FELSiteResult")
        .def_ro("site_index", &SiteResult::site_index)
        .def_ro("alpha", &SiteResult::alpha)
        .def_ro("beta", &SiteResult::beta)
        .def_ro("beta_nuisance", &SiteResult::beta_nuisance)
        .def_ro("beta_background", &SiteResult::beta_nuisance)
        .def_ro("alpha_null", &SiteResult::alpha_null)
        .def_ro("lrt", &SiteResult::lrt)
        .def_ro("p_value", &SiteResult::p_value)
        .def_ro("total_branch_length", &SiteResult::total_branch_length)
        .def_ro("log_l_alt", &SiteResult::log_l_alt)
        .def_ro("log_l_null", &SiteResult::log_l_null)
        .def_ro("subs", &SiteResult::subs);

    nb::class_<FELAnalyzer>(m, "FELAnalyzer")
        .def("__init__", [](FELAnalyzer* self, Tree tree, Alignment aln, const MG94Parameters& params, const FELSettings& settings) {
            new (self) FELAnalyzer(std::move(tree), std::move(aln), params, settings);
        }, "tree"_a, "alignment"_a, "params"_a = MG94Parameters{}, "settings"_a = FELSettings{})
        .def_static("create_and_fit", [](Tree tree, Alignment aln, const FELSettings& settings) {
            return FELAnalyzer::create_and_fit(std::move(tree), std::move(aln), settings);
        }, "tree"_a, "alignment"_a, "settings"_a = FELSettings{})
        .def_static("create_and_fit", [](Tree tree, Alignment aln, Scalar pvalue_threshold, bool full_model) {
            return FELAnalyzer::create_and_fit(std::move(tree), std::move(aln), pvalue_threshold, nullptr, full_model);
        }, "tree"_a, "alignment"_a, "pvalue_threshold"_a = 0.1, "full_model"_a = true)
        .def("run", &FELAnalyzer::run, "show_progress"_a = false, "force_progress"_a = false)
        .def("to_json", [](const FELAnalyzer& fel, const std::string& fmt) {
            auto format = (fmt == "legacy") ? JSONFormat::Legacy : JSONFormat::ModernV3;
            return fel.to_json("", "", format).dump();
        }, "format"_a = "modern")
        .def("to_legacy_json", [](const FELAnalyzer& fel) {
            return fel.to_legacy_json().dump();
        })
        .def("to_modern_json", [](const FELAnalyzer& fel) {
            return fel.to_modern_json().dump();
        })
        .def_ro("global_log_l", &FELAnalyzer::global_log_l)
        .def_ro("global_aicc", &FELAnalyzer::global_aicc)
        .def_ro("site_results", &FELAnalyzer::site_results)
        .def_ro("num_tested", &FELAnalyzer::num_tested)
        .def_ro("num_background", &FELAnalyzer::num_background)
        .def_ro("test_branch_names", &FELAnalyzer::test_branch_names)
        .def_ro("bg_branch_names", &FELAnalyzer::bg_branch_names)
        .def_ro("is_test_branch", &FELAnalyzer::is_test_branch)
        .def_ro("settings", &FELAnalyzer::settings);
}

} // namespace hyphy::bindings
