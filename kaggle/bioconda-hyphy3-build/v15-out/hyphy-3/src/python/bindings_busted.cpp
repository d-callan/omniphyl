#include "bindings_registry.hpp"
#include "hyphy/analyses/busted.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace hyphy::core;
using namespace hyphy::analyses;

namespace hyphy::bindings {

void register_busted(nb::module_ &m) {
    // BUSTED Results & Analyzer
    nb::class_<BUSTEDSettings>(m, "BUSTEDSettings")
        .def(nb::init<>())
        .def_rw("full_auto", &BUSTEDSettings::full_auto)
        .def_rw("num_rate_classes", &BUSTEDSettings::num_rate_classes)
        .def_rw("srv", &BUSTEDSettings::srv)
        .def_rw("num_syn_rate_classes", &BUSTEDSettings::num_syn_rate_classes)
        .def_rw("auto_select_k", &BUSTEDSettings::auto_select_k)
        .def_rw("max_k", &BUSTEDSettings::max_k)
        .def_rw("refine_branch_lengths", &BUSTEDSettings::refine_branch_lengths)
        .def_rw("multiple_hits", &BUSTEDSettings::multiple_hits)
        .def_rw("error_sink", &BUSTEDSettings::error_sink)
        .def_rw("error_sink_max_weight", &BUSTEDSettings::error_sink_max_weight)
        .def_rw("error_sink_bf_threshold", &BUSTEDSettings::error_sink_bf_threshold)
        .def_rw("test_branches", &BUSTEDSettings::test_branches)
        .def_rw("phenotype_mode", &BUSTEDSettings::phenotype_mode)
        .def_rw("relax_mode", &BUSTEDSettings::relax_mode)
        .def_rw("include_relax_in_ph", &BUSTEDSettings::include_relax_in_ph)
        .def_rw("comparison_branches", &BUSTEDSettings::comparison_branches)
        .def_rw("p_value_threshold", &BUSTEDSettings::p_value_threshold);

    nb::class_<BUSTEDRateDistribution>(m, "BUSTEDRateDistribution")
        .def_ro("omegas", &BUSTEDRateDistribution::omegas)
        .def_ro("weights", &BUSTEDRateDistribution::weights)
        .def_ro("syn_rates", &BUSTEDRateDistribution::syn_rates)
        .def_ro("syn_weights", &BUSTEDRateDistribution::syn_weights)
        .def_ro("error_sink_weight", &BUSTEDRateDistribution::error_sink_weight)
        .def_ro("annotations", &BUSTEDRateDistribution::annotations);

    nb::class_<BUSTEDFit>(m, "BUSTEDFit")
        .def_ro("log_likelihood", &BUSTEDFit::log_likelihood)
        .def_ro("aicc", &BUSTEDFit::aicc)
        .def_ro("tree_scale", &BUSTEDFit::tree_scale)
        .def_ro("num_rate_classes", &BUSTEDFit::num_rate_classes)
        .def_ro("test_distribution", &BUSTEDFit::test_distribution)
        .def_ro("has_background_distribution", &BUSTEDFit::has_background_distribution)
        .def_ro("background_distribution", &BUSTEDFit::background_distribution)
        .def_ro("branch_lengths", &BUSTEDFit::branch_lengths)
        .def_ro("delta", &BUSTEDFit::delta)
        .def_ro("psi", &BUSTEDFit::psi)
        .def_ro("frac_delta", &BUSTEDFit::frac_delta)
        .def_ro("frac_psi", &BUSTEDFit::frac_psi)
        .def_ro("error_sink_weight", &BUSTEDFit::error_sink_weight);

    nb::class_<BUSTEDResult>(m, "BUSTEDResult")
        .def_ro("optimal_k", &BUSTEDResult::optimal_k)
        .def_ro("lrt", &BUSTEDResult::lrt)
        .def_ro("p_value", &BUSTEDResult::p_value)
        .def_ro("unconstrained", &BUSTEDResult::unconstrained)
        .def_ro("constrained", &BUSTEDResult::constrained)
        .def_ro("has_background_test", &BUSTEDResult::has_background_test)
        .def_ro("background_lrt", &BUSTEDResult::background_lrt)
        .def_ro("background_p_value", &BUSTEDResult::background_p_value)
        .def_ro("shared_lrt", &BUSTEDResult::shared_lrt)
        .def_ro("shared_p_value", &BUSTEDResult::shared_p_value)
        .def_ro("shared_df", &BUSTEDResult::shared_df)
        .def_ro("union_intersection_p_value", &BUSTEDResult::union_intersection_p_value)
        .def_ro("background_delta_aicc", &BUSTEDResult::background_delta_aicc)
        .def_ro("background_neutral_supported", &BUSTEDResult::background_neutral_supported)
        .def_ro("ranking_score", &BUSTEDResult::ranking_score)
        .def_ro("phenotype_associated", &BUSTEDResult::phenotype_associated)
        .def_ro("phenotype_verdict", &BUSTEDResult::phenotype_verdict)
        .def_ro("constrained_background", &BUSTEDResult::constrained_background)
        .def_ro("shared_distributions", &BUSTEDResult::shared_distributions)
        .def_ro("has_relax_test", &BUSTEDResult::has_relax_test)
        .def_ro("relax_k", &BUSTEDResult::relax_k)
        .def_ro("relax_lrt", &BUSTEDResult::relax_lrt)
        .def_ro("relax_p_value", &BUSTEDResult::relax_p_value)
        .def_ro("relax_direction", &BUSTEDResult::relax_direction)
        .def_ro("relax_alternative", &BUSTEDResult::relax_alternative)
        .def_ro("settings", &BUSTEDResult::settings)
        .def_ro("evidence_ratios", &BUSTEDResult::evidence_ratios)
        .def_ro("runtime_seconds", &BUSTEDResult::runtime_seconds);

    nb::class_<BUSTEDAnalyzer>(m, "BUSTEDAnalyzer")
        .def("__init__", [](BUSTEDAnalyzer* self, Tree tree, Alignment aln, const MG94Parameters& params) {
            new (self) BUSTEDAnalyzer(std::move(tree), std::move(aln), params);
        }, "tree"_a, "alignment"_a, "params"_a = MG94Parameters{})
        .def_static("create_and_fit", [](Tree tree, Alignment aln) {
            return BUSTEDAnalyzer::create_and_fit(std::move(tree), std::move(aln));
        }, "tree"_a, "alignment"_a)
        .def("run", &BUSTEDAnalyzer::run, "settings"_a = BUSTEDSettings{}, "show_progress"_a = false, "force_progress"_a = false)
        .def("to_json", [](const BUSTEDAnalyzer& busted, const BUSTEDResult& res, const std::string& fmt) {
            auto format = (fmt == "legacy") ? JSONFormat::Legacy : JSONFormat::ModernV3;
            return busted.to_json(res, format).dump();
        }, "res"_a, "format"_a = "modern")
        .def("to_legacy_json", [](const BUSTEDAnalyzer& busted, const BUSTEDResult& res) {
            return busted.to_legacy_json(res).dump();
        })
        .def("to_modern_json", [](const BUSTEDAnalyzer& busted, const BUSTEDResult& res) {
            return busted.to_modern_json(res).dump();
        });
}

} // namespace hyphy::bindings
