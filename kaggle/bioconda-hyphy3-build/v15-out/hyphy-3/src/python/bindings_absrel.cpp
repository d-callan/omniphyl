#include "bindings_registry.hpp"
#include "hyphy/analyses/absrel.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace hyphy::core;
using namespace hyphy::analyses;

namespace hyphy::bindings {

void register_absrel(nb::module_ &m) {
    // aBSREL Results & Analyzer
    nb::class_<ABSRELSettings>(m, "ABSRELSettings")
        .def(nb::init<>())
        .def_rw("max_rate_classes", &ABSRELSettings::max_rate_classes)
        .def_rw("p_threshold", &ABSRELSettings::p_threshold)
        .def_rw("test_branches", &ABSRELSettings::test_branches)
        .def_rw("verbose", &ABSRELSettings::verbose);

    nb::class_<ABSRELRateDistribution>(m, "ABSRELRateDistribution")
        .def(nb::init<>())
        .def_ro("rates", &ABSRELRateDistribution::rates)
        .def_ro("weights", &ABSRELRateDistribution::weights);

    nb::class_<ABSRELBranchResult>(m, "ABSRELBranchResult")
        .def(nb::init<>())
        .def_ro("branch_name", &ABSRELBranchResult::branch_name)
        .def_ro("node_id", &ABSRELBranchResult::node_id)
        .def_ro("baseline_omega", &ABSRELBranchResult::baseline_omega)
        .def_ro("baseline_branch_length", &ABSRELBranchResult::baseline_branch_length)
        .def_ro("rate_classes", &ABSRELBranchResult::rate_classes)
        .def_ro("rate_distribution", &ABSRELBranchResult::rate_distribution)
        .def_ro("full_branch_length", &ABSRELBranchResult::full_branch_length)
        .def_ro("full_es", &ABSRELBranchResult::full_es)
        .def_ro("full_en", &ABSRELBranchResult::full_en)
        .def_ro("lrt", &ABSRELBranchResult::lrt)
        .def_ro("uncorrected_p_value", &ABSRELBranchResult::uncorrected_p_value)
        .def_ro("corrected_p_value", &ABSRELBranchResult::corrected_p_value)
        .def_ro("is_tested", &ABSRELBranchResult::is_tested)
        .def_ro("is_positive", &ABSRELBranchResult::is_positive)
        .def_ro("sites_ebf_100", &ABSRELBranchResult::sites_ebf_100)
        .def_ro("site_ebf", &ABSRELBranchResult::site_ebf);

    nb::class_<ABSRELFitSummary>(m, "ABSRELFitSummary")
        .def(nb::init<>())
        .def_ro("log_likelihood", &ABSRELFitSummary::log_likelihood)
        .def_ro("parameters", &ABSRELFitSummary::parameters)
        .def_ro("aicc", &ABSRELFitSummary::aicc);

    nb::class_<ABSRELResult>(m, "ABSRELResult")
        .def(nb::init<>())
        .def_ro("gtr_fit", &ABSRELResult::gtr_fit)
        .def_ro("baseline_fit", &ABSRELResult::baseline_fit)
        .def_ro("full_adaptive_fit", &ABSRELResult::full_adaptive_fit)
        .def_ro("branches", &ABSRELResult::branches)
        .def_ro("tested_branches", &ABSRELResult::tested_branches)
        .def_ro("positive_branches", &ABSRELResult::positive_branches)
        .def_ro("p_threshold", &ABSRELResult::p_threshold)
        .def_ro("runtime_seconds", &ABSRELResult::runtime_seconds)
        .def("to_json", [](const ABSRELResult& res, const Tree& tree, const Alignment& aln, const std::string& fmt) {
            auto format = (fmt == "legacy") ? JSONFormat::Legacy : JSONFormat::ModernV3;
            return res.to_json(tree, aln, format).dump();
        }, "tree"_a, "alignment"_a, "format"_a = "modern")
        .def("to_legacy_json", [](const ABSRELResult& res, const Tree& tree, const Alignment& aln) {
            return res.to_legacy_json(tree, aln).dump();
        }, "tree"_a, "alignment"_a)
        .def("to_modern_json", [](const ABSRELResult& res, const Tree& tree, const Alignment& aln) {
            return res.to_modern_json(tree, aln).dump();
        }, "tree"_a, "alignment"_a);

    nb::class_<ABSRELAnalyzer>(m, "ABSRELAnalyzer")
        .def("__init__", [](ABSRELAnalyzer* self, Tree tree, Alignment aln, const ABSRELSettings& settings) {
            new (self) ABSRELAnalyzer(std::move(tree), std::move(aln), settings);
        }, "tree"_a, "alignment"_a, "settings"_a = ABSRELSettings{})
        .def_static("create", &ABSRELAnalyzer::create, "tree"_a, "alignment"_a, "settings"_a = ABSRELSettings{})
        .def("run", [](ABSRELAnalyzer& absrel, bool show_progress, bool force_progress) {
            return absrel.run(nullptr, show_progress, force_progress);
        }, "show_progress"_a = false, "force_progress"_a = false);
}

} // namespace hyphy::bindings
