#include "bindings_registry.hpp"
#include "hyphy/analyses/relax.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace hyphy::core;
using namespace hyphy::analyses;

namespace hyphy::bindings {

void register_relax(nb::module_ &m) {
    // RELAX Results & Analyzer
    nb::class_<RELAXSettings>(m, "RELAXSettings")
        .def(nb::init<>())
        .def_rw("p_value_threshold", &RELAXSettings::p_value_threshold)
        .def_rw("test_branch_regex", &RELAXSettings::test_branch_regex)
        .def_rw("test_branch_names", &RELAXSettings::test_branch_names)
        .def_rw("reference_branch_regex", &RELAXSettings::reference_branch_regex)
        .def_rw("reference_branch_names", &RELAXSettings::reference_branch_names)
        .def_rw("refine_branch_lengths", &RELAXSettings::refine_branch_lengths)
        .def_rw("max_joint_cycles", &RELAXSettings::max_joint_cycles)
        .def_rw("joint_tolerance", &RELAXSettings::joint_tolerance)
        .def_rw("verbose", &RELAXSettings::verbose);

    nb::class_<RELAXRateDistribution>(m, "RELAXRateDistribution")
        .def(nb::init<>())
        .def_ro("omegas", &RELAXRateDistribution::omegas)
        .def_ro("weights", &RELAXRateDistribution::weights);

    nb::class_<RELAXModelFit>(m, "RELAXModelFit")
        .def(nb::init<>())
        .def_ro("log_likelihood", &RELAXModelFit::log_likelihood)
        .def_ro("aicc", &RELAXModelFit::aicc)
        .def_ro("parameters", &RELAXModelFit::parameters)
        .def_ro("tree_scale", &RELAXModelFit::tree_scale)
        .def_ro("k", &RELAXModelFit::k)
        .def_ro("realized_cycles", &RELAXModelFit::realized_cycles)
        .def_ro("converged", &RELAXModelFit::converged)
        .def_ro("reference_distribution", &RELAXModelFit::reference_distribution)
        .def_ro("test_distribution", &RELAXModelFit::test_distribution)
        .def_ro("unclassified_distribution", &RELAXModelFit::unclassified_distribution)
        .def_ro("branch_lengths", &RELAXModelFit::branch_lengths)
        .def_ro("site_log_likelihoods", &RELAXModelFit::site_log_likelihoods);

    nb::class_<RELAXResult>(m, "RELAXResult")
        .def(nb::init<>())
        .def_ro("gtr_log_likelihood", &RELAXResult::gtr_log_likelihood)
        .def_ro("gtr_aicc", &RELAXResult::gtr_aicc)
        .def_ro("gtr_parameters", &RELAXResult::gtr_parameters)
        .def_ro("mg94_log_likelihood", &RELAXResult::mg94_log_likelihood)
        .def_ro("mg94_aicc", &RELAXResult::mg94_aicc)
        .def_ro("mg94_parameters", &RELAXResult::mg94_parameters)
        .def_ro("mg94_omega_R", &RELAXResult::mg94_omega_R)
        .def_ro("mg94_omega_T", &RELAXResult::mg94_omega_T)
        .def_ro("alternative_fit", &RELAXResult::alternative_fit)
        .def_ro("null_fit", &RELAXResult::null_fit)
        .def_ro("branch_switched_evidence_ratios", &RELAXResult::branch_switched_evidence_ratios)
        .def_ro("lrt", &RELAXResult::lrt)
        .def_ro("p_value", &RELAXResult::p_value)
        .def_ro("k", &RELAXResult::k)
        .def_ro("is_relaxed", &RELAXResult::is_relaxed)
        .def_ro("is_intensified", &RELAXResult::is_intensified)
        .def_ro("is_significant", &RELAXResult::is_significant)
        .def_ro("test_branches", &RELAXResult::test_branches)
        .def_ro("reference_branches", &RELAXResult::reference_branches)
        .def_ro("branch_class", &RELAXResult::branch_class)
        .def_ro("runtime_seconds", &RELAXResult::runtime_seconds)
        .def("to_json", [](const RELAXResult& res, const Tree& tree, const Alignment& aln, const std::string& fmt) {
            auto format = (fmt == "legacy") ? JSONFormat::Legacy : JSONFormat::ModernV3;
            return res.to_json(tree, aln, format).dump();
        }, "tree"_a, "alignment"_a, "format"_a = "modern")
        .def("to_legacy_json", [](const RELAXResult& res, const Tree& tree, const Alignment& aln) {
            return res.to_legacy_json(tree, aln).dump();
        }, "tree"_a, "alignment"_a)
        .def("to_modern_json", [](const RELAXResult& res, const Tree& tree, const Alignment& aln) {
            return res.to_modern_json(tree, aln).dump();
        }, "tree"_a, "alignment"_a);

    nb::class_<RELAXAnalyzer>(m, "RELAXAnalyzer")
        .def("__init__", [](RELAXAnalyzer* self, Tree tree, Alignment aln, const RELAXSettings& settings) {
            new (self) RELAXAnalyzer(std::move(tree), std::move(aln), settings);
        }, "tree"_a, "alignment"_a, "settings"_a = RELAXSettings{})
        .def_static("create", &RELAXAnalyzer::create, "tree"_a, "alignment"_a, "settings"_a = RELAXSettings{})
        .def("run", [](RELAXAnalyzer& relax, bool show_progress, bool force_progress) {
            return relax.run(nullptr, show_progress, force_progress);
        }, "show_progress"_a = false, "force_progress"_a = false);
}

} // namespace hyphy::bindings
