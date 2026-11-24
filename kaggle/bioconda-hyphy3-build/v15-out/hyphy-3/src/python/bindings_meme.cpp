#include "bindings_registry.hpp"
#include "hyphy/analyses/meme.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace hyphy::core;
using namespace hyphy::analyses;

namespace hyphy::bindings {

void register_meme(nb::module_ &m) {
    // MEME Results & Analyzer
    nb::class_<MEMESiteResult>(m, "MEMESiteResult")
        .def_ro("site_index", &MEMESiteResult::site_index)
        .def_ro("alpha", &MEMESiteResult::alpha)
        .def_ro("beta1", &MEMESiteResult::beta1)
        .def_ro("p1", &MEMESiteResult::p1)
        .def_ro("beta_plus", &MEMESiteResult::beta_plus)
        .def_ro("p_plus", &MEMESiteResult::p_plus)
        .def_ro("lrt", &MEMESiteResult::lrt)
        .def_ro("p_value", &MEMESiteResult::p_value)
        .def_ro("branches_under_selection", &MEMESiteResult::branches_under_selection)
        .def_ro("total_branch_length", &MEMESiteResult::total_branch_length)
        .def_ro("log_l_meme", &MEMESiteResult::log_l_meme)
        .def_ro("log_l_fel", &MEMESiteResult::log_l_fel)
        .def_ro("p_value_meme_vs_fel", &MEMESiteResult::p_value_meme_vs_fel)
        .def_ro("branch_ebf", &MEMESiteResult::branch_ebf);

    nb::class_<MEMEAnalyzer>(m, "MEMEAnalyzer")
        .def("__init__", [](MEMEAnalyzer* self, Tree tree, Alignment aln, const MG94Parameters& params) {
            new (self) MEMEAnalyzer(std::move(tree), std::move(aln), params);
        }, "tree"_a, "alignment"_a, "params"_a = MG94Parameters{})
        .def_static("create_and_fit", [](Tree tree, Alignment aln, Scalar pvalue_threshold, bool full_model) {
            return MEMEAnalyzer::create_and_fit(std::move(tree), std::move(aln), pvalue_threshold, nullptr, full_model);
        }, "tree"_a, "alignment"_a, "pvalue_threshold"_a = 0.1, "full_model"_a = true)
        .def("run", &MEMEAnalyzer::run, "show_progress"_a = false, "force_progress"_a = false)
        .def("to_json", [](const MEMEAnalyzer& meme, const std::string& fmt) {
            auto format = (fmt == "legacy") ? JSONFormat::Legacy : JSONFormat::ModernV3;
            return meme.to_json(format).dump();
        }, "format"_a = "modern")
        .def("to_legacy_json", [](const MEMEAnalyzer& meme) {
            return meme.to_legacy_json().dump();
        })
        .def("to_modern_json", [](const MEMEAnalyzer& meme) {
            return meme.to_modern_json().dump();
        })
        .def_ro("global_log_l", &MEMEAnalyzer::global_log_l)
        .def_ro("global_aicc", &MEMEAnalyzer::global_aicc)
        .def_ro("site_results", &MEMEAnalyzer::site_results);
}

} // namespace hyphy::bindings
