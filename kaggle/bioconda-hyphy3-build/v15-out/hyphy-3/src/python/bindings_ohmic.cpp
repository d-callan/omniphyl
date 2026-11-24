#include "bindings_registry.hpp"
#include "hyphy/core/ohmic_network.hpp"
#include "hyphy/core/wheatstone_likelihood.hpp"
#include "hyphy/analyses/ohmic.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace hyphy::core;
using namespace hyphy::analyses;

namespace hyphy::bindings {

void register_ohmic(nb::module_ &m) {
    // Ohmic Phylogenetic Networks & Wheatstone Pruning
    nb::enum_<OhmicScaleMode>(m, "OhmicScaleMode")
        .value("Auto", OhmicScaleMode::Auto)
        .value("ExactDense", OhmicScaleMode::ExactDense)
        .value("ScalableWoodbury", OhmicScaleMode::ScalableWoodbury);

    nb::class_<OhmicSettings>(m, "OhmicSettings")
        .def(nb::init<>())
        .def_rw("scale_mode", &OhmicSettings::scale_mode)
        .def_rw("dense_switch_threshold", &OhmicSettings::dense_switch_threshold)
        .def_rw("rank_ratio_threshold", &OhmicSettings::rank_ratio_threshold)
        .def_rw("woodbury_rank", &OhmicSettings::woodbury_rank)
        .def_rw("min_shunt_stress", &OhmicSettings::min_shunt_stress)
        .def_rw("max_bridges", &OhmicSettings::max_bridges)
        .def_rw("collapse_ohmic_polytomies", &OhmicSettings::collapse_ohmic_polytomies)
        .def_rw("include_tertiary", &OhmicSettings::include_tertiary)
        .def_rw("use_hamming_distance", &OhmicSettings::use_hamming_distance)
        .def_rw("fixed_sigma", &OhmicSettings::fixed_sigma)
        .def_rw("epsilon_0", &OhmicSettings::epsilon_0);

    nb::class_<WheatstoneBridge>(m, "WheatstoneBridge")
        .def(nb::init<>())
        .def_rw("hub_node_id", &WheatstoneBridge::hub_node_id)
        .def_rw("bridge_node_id", &WheatstoneBridge::bridge_node_id)
        .def_rw("children", &WheatstoneBridge::children)
        .def_rw("weights", &WheatstoneBridge::weights)
        .def_rw("branch_lengths", &WheatstoneBridge::branch_lengths)
        .def_rw("ohmic_support", &WheatstoneBridge::ohmic_support)
        .def_rw("shunt_stress", &WheatstoneBridge::shunt_stress);

    nb::class_<OhmicNetwork>(m, "OhmicNetwork")
        .def(nb::init<>())
        .def_ro("backbone_tree", &OhmicNetwork::backbone_tree)
        .def_ro("collapsed_tree", &OhmicNetwork::collapsed_tree)
        .def_rw("bridges", &OhmicNetwork::bridges)
        .def_ro("node_to_bridge_idx", &OhmicNetwork::node_to_bridge_idx)
        .def_ro("active_mode", &OhmicNetwork::active_mode)
        .def_ro("num_super_nodes", &OhmicNetwork::num_super_nodes)
        .def_ro("num_collapsed_polytomies", &OhmicNetwork::num_collapsed_polytomies)
        .def_ro("D_tn93", &OhmicNetwork::D_tn93)
        .def_ro("Omega", &OhmicNetwork::Omega)
        .def_ro("P_edge", &OhmicNetwork::P_edge)
        .def_ro("foster_sum", &OhmicNetwork::foster_sum)
        .def_ro("inv_aug_degrees", &OhmicNetwork::inv_aug_degrees)
        .def_ro("Z_resistance_coords", &OhmicNetwork::Z_resistance_coords)
        .def("effective_resistance", &OhmicNetwork::effective_resistance, "i"_a, "j"_a)
        .def_static("infer", [](const Alignment& aln, const OhmicSettings& settings) {
            return OhmicNetwork::infer(aln, settings, nullptr);
        }, "alignment"_a, "settings"_a = OhmicSettings{});

    nb::class_<OhmicAnalysisSettings>(m, "OhmicAnalysisSettings")
        .def(nb::init<>())
        .def_rw("network_settings", &OhmicAnalysisSettings::network_settings)
        .def_rw("use_codon_model", &OhmicAnalysisSettings::use_codon_model)
        .def_rw("optimize_bridge_weights", &OhmicAnalysisSettings::optimize_bridge_weights)
        .def_rw("optimize_branch_lengths", &OhmicAnalysisSettings::optimize_branch_lengths)
        .def_rw("quiet", &OhmicAnalysisSettings::quiet)
        .def_rw("input_file", &OhmicAnalysisSettings::input_file)
        .def_rw("output_file", &OhmicAnalysisSettings::output_file);

    nb::class_<OhmicBridgeSummary>(m, "OhmicBridgeSummary")
        .def_ro("bridge_index", &OhmicBridgeSummary::bridge_index)
        .def_ro("hub_name", &OhmicBridgeSummary::hub_name)
        .def_ro("bridge_name", &OhmicBridgeSummary::bridge_name)
        .def_ro("child_names", &OhmicBridgeSummary::child_names)
        .def_ro("initial_weights", &OhmicBridgeSummary::initial_weights)
        .def_ro("fitted_weights", &OhmicBridgeSummary::fitted_weights)
        .def_ro("fitted_branch_lengths", &OhmicBridgeSummary::fitted_branch_lengths)
        .def_ro("ohmic_support", &OhmicBridgeSummary::ohmic_support)
        .def_ro("shunt_stress", &OhmicBridgeSummary::shunt_stress);

    nb::class_<OhmicResult>(m, "OhmicResult")
        .def_ro("network", &OhmicResult::network)
        .def_ro("num_taxa", &OhmicResult::num_taxa)
        .def_ro("num_sites", &OhmicResult::num_sites)
        .def_ro("num_patterns", &OhmicResult::num_patterns)
        .def_ro("num_super_nodes", &OhmicResult::num_super_nodes)
        .def_ro("num_bridges", &OhmicResult::num_bridges)
        .def_ro("num_collapsed_polytomies", &OhmicResult::num_collapsed_polytomies)
        .def_ro("foster_sum", &OhmicResult::foster_sum)
        .def_ro("single_tree_log_l", &OhmicResult::single_tree_log_l)
        .def_ro("single_tree_aicc", &OhmicResult::single_tree_aicc)
        .def_ro("ohmic_log_l", &OhmicResult::ohmic_log_l)
        .def_ro("ohmic_aicc", &OhmicResult::ohmic_aicc)
        .def_ro("delta_log_l", &OhmicResult::delta_log_l)
        .def_ro("delta_aicc", &OhmicResult::delta_aicc)
        .def_ro("bridge_summaries", &OhmicResult::bridge_summaries)
        .def_ro("runtime_seconds", &OhmicResult::runtime_seconds)
        .def("to_modern_json", [](const OhmicResult& res, const std::string& filepath) {
            return res.to_modern_json(filepath).dump();
        }, "filepath"_a = "")
        .def("to_legacy_json", [](const OhmicResult& res, const std::string& filepath) {
            return res.to_legacy_json(filepath).dump();
        }, "filepath"_a = "");

    nb::class_<OhmicAnalyzer>(m, "OhmicAnalyzer")
        .def(nb::init<Alignment, OhmicAnalysisSettings>(), "alignment"_a, "settings"_a = OhmicAnalysisSettings{})
        .def("run", [](OhmicAnalyzer& analyzer) {
            return analyzer.run(nullptr);
        });
}

} // namespace hyphy::bindings
