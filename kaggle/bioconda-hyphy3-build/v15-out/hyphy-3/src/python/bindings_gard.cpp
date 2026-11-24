#include "bindings_registry.hpp"
#include "hyphy/analyses/gard.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace hyphy::core;
using namespace hyphy::analyses;

namespace hyphy::bindings {

void register_gard(nb::module_ &m) {
    // GARD Settings, Partitions, Models, Results & Analyzer
    nb::class_<GARDSettings>(m, "GARDSettings")
        .def(nb::init<>())
        .def_rw("input_file", &GARDSettings::input_file)
        .def_rw("output_file", &GARDSettings::output_file)
        .def_rw("run_mode", &GARDSettings::run_mode)
        .def_rw("max_breakpoints", &GARDSettings::max_breakpoints)
        .def_rw("population_size", &GARDSettings::population_size)
        .def_rw("mutation_rate", &GARDSettings::mutation_rate)
        .def_rw("rate_small_shifts", &GARDSettings::rate_small_shifts)
        .def_rw("caic_diversity_threshold", &GARDSettings::caic_diversity_threshold)
        .def_rw("max_stagnant_generations", &GARDSettings::max_stagnant_generations)
        .def_rw("caic_improvement_threshold", &GARDSettings::caic_improvement_threshold)
        .def_rw("quiet", &GARDSettings::quiet)
        .def_rw("dynamic_chart", &GARDSettings::dynamic_chart)
        .def_rw("force_tty", &GARDSettings::force_tty)
        .def_rw("legacy_json", &GARDSettings::legacy_json);

    nb::class_<GARDPartition>(m, "GARDPartition")
        .def_ro("start_site", &GARDPartition::start_site)
        .def_ro("end_site", &GARDPartition::end_site)
        .def_ro("tree", &GARDPartition::tree)
        .def_ro("log_likelihood", &GARDPartition::log_likelihood);

    nb::class_<GARDModel>(m, "GARDModel")
        .def_ro("breakpoints", &GARDModel::breakpoints)
        .def_ro("log_likelihood", &GARDModel::log_likelihood)
        .def_ro("aicc", &GARDModel::aicc)
        .def_ro("partitions", &GARDModel::partitions);

    nb::class_<GARDResult>(m, "GARDResult")
        .def_ro("baseline_aicc", &GARDResult::baseline_aicc)
        .def_ro("baseline_log_l", &GARDResult::baseline_log_l)
        .def_ro("baseline_tree", &GARDResult::baseline_tree)
        .def_ro("num_sites", &GARDResult::num_sites)
        .def_ro("num_taxa", &GARDResult::num_taxa)
        .def_ro("potential_breakpoints", &GARDResult::potential_breakpoints)
        .def_ro("min_partition_size", &GARDResult::min_partition_size)
        .def_ro("variable_sites", &GARDResult::variable_sites)
        .def_ro("best_model", &GARDResult::best_model)
        .def_ro("single_tree_aicc", &GARDResult::single_tree_aicc)
        .def_ro("topological_incongruence", &GARDResult::topological_incongruence)
        .def_ro("site_breakpoint_support", &GARDResult::site_breakpoint_support)
        .def_ro("improvements", &GARDResult::improvements)
        .def_ro("total_models_evaluated", &GARDResult::total_models_evaluated)
        .def_ro("time_elapsed", &GARDResult::time_elapsed)
        .def("to_modern_json", [](const GARDResult& res) { return res.to_json().dump(); })
        .def("to_legacy_json", [](const GARDResult& res) { return res.to_legacy_json().dump(); });

    nb::class_<GARDAnalyzer>(m, "GARDAnalyzer")
        .def(nb::init<const Alignment&, GARDSettings>(), "alignment"_a, "settings"_a = GARDSettings{})
        .def("run", &GARDAnalyzer::run);
}

} // namespace hyphy::bindings
