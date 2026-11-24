#include "bindings_registry.hpp"
#include "hyphy/core/types.hpp"
#include "hyphy/core/genetic_code.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/rate_matrix.hpp"
#include "hyphy/core/likelihood.hpp"
#include "hyphy/core/neighbor_joining.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace hyphy::core;

namespace hyphy::bindings {

void register_core(nb::module_ &m) {
    // GeneticCode
    nb::class_<GeneticCode>(m, "GeneticCode")
        .def_static("universal", &GeneticCode::universal)
        .def_static("from_name", &GeneticCode::from_name, "name"_a)
        .def_ro("name", &GeneticCode::name)
        .def_ro("num_sense_codons", &GeneticCode::num_sense_codons);

    // TreeNode
    nb::class_<TreeNode>(m, "TreeNode")
        .def_ro("id", &TreeNode::id)
        .def_ro("name", &TreeNode::name)
        .def_ro("parent_id", &TreeNode::parent_id)
        .def_ro("children", &TreeNode::children)
        .def_rw("branch_length", &TreeNode::branch_length)
        .def_ro("is_leaf", &TreeNode::is_leaf);

    // Tree
    nb::class_<Tree>(m, "Tree")
        .def(nb::init<>())
        .def_static("from_newick", &Tree::from_newick, "newick"_a, "keep_rooted"_a = true)
        .def_static("from_newick_file", &Tree::from_newick_file, "filepath"_a, "keep_rooted"_a = true)
        .def("num_nodes", &Tree::num_nodes)
        .def("num_leaves", &Tree::num_leaves)
        .def_ro("root_id", &Tree::root_id)
        .def_ro("post_order", &Tree::post_order)
        .def_ro("nodes", &Tree::nodes)
        .def_ro("leaf_name_to_id", &Tree::leaf_name_to_id)
        .def("get_branch_lengths", [](const Tree& tree) {
            std::vector<Scalar> bl(tree.num_nodes(), 0.0);
            for (size_t i = 0; i < tree.nodes.size(); ++i) {
                bl[i] = tree.nodes[i].branch_length;
            }
            return bl;
        })
        .def("set_branch_lengths", [](Tree& tree, const std::vector<Scalar>& bl) {
            for (size_t i = 0; i < tree.nodes.size() && i < bl.size(); ++i) {
                tree.nodes[i].branch_length = bl[i];
            }
        })
        .def("get_branch_length", [](const Tree& tree, int32_t id) {
            return tree.nodes.at(id).branch_length;
        }, "id"_a)
        .def("set_branch_length", [](Tree& tree, int32_t id, Scalar length) {
            tree.nodes.at(id).branch_length = length;
        }, "id"_a, "length"_a);

    // Alignment
    nb::class_<Alignment>(m, "Alignment")
        .def(nb::init<>())
        .def_static("load", [](const std::string& filepath) { return Alignment::load(filepath); }, "filepath"_a)
        .def_static("load_nucleotide", [](const std::string& filepath) { return Alignment::load_nucleotide(filepath); }, "filepath"_a)
        .def_static("from_fasta", [](const std::string& filepath) { return Alignment::from_fasta(filepath); }, "filepath"_a)
        .def_static("from_fasta_nucleotide", [](const std::string& filepath) { return Alignment::from_fasta_nucleotide(filepath); }, "filepath"_a)
        .def_static("from_nexus", [](const std::string& filepath) { return Alignment::from_nexus(filepath); }, "filepath"_a)
        .def_static("from_nexus_nucleotide", [](const std::string& filepath) { return Alignment::from_nexus_nucleotide(filepath); }, "filepath"_a)
        .def("slice_nucleotide", &Alignment::slice_nucleotide, "start"_a, "end"_a)
        .def_ro("num_taxa", &Alignment::num_taxa)
        .def_prop_ro("num_sequences", [](const Alignment& a) { return a.num_taxa; })
        .def_ro("num_codons", &Alignment::num_codons)
        .def_ro("num_nucleotides", &Alignment::num_nucleotides)
        .def_prop_ro("num_sites", [](const Alignment& a) { return a.num_codons; })
        .def_prop_ro("num_patterns", [](const Alignment& a) { return a.patterns.size(); })
        .def_ro("embedded_tree_newick", &Alignment::embedded_tree_newick)
        .def_ro("taxon_names", &Alignment::taxon_names);

    // MG94Parameters
    nb::class_<MG94Parameters>(m, "MG94Parameters")
        .def(nb::init<>())
        .def_rw("alpha", &MG94Parameters::alpha)
        .def_rw("beta", &MG94Parameters::beta)
        .def_rw("theta_AC", &MG94Parameters::theta_AC)
        .def_rw("theta_AT", &MG94Parameters::theta_AT)
        .def_rw("theta_CG", &MG94Parameters::theta_CG)
        .def_rw("theta_CT", &MG94Parameters::theta_CT)
        .def_rw("theta_GT", &MG94Parameters::theta_GT)
        .def_rw("delta", &MG94Parameters::delta)
        .def_rw("psi", &MG94Parameters::psi);
    // Differentiable Engine: Inside-Outside Likelihood and Analytical Adjoint Gradients!
    m.def("compute_branch_length_gradients", [](
        const Tree& tree,
        const Alignment& aln,
        const MG94Parameters& params,
        const std::vector<Scalar>& branch_lengths
    ) -> std::pair<Scalar, std::vector<Scalar>> {
        return LikelihoodEngine::compute_branch_length_gradients(tree, aln, params, branch_lengths);
    }, "tree"_a, "aln"_a, "params"_a, "branch_lengths"_a = std::vector<Scalar>{}, "Compute log-likelihood and exact analytical branch gradients d ln L / d t_b via Inside-Outside");

    // Neighbor Joining
    m.def("compute_tn93_distances", &NeighborJoining::compute_tn93_distances, "aln"_a);
    m.def("infer_nj_tree", &NeighborJoining::infer, "aln"_a);
}

} // namespace hyphy::bindings
