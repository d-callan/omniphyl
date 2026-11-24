#include "HyPhyBridge.hpp"
#include <hyphy/core/tree.hpp>
#include <hyphy/core/alignment.hpp>
#include <hyphy/analyses/fel.hpp>
#include <hyphy/analyses/meme.hpp>
#include <hyphy/analyses/busted.hpp>
#include <fstream>
#include <sstream>
#include <regex>
#include <unordered_set>

namespace hyphy::studio {

struct Session::Impl {
    core::Tree tree;
    core::Alignment aln;
    bool has_tree = false;
    bool has_aln = false;
    std::string newick_source;
    std::unordered_set<std::string> tested_branches;
};

Session::Session() : impl(new Impl()) {}
Session::~Session() { delete impl; }

Session::Session(Session&& other) noexcept : impl(other.impl) {
    other.impl = nullptr;
}

Session& Session::operator=(Session&& other) noexcept {
    if (this != &other) {
        delete impl;
        impl = other.impl;
        other.impl = nullptr;
    }
    return *this;
}

bool Session::load_alignment_file(const std::string& filepath) {
    try {
        impl->aln = core::Alignment::load(filepath);
        impl->has_aln = true;
        if (!impl->aln.embedded_tree_newick.empty()) {
            load_tree_newick(impl->aln.embedded_tree_newick);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool Session::load_alignment_string(const std::string& content) {
    std::string temp_path = "/tmp/hyphy_studio_temp_aln.txt";
    std::ofstream out(temp_path);
    if (!out.is_open()) return false;
    out << content;
    out.close();
    return load_alignment_file(temp_path);
}

std::string Session::get_embedded_tree_newick() const {
    if (!impl->has_aln) return "";
    return impl->aln.embedded_tree_newick;
}

AlignmentSummary Session::get_alignment_summary() const {
    AlignmentSummary summary;
    if (!impl->has_aln) return summary;
    summary.taxa_count = impl->aln.num_taxa;
    summary.codon_count = impl->aln.num_codons;
    summary.nucleotide_count = impl->aln.num_codons * 3;
    summary.taxa_names = impl->aln.taxon_names;
    return summary;
}

bool Session::load_tree_file(const std::string& filepath) {
    try {
        impl->tree = core::Tree::from_newick_file(filepath);
        impl->newick_source = impl->tree.to_newick();
        impl->has_tree = true;
        impl->tested_branches.clear();
        for (const auto& n : impl->tree.nodes) {
            impl->tested_branches.insert(n.name);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool Session::load_tree_newick(const std::string& newick_str) {
    try {
        impl->tree = core::Tree::from_newick(newick_str);
        impl->newick_source = newick_str;
        impl->has_tree = true;
        impl->tested_branches.clear();
        for (const auto& n : impl->tree.nodes) {
            impl->tested_branches.insert(n.name);
        }
        return true;
    } catch (...) {
        return false;
    }
}

std::string Session::get_tree_newick() const {
    if (!impl->has_tree) return "";
    return impl->tree.to_newick();
}

std::vector<TreeBranchInfo> Session::get_branches() const {
    std::vector<TreeBranchInfo> list;
    if (!impl->has_tree) return list;
    for (const auto& n : impl->tree.nodes) {
        TreeBranchInfo info;
        info.id = n.id;
        info.name = n.name;
        info.length = n.branch_length;
        info.is_leaf = n.children.empty();
        info.is_tested = (impl->tested_branches.count(n.name) > 0);
        list.push_back(info);
    }
    return list;
}

void Session::set_branch_tested(const std::string& branch_name, bool is_tested) {
    if (!impl->has_tree) return;
    if (is_tested) {
        impl->tested_branches.insert(branch_name);
    } else {
        impl->tested_branches.erase(branch_name);
    }
}

void Session::set_all_branches_tested(bool is_tested) {
    if (!impl->has_tree) return;
    impl->tested_branches.clear();
    if (is_tested) {
        for (const auto& n : impl->tree.nodes) {
            impl->tested_branches.insert(n.name);
        }
    }
}

void Session::set_clade_tested(int32_t node_id, bool is_tested) {
    if (!impl->has_tree || node_id < 0 || static_cast<size_t>(node_id) >= impl->tree.nodes.size()) return;
    std::vector<int32_t> stack;
    stack.push_back(node_id);
    while (!stack.empty()) {
        int32_t cur = stack.back();
        stack.pop_back();
        const std::string& bname = impl->tree.nodes[cur].name;
        if (is_tested) impl->tested_branches.insert(bname);
        else impl->tested_branches.erase(bname);
        for (int32_t child_id : impl->tree.nodes[cur].children) {
            stack.push_back(child_id);
        }
    }
}

void Session::set_branches_matching_regex(const std::string& regex_pattern, bool is_tested) {
    if (!impl->has_tree) return;
    try {
        std::regex re(regex_pattern, std::regex::icase);
        for (const auto& n : impl->tree.nodes) {
            if (std::regex_search(n.name, re)) {
                if (is_tested) impl->tested_branches.insert(n.name);
                else impl->tested_branches.erase(n.name);
            }
        }
    } catch (...) {}
}

std::string Session::run_fel(double pval_threshold) {
    if (!impl->has_tree || !impl->has_aln) return "{}";
    analyses::FELSettings settings;
    settings.p_value_threshold = pval_threshold;
    analyses::FELAnalyzer analyzer(impl->tree, impl->aln, core::MG94Parameters{}, settings);
    analyzer.run(false, false);
    return analyzer.to_modern_json().dump();
}

std::string Session::run_meme(double pval_threshold) {
    if (!impl->has_tree || !impl->has_aln) return "{}";
    core::MG94Parameters mg94;
    analyses::MEMEAnalyzer analyzer(impl->tree, impl->aln, mg94);
    analyzer.p_value_threshold = pval_threshold;
    analyzer.run(false, false);
    return analyzer.to_modern_json().dump();
}

std::string Session::run_busted() {
    if (!impl->has_tree || !impl->has_aln) return "{}";
    analyses::BUSTEDSettings settings;
    analyses::BUSTEDAnalyzer analyzer(impl->tree, impl->aln);
    auto res = analyzer.run(settings, false, false);
    return analyzer.to_modern_json(res).dump();
}

} // namespace hyphy::studio
