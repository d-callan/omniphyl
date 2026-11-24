#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace hyphy::studio {

struct TreeBranchInfo {
    int32_t id;
    std::string name;
    double length;
    bool is_leaf;
    bool is_tested;
};

struct AlignmentSummary {
    size_t taxa_count = 0;
    size_t codon_count = 0;
    size_t nucleotide_count = 0;
    std::vector<std::string> taxa_names;
};

class Session {
private:
    struct Impl;
    Impl* impl;

public:
    Session();
    ~Session();

    // Disable copy to maintain unique ownership
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&) noexcept;
    Session& operator=(Session&&) noexcept;

    // Alignment management (supports .nex, .nexus, .fas, .fasta, .fna)
    bool load_alignment_file(const std::string& filepath);
    bool load_alignment_string(const std::string& content);
    std::string get_embedded_tree_newick() const;
    AlignmentSummary get_alignment_summary() const;

    // Tree management & Pre-flight Interactive Staging
    bool load_tree_file(const std::string& filepath);
    bool load_tree_newick(const std::string& newick_str);
    std::string get_tree_newick() const;
    std::vector<TreeBranchInfo> get_branches() const;
    
    // Interactive branch tagging
    void set_branch_tested(const std::string& branch_name, bool is_tested);
    void set_all_branches_tested(bool is_tested);
    void set_clade_tested(int32_t node_id, bool is_tested);
    void set_branches_matching_regex(const std::string& regex_pattern, bool is_tested);

    // Direct in-process analysis execution (returns Modern HyPhy v3.0 JSON string)
    std::string run_fel(double pval_threshold = 0.1);
    std::string run_meme(double pval_threshold = 0.1);
    std::string run_busted();
};

struct CxxBuffer {
    const char* data;
    size_t length;
};

inline CxxBuffer to_buffer(const std::string& s) {
    return CxxBuffer{s.data(), s.size()};
}

} // namespace hyphy::studio
