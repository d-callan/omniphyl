import Foundation
import HyPhyBridge
import CxxStdlib

private func cxxToString(_ s: std.string) -> String {
    let buf = hyphy.studio.to_buffer(s)
    guard let dataPtr = buf.data else { return "" }
    let rawBuf = UnsafeRawBufferPointer(start: dataPtr, count: Int(buf.length))
    return String(decoding: rawBuf, as: UTF8.self)
}

public actor HyPhyEngine {
    private var session: hyphy.studio.Session

    public init() {
        self.session = hyphy.studio.Session()
    }

    public func loadDataset(alignment: String, newick: String = "") -> (taxa: Int, codons: Int, branches: [BranchItem], treeNewick: String) {
        let alnOk = alignment.withCString { cstr in
            session.load_alignment_string(std.string(cstr))
        }
        guard alnOk else {
            return (0, 0, [], "")
        }

        var treeOk = false
        let trimmedNewick = newick.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedNewick.isEmpty {
            treeOk = trimmedNewick.withCString { cstr in
                session.load_tree_newick(std.string(cstr))
            }
        }

        var effectiveTree = trimmedNewick
        if !treeOk {
            let embedded = cxxToString(session.get_embedded_tree_newick())
            if !embedded.isEmpty {
                effectiveTree = embedded
                treeOk = true
            }
        }

        guard treeOk else {
            return (0, 0, [], "")
        }

        let summary = session.get_alignment_summary()
        let rawBranches = session.get_branches()
        var branches: [BranchItem] = []
        for i in 0..<rawBranches.size() {
            let b = rawBranches[i]
            branches.append(BranchItem(
                id: Int(b.id),
                name: cxxToString(b.name),
                length: b.length,
                isLeaf: b.is_leaf,
                isTested: b.is_tested
            ))
        }

        return (Int(summary.taxa_count), Int(summary.codon_count), branches, effectiveTree)
    }

    public func loadDataset(fasta: String, newick: String) -> (taxa: Int, codons: Int, branches: [BranchItem]) {
        let (taxa, codons, branches, _) = loadDataset(alignment: fasta, newick: newick)
        return (taxa, codons, branches)
    }

    public func applyBranchTesting(_ branches: [BranchItem]) {
        for b in branches {
            let isTested = b.isTested
            b.name.withCString { cstr in
                session.set_branch_tested(std.string(cstr), isTested)
            }
        }
    }

    public func runFEL(pvalThreshold: Double = 0.1) throws -> String {
        let jsonStdStr = session.run_fel(pvalThreshold)
        return cxxToString(jsonStdStr)
    }

    public func runMEME(pvalThreshold: Double = 0.1) throws -> String {
        let jsonStdStr = session.run_meme(pvalThreshold)
        return cxxToString(jsonStdStr)
    }

    public func runBUSTED() throws -> String {
        let jsonStdStr = session.run_busted()
        return cxxToString(jsonStdStr)
    }
}
