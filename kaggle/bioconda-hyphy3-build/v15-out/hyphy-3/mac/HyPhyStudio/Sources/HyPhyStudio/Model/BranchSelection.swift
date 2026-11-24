import Foundation

public struct BranchItem: Identifiable, Hashable, Sendable {
    public var id: Int
    public var name: String
    public var length: Double
    public var isLeaf: Bool
    public var isTested: Bool

    public init(id: Int, name: String, length: Double, isLeaf: Bool, isTested: Bool = true) {
        self.id = id
        self.name = name
        self.length = length
        self.isLeaf = isLeaf
        self.isTested = isTested
    }
}

@Observable
public final class TreeStagingModel {
    public var branches: [BranchItem] = []
    public var searchQuery: String = ""
    public var selectedPreset: String = "All Branches"

    public init(branches: [BranchItem] = []) {
        self.branches = branches
    }

    public var filteredBranches: [BranchItem] {
        if searchQuery.trimmingCharacters(in: .whitespaces).isEmpty {
            return branches
        }
        return branches.filter {
            $0.name.localizedCaseInsensitiveContains(searchQuery)
        }
    }

    public var testedCount: Int {
        branches.filter { $0.isTested }.count
    }

    public var backgroundCount: Int {
        branches.filter { !$0.isTested }.count
    }

    public func setAllTested(_ tested: Bool) {
        for i in 0..<branches.count {
            branches[i].isTested = tested
        }
    }

    public func setLeavesTested(_ tested: Bool) {
        for i in 0..<branches.count {
            if branches[i].isLeaf {
                branches[i].isTested = tested
            }
        }
    }

    public func setInternalsTested(_ tested: Bool) {
        for i in 0..<branches.count {
            if !branches[i].isLeaf {
                branches[i].isTested = tested
            }
        }
    }

    public func tagMatchingSearch(_ tested: Bool) {
        guard !searchQuery.isEmpty else { return }
        for i in 0..<branches.count {
            if branches[i].name.localizedCaseInsensitiveContains(searchQuery) {
                branches[i].isTested = tested
            }
        }
    }

    public func tagMatchingRegex(_ pattern: String, tested: Bool) {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else { return }
        for i in 0..<branches.count {
            let name = branches[i].name
            let range = NSRange(location: 0, length: name.utf16.count)
            if regex.firstMatch(in: name, options: [], range: range) != nil {
                branches[i].isTested = tested
            }
        }
    }
}
