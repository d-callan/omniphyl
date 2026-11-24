import SwiftUI

public struct TreeStagingView: View {
    @Bindable public var model: TreeStagingModel
    public var onApply: (() -> Void)?

    @State private var selectedBranchID: Int?
    @State private var isRegexMode = false

    public init(model: TreeStagingModel, onApply: (() -> Void)? = nil) {
        self.model = model
        self.onApply = onApply
    }

    public var body: some View {
        VStack(spacing: 0) {
            // 1. Pre-flight Header
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Phylogenetic Tree Staging & Pre-flight")
                            .font(.system(size: 14, weight: .bold))
                        Text("Designate foreground lineages for episodic diversifying selection (MEME / RELAX / BUSTED).")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }
                    Spacer()

                    // Tested vs Background Statistics Pills
                    HStack(spacing: 6) {
                        Label("\(model.testedCount) Tested", systemImage: "checkmark.circle.fill")
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.orange.opacity(0.12))
                            .foregroundStyle(Color.orange)
                            .clipShape(Capsule())

                        Label("\(model.backgroundCount) Background", systemImage: "circle.slash")
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.secondary.opacity(0.12))
                            .foregroundStyle(.secondary)
                            .clipShape(Capsule())
                    }
                }

                // 2. Search & Filter Bar
                HStack(spacing: 8) {
                    HStack(spacing: 4) {
                        Image(systemName: "magnifyingglass")
                            .foregroundStyle(.secondary)
                            .font(.system(size: 11))
                        TextField("Search taxa, nodes, or regex (e.g. .*_avian|cat)...", text: $model.searchQuery)
                            .textFieldStyle(.plain)
                            .font(.system(size: 11))
                        if !model.searchQuery.isEmpty {
                            Button(action: { model.searchQuery = "" }) {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(6)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.2), lineWidth: 1))

                    Toggle("RegEx", isOn: $isRegexMode)
                        .toggleStyle(.button)
                        .font(.system(size: 10, weight: .medium))

                    Menu("Batch Actions") {
                        Button("Tag Matches as Tested (Foreground)") {
                            if isRegexMode {
                                model.tagMatchingRegex(model.searchQuery, tested: true)
                            } else {
                                model.tagMatchingSearch(true)
                            }
                            onApply?()
                        }
                        Button("Tag Matches as Background") {
                            if isRegexMode {
                                model.tagMatchingRegex(model.searchQuery, tested: false)
                            } else {
                                model.tagMatchingSearch(false)
                            }
                            onApply?()
                        }
                        Divider()
                        Button("Test All Branches") {
                            model.setAllTested(true)
                            onApply?()
                        }
                        Button("Test Extant Leaves Only") {
                            model.setAllTested(false)
                            model.setLeavesTested(true)
                            onApply?()
                        }
                        Button("Test Internal Branches Only") {
                            model.setAllTested(false)
                            model.setInternalsTested(true)
                            onApply?()
                        }
                        Button("Clear All (Set All Background)") {
                            model.setAllTested(false)
                            onApply?()
                        }
                    }
                    .menuStyle(.borderedButton)
                    .font(.system(size: 11))
                }
            }
            .padding(12)
            .background(Color(nsColor: .windowBackgroundColor))

            Divider()

            // 3. Branch Staging Table
            Table(model.filteredBranches, selection: $selectedBranchID) {
                TableColumn("Branch / Node") { branch in
                    HStack(spacing: 6) {
                        Image(systemName: branch.isLeaf ? "leaf.fill" : "arrow.triangle.branch")
                            .font(.system(size: 10))
                            .foregroundStyle(branch.isLeaf ? Color.green : Color.secondary)
                        Text(branch.name.isEmpty ? "Node\(branch.id)" : branch.name)
                            .font(.system(size: 11, weight: branch.isLeaf ? .medium : .regular, design: .monospaced))
                    }
                }
                .width(min: 140, ideal: 180)

                TableColumn("Type") { branch in
                    Text(branch.isLeaf ? "Leaf Taxon" : "Internal Node")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
                .width(100)

                TableColumn("Branch Length") { branch in
                    Text(String(format: "%.5f", branch.length))
                        .font(.system(size: 10, design: .monospaced))
                }
                .width(90)

                TableColumn("Testing Status") { branch in
                    HStack {
                        Toggle(isOn: Binding(
                            get: { branch.isTested },
                            set: { newValue in
                                if let idx = model.branches.firstIndex(where: { $0.id == branch.id }) {
                                    model.branches[idx].isTested = newValue
                                    onApply?()
                                }
                            }
                        )) {
                            Text(branch.isTested ? "Tested (Foreground)" : "Background")
                                .font(.system(size: 10, weight: .semibold))
                                .foregroundStyle(branch.isTested ? Color.orange : Color.secondary)
                        }
                        .toggleStyle(.checkbox)
                    }
                }
                .width(min: 150, ideal: 170)
            }
            .tableStyle(.bordered)
        }
    }
}
