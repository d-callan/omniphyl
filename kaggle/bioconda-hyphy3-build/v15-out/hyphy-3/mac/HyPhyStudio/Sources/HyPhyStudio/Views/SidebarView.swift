import SwiftUI

public struct SidebarView: View {
    @Binding public var datasets: [DatasetModel]
    @Binding public var selectedDatasetID: UUID?
    @Binding public var selectedRunID: UUID?
    public var onAddDataset: (() -> Void)?
    public var onImportSample: (() -> Void)?

    @State private var searchText = ""

    public init(
        datasets: Binding<[DatasetModel]>,
        selectedDatasetID: Binding<UUID?>,
        selectedRunID: Binding<UUID?>,
        onAddDataset: (() -> Void)? = nil,
        onImportSample: (() -> Void)? = nil
    ) {
        self._datasets = datasets
        self._selectedDatasetID = selectedDatasetID
        self._selectedRunID = selectedRunID
        self.onAddDataset = onAddDataset
        self.onImportSample = onImportSample
    }

    public var body: some View {
        List(selection: $selectedDatasetID) {
            Section("Datasets") {
                ForEach(filteredDatasets) { dataset in
                    DisclosureGroup(isExpanded: .constant(true)) {
                        // Analysis Runs under this dataset
                        ForEach(dataset.runs) { run in
                            HStack(spacing: 6) {
                                Image(systemName: runIcon(run.method))
                                    .foregroundStyle(runColor(run.method))
                                    .font(.system(size: 10))

                                Text(run.method.uppercased())
                                    .font(.system(size: 11, weight: .semibold, design: .monospaced))

                                Spacer()

                                Text(String(format: "%.2fs", run.durationSeconds))
                                    .font(.system(size: 9, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.vertical, 2)
                            .tag(dataset.id)
                            .onTapGesture {
                                selectedDatasetID = dataset.id
                                selectedRunID = run.id
                            }
                        }
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: "dna")
                                .foregroundStyle(Color.orange)

                            VStack(alignment: .leading, spacing: 2) {
                                Text(dataset.name)
                                    .font(.system(size: 12, weight: .medium))
                                Text("\(dataset.taxaCount) taxa • \(dataset.codonSites) codons")
                                    .font(.system(size: 10))
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .tag(dataset.id)
                        .onTapGesture {
                            selectedDatasetID = dataset.id
                            selectedRunID = nil
                        }
                    }
                }
            }
        }
        .searchable(text: $searchText, prompt: "Search datasets, genes, tags...")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Menu {
                    Button("Load Local Alignment + Tree...", action: { onAddDataset?() })
                    Divider()
                    Button("Load Sample: CD2 Primate Receptor", action: { onImportSample?() })
                } label: {
                    Label("Add Dataset", systemImage: "plus")
                }
            }
        }
    }

    private var filteredDatasets: [DatasetModel] {
        if searchText.isEmpty { return datasets }
        return datasets.filter {
            $0.name.localizedCaseInsensitiveContains(searchText) ||
            $0.tags.contains(where: { $0.localizedCaseInsensitiveContains(searchText) })
        }
    }

    private func runIcon(_ method: String) -> String {
        switch method.lowercased() {
        case "meme": return "triangle.fill"
        case "fel": return "circle.fill"
        case "busted": return "sparkles"
        default: return "chart.bar.fill"
        }
    }

    private func runColor(_ method: String) -> Color {
        switch method.lowercased() {
        case "meme": return .orange
        case "fel": return .blue
        case "busted": return .purple
        default: return .secondary
        }
    }
}
