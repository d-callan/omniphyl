import SwiftUI
import AppKit

public struct InspectorView: View {
    public let dataset: DatasetModel?
    public let run: AnalysisRunModel?

    public init(dataset: DatasetModel?, run: AnalysisRunModel?) {
        self.dataset = dataset
        self.run = run
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let r = run, let ds = dataset {
                // Run Details
                VStack(alignment: .leading, spacing: 6) {
                    Text("ACTIVE ANALYSIS RUN")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.secondary)
                        .tracking(1.0)

                    HStack {
                        Text(r.method.uppercased())
                            .font(.system(size: 16, weight: .bold, design: .monospaced))
                            .foregroundStyle(r.method == "meme" ? Color.orange : Color.blue)
                        Spacer()
                        Text("COMPLETED")
                            .font(.system(size: 9, weight: .bold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.green.opacity(0.12))
                            .foregroundStyle(Color.green)
                            .clipShape(Capsule())
                    }

                    Text("Dataset: \(ds.name)")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                    Text("Duration: \(String(format: "%.2f", r.durationSeconds))s • Cutoff: p ≤ \(String(format: "%.2f", r.pValueThreshold))")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }

                Divider()

                // Export Tools
                VStack(alignment: .leading, spacing: 8) {
                    Text("PUBLICATION & EXPORT")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.secondary)
                        .tracking(1.0)

                    Button(action: exportHTMLReport) {
                        Label("Export Standalone HTML Report...", systemImage: "arrow.down.doc")
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.bordered)

                    Button(action: exportJSON) {
                        Label("Export Modern v3.0 JSON...", systemImage: "curlybraces")
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.bordered)

                    Button(action: copyJSONToClipboard) {
                        Label("Copy JSON to Clipboard", systemImage: "doc.on.doc")
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.bordered)
                }
            } else if let ds = dataset {
                // Dataset Summary
                VStack(alignment: .leading, spacing: 8) {
                    Text("DATASET PROPERTIES")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.secondary)
                        .tracking(1.0)

                    Text(ds.name)
                        .font(.headline)

                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                        GridRow {
                            Text("Taxa Count:").foregroundStyle(.secondary).font(.caption)
                            Text("\(ds.taxaCount)").font(.system(size: 11, weight: .semibold, design: .monospaced))
                        }
                        GridRow {
                            Text("Codon Sites:").foregroundStyle(.secondary).font(.caption)
                            Text("\(ds.codonSites)").font(.system(size: 11, weight: .semibold, design: .monospaced))
                        }
                        GridRow {
                            Text("Genetic Code:").foregroundStyle(.secondary).font(.caption)
                            Text(ds.geneticCode).font(.system(size: 11, weight: .semibold, design: .monospaced))
                        }
                        GridRow {
                            Text("Past Runs:").foregroundStyle(.secondary).font(.caption)
                            Text("\(ds.runs.count)").font(.system(size: 11, weight: .semibold, design: .monospaced))
                        }
                    }
                }
            } else {
                ContentUnavailableView("No Selection", systemImage: "slider.horizontal.3", description: Text("Select a dataset or analysis run to view details."))
            }

            Spacer()
        }
        .padding(14)
        .frame(minWidth: 220, idealWidth: 250)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private func exportHTMLReport() {
        guard let r = run else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.html]
        panel.nameFieldStringValue = "\(r.method)_\(dataset?.name ?? "report").html"
        if panel.runModal() == .OK, let url = panel.url {
            // Write report HTML
            try? r.modernJSON.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    private func exportJSON() {
        guard let r = run else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "\(r.method)_\(dataset?.name ?? "result").v3.json"
        if panel.runModal() == .OK, let url = panel.url {
            try? r.modernJSON.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    private func copyJSONToClipboard() {
        guard let r = run else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(r.modernJSON, forType: .string)
    }
}
