import SwiftUI

public struct RunSheetView: View {
    @Environment(\.dismiss) private var dismiss
    public let dataset: DatasetModel
    public let treeModel: TreeStagingModel
    public var onComplete: ((AnalysisRunModel) -> Void)?

    @State private var selectedMethod: String = "meme"
    @State private var pValueThreshold: Double = 0.10
    @State private var isRunning = false
    @State private var statusMessage = ""
    @State private var errorMessage: String?

    public init(
        dataset: DatasetModel,
        treeModel: TreeStagingModel,
        onComplete: ((AnalysisRunModel) -> Void)? = nil
    ) {
        self.dataset = dataset
        self.treeModel = treeModel
        self.onComplete = onComplete
    }

    public var body: some View {
        VStack(spacing: 16) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Launch In-Process HyPhy Analysis")
                        .font(.headline)
                    Text("Dataset: \(dataset.name) (\(dataset.taxaCount) taxa, \(dataset.codonSites) codons)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .disabled(isRunning)
            }

            Divider()

            // Method Picker
            VStack(alignment: .leading, spacing: 8) {
                Text("Evolutionary Hypothesis / Method:")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundStyle(.secondary)

                Picker("", selection: $selectedMethod) {
                    Text("MEME — Episodic Diversifying Selection (Site & Branch)").tag("meme")
                    Text("FEL — Pervasive Diversifying & Purifying Selection (Site-Level)").tag("fel")
                    Text("BUSTED — Gene-wide Episodic Selection (Whole-Gene)").tag("busted")
                }
                .pickerStyle(.radioGroup)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            // Method Description Card
            HStack(spacing: 8) {
                Image(systemName: "info.circle")
                    .foregroundStyle(Color.orange)
                Text(methodDescription)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
            .padding(10)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))

            // Parameters
            HStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("p-value Significance Threshold:")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundStyle(.secondary)

                    Picker("", selection: $pValueThreshold) {
                        Text("p ≤ 0.10 (Standard exploratory)").tag(0.10)
                        Text("p ≤ 0.05 (Rigorous nominal)").tag(0.05)
                        Text("p ≤ 0.01 (Conservative)").tag(0.01)
                    }
                    .pickerStyle(.segmented)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("Active Branch Partition:")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundStyle(.secondary)

                    Text("\(treeModel.testedCount) Tested (Foreground), \(treeModel.backgroundCount) Background")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.primary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if let err = errorMessage {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(Color.red)
            }

            Spacer()

            // Run Button & Progress
            HStack {
                if isRunning {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text(statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()

                Button(action: executeAnalysis) {
                    HStack(spacing: 6) {
                        Image(systemName: "play.fill")
                        Text(isRunning ? "Computing..." : "Run Analysis (In-Process)")
                    }
                    .padding(.horizontal, 8)
                }
                .buttonStyle(.borderedProminent)
                .tint(Color.orange)
                .disabled(isRunning)
            }
        }
        .padding(20)
        .frame(width: 580, height: 380)
    }

    private var methodDescription: String {
        switch selectedMethod {
        case "meme":
            return "MEME infers episodic diversifying selection by fitting a two-rate mixture model (β1, β+) and computes branch-level Empirical Bayes Factors (EBF) to identify lineages subject to selection."
        case "fel":
            return "FEL uses maximum likelihood to estimate site-wise synonymous (α) and non-synonymous (β) rates under pervasive selection across the entire phylogeny."
        case "busted":
            return "BUSTED tests whether a gene has experienced episodic diversifying selection on at least one site on tested branches."
        default:
            return ""
        }
    }

    private func executeAnalysis() {
        isRunning = true
        statusMessage = "Initializing In-Process C++ Engine..."
        errorMessage = nil

        let engine = HyPhyEngine()
        let fasta = dataset.rawFasta
        let newick = dataset.treeNewick
        let method = selectedMethod
        let pval = pValueThreshold
        let branches = treeModel.branches

        Task {
            let startTime = Date()
            let (taxa, codons, _) = await engine.loadDataset(fasta: fasta, newick: newick)
            guard taxa > 0 && codons > 0 else {
                await MainActor.run {
                    isRunning = false
                    errorMessage = "Failed to load alignment or tree into HyPhy engine."
                }
                return
            }

            await engine.applyBranchTesting(branches)

            await MainActor.run {
                statusMessage = "Optimizing likelihood on Apple Silicon cores..."
            }

            do {
                let jsonResult: String
                switch method {
                case "meme":
                    jsonResult = try await engine.runMEME(pvalThreshold: pval)
                case "busted":
                    jsonResult = try await engine.runBUSTED()
                default:
                    jsonResult = try await engine.runFEL(pvalThreshold: pval)
                }

                let duration = Date().timeIntervalSince(startTime)
                let runModel = AnalysisRunModel(
                    method: method,
                    pValueThreshold: pval,
                    modernJSON: jsonResult,
                    significantSitesCount: 0,
                    durationSeconds: duration,
                    status: "completed"
                )

                await MainActor.run {
                    isRunning = false
                    onComplete?(runModel)
                    dismiss()
                }
            } catch {
                await MainActor.run {
                    isRunning = false
                    errorMessage = "Execution failed: \(error.localizedDescription)"
                }
            }
        }
    }
}
