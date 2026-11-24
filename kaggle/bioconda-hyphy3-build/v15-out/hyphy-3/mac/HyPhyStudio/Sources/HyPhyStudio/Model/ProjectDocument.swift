import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    public static var hyphyProject: UTType {
        UTType(exportedAs: "org.veg.hyphyproj", conformingTo: .package)
    }
}

public struct HyPhyProjectDocument: FileDocument {
    public static var readableContentTypes: [UTType] {
        var types: [UTType] = [.hyphyProject, .folder]
        if let nex = UTType(filenameExtension: "nex") { types.append(nex) }
        if let nexus = UTType(filenameExtension: "nexus") { types.append(nexus) }
        if let fas = UTType(filenameExtension: "fas") { types.append(fas) }
        if let fasta = UTType(filenameExtension: "fasta") { types.append(fasta) }
        if let fna = UTType(filenameExtension: "fna") { types.append(fna) }
        types.append(.plainText)
        return types
    }

    public var title: String
    public var datasets: [DatasetModel]

    public init(title: String = "Untitled Experiment", datasets: [DatasetModel] = []) {
        self.title = title
        self.datasets = datasets
    }

    public init(configuration: ReadConfiguration) throws {
        let wrapper = configuration.file
        if wrapper.isDirectory, let fileWrappers = wrapper.fileWrappers {
            if let projData = fileWrappers["project.json"]?.regularFileContents,
               let projectMeta = try? JSONDecoder().decode(ProjectMetadata.self, from: projData) {
                self.title = projectMeta.title
                self.datasets = projectMeta.datasets
            } else {
                self.title = "Imported Project"
                self.datasets = []
            }
        } else if let regularData = wrapper.regularFileContents,
                  let fileContent = String(data: regularData, encoding: .utf8) {
            let filename = configuration.file.filename ?? "Imported Alignment"
            let titleName = (filename as NSString).deletingPathExtension
            self.title = titleName
            let ds = DatasetModel(name: titleName, rawFasta: fileContent)
            self.datasets = [ds]
        } else {
            throw CocoaError(.fileReadCorruptFile)
        }
    }

    public func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        var rootWrappers: [String: FileWrapper] = [:]

        // 1. Serialize project.json metadata
        let meta = ProjectMetadata(title: self.title, datasets: self.datasets)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let metaData = try encoder.encode(meta)
        rootWrappers["project.json"] = FileWrapper(regularFileWithContents: metaData)

        // 2. Datasets folder
        var datasetWrappers: [String: FileWrapper] = [:]
        for ds in datasets {
            var dsFolderWrappers: [String: FileWrapper] = [:]
            if let fnaData = ds.rawFasta.data(using: .utf8) {
                dsFolderWrappers["alignment.fna"] = FileWrapper(regularFileWithContents: fnaData)
            }
            if let nwkData = ds.treeNewick.data(using: .utf8) {
                dsFolderWrappers["tree.nwk"] = FileWrapper(regularFileWithContents: nwkData)
            }
            // Runs folder
            var runsWrappers: [String: FileWrapper] = [:]
            for run in ds.runs {
                if let jsonData = run.modernJSON.data(using: .utf8) {
                    runsWrappers["run_\(run.method)_\(run.id.uuidString.prefix(8)).json"] = FileWrapper(regularFileWithContents: jsonData)
                }
            }
            dsFolderWrappers["runs"] = FileWrapper(directoryWithFileWrappers: runsWrappers)
            datasetWrappers[ds.id.uuidString] = FileWrapper(directoryWithFileWrappers: dsFolderWrappers)
        }
        rootWrappers["datasets"] = FileWrapper(directoryWithFileWrappers: datasetWrappers)

        return FileWrapper(directoryWithFileWrappers: rootWrappers)
    }

    private struct ProjectMetadata: Codable, Sendable {
        var title: String
        var datasets: [DatasetModel]
    }
}
