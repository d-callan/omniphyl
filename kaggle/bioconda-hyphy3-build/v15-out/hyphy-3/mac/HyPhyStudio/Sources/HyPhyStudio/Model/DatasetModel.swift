import Foundation

public struct DatasetModel: Identifiable, Codable, Sendable {
    public var id: UUID
    public var name: String
    public var geneticCode: String
    public var taxaCount: Int
    public var codonSites: Int
    public var rawFasta: String
    public var treeNewick: String
    public var createdAt: Date
    public var modifiedAt: Date
    public var notes: String
    public var tags: [String]
    public var runs: [AnalysisRunModel]

    public init(
        id: UUID = UUID(),
        name: String,
        rawFasta: String = "",
        treeNewick: String = "",
        geneticCode: String = "Universal",
        taxaCount: Int = 0,
        codonSites: Int = 0,
        notes: String = "",
        tags: [String] = [],
        runs: [AnalysisRunModel] = []
    ) {
        self.id = id
        self.name = name
        self.rawFasta = rawFasta
        self.treeNewick = treeNewick
        self.geneticCode = geneticCode
        self.taxaCount = taxaCount
        self.codonSites = codonSites
        self.createdAt = Date()
        self.modifiedAt = Date()
        self.notes = notes
        self.tags = tags
        self.runs = runs
    }
}

public struct AnalysisRunModel: Identifiable, Codable, Sendable {
    public var id: UUID
    public var method: String // "fel", "meme", "busted"
    public var status: String // "running", "completed", "failed"
    public var pValueThreshold: Double
    public var logLikelihood: Double?
    public var globalOmega: Double?
    public var significantSitesCount: Int
    public var modernJSON: String // Modern v3 JSON string
    public var createdAt: Date
    public var durationSeconds: Double

    public init(
        id: UUID = UUID(),
        method: String,
        pValueThreshold: Double = 0.1,
        modernJSON: String = "",
        significantSitesCount: Int = 0,
        logLikelihood: Double? = nil,
        globalOmega: Double? = nil,
        durationSeconds: Double = 0.0,
        status: String = "completed"
    ) {
        self.id = id
        self.method = method
        self.pValueThreshold = pValueThreshold
        self.modernJSON = modernJSON
        self.significantSitesCount = significantSitesCount
        self.logLikelihood = logLikelihood
        self.globalOmega = globalOmega
        self.durationSeconds = durationSeconds
        self.status = status
        self.createdAt = Date()
    }
}
