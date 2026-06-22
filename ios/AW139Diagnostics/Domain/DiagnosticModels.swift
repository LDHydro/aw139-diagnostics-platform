import Foundation

/// The inputs a mechanic fills in — same fields as the web `DiagnosticForm`.
struct DiagnosticRequest {
    var aircraftModel: String = "AW139"
    var serialNumber: String = ""
    var ata: String = ""
    var taskType: TaskType = .faultIsolation
    var problemDescription: String = ""

    /// The natural-language query handed to the retrieval + prompt pipeline.
    /// Combines the ATA system context with the free-text problem description,
    /// matching how the web client phrases queries to the RAG service.
    var query: String {
        let system = ATACatalog.displayName(for: ata)
        if system.isEmpty || ata.isEmpty {
            return problemDescription
        }
        return "[\(system)] \(problemDescription)"
    }
}

/// One document stored in the offline corpus (produced by tools/build_offline_corpus.py).
/// Mirrors the structure consumed by rag_api.py: `doc_path`, `text`, `embedding`.
struct CorpusDocument: Codable, Identifiable {
    let id: String
    let docPath: String
    let ata: String?
    let text: String
    let embedding: [Float]

    enum CodingKeys: String, CodingKey {
        case id
        case docPath = "doc_path"
        case ata
        case text
        case embedding
    }
}

/// A retrieved document plus its similarity score, ready for prompt assembly.
struct RetrievedDocument: Identifiable {
    let document: CorpusDocument
    let similarity: Float
    var id: String { document.id }

    var dmcCode: String { DMC.extractCode(from: document.docPath) }
    var ataIdentifier: String { DMC.formatATAIdentifier(from: document.docPath) }
}

/// The query categories rag_api.py routes between to pick a system prompt.
enum QueryKind {
    case faultCode
    case calibration
    case procedure
    case electrical
    case general
}

/// Final result of a diagnostic run.
struct DiagnosticResult {
    var kind: QueryKind
    var answer: String
    var retrieved: [RetrievedDocument]
    var dmcList: String
}
