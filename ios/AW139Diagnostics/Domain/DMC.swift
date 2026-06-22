import Foundation

/// DMC (Data Module Code) helpers, ported from rag_api.py so citations in the
/// offline app match the server behaviour.
enum DMC {

    /// Matches the AW139 DMC pattern: XX-X-XX-XX-XX-XXA-XXXA-X
    private static let dmcRegex = try! NSRegularExpression(
        pattern: "(\\d{2}-[A-Z]-\\d{2}-\\d{2}-\\d{2}-\\d{2}[A-Z]-\\d{3}[A-Z]?-[A-Z])"
    )

    /// Extract a full DMC code from a document path for citation.
    /// e.g. "dmc-39-A-24-32-00-00A-320A-A.xml" -> "39-A-24-32-00-00A-320A-A"
    static func extractCode(from docPath: String) -> String {
        let filename = (docPath as NSString).lastPathComponent
        let cleaned = filename
            .replacingOccurrences(of: "dmc-", with: "")
            .replacingOccurrences(of: ".xml", with: "")
            .replacingOccurrences(of: ".pdf", with: "")
            .replacingOccurrences(of: ".txt", with: "")

        let range = NSRange(cleaned.startIndex..., in: cleaned)
        if let match = dmcRegex.firstMatch(in: cleaned, range: range),
           let r = Range(match.range(at: 1), in: cleaned) {
            return String(cleaned[r])
        }
        // Fallback: return the cleaned filename if it looks like a DMC.
        if cleaned.count > 10 && cleaned.contains("-") {
            return cleaned
        }
        return ""
    }

    /// Extract a short ATA identifier from a document path.
    static func formatATAIdentifier(from docPath: String) -> String {
        let filename = (docPath as NSString).lastPathComponent
        return filename
            .replacingOccurrences(of: "dmc-", with: "")
            .replacingOccurrences(of: ".xml", with: "")
            .replacingOccurrences(of: ".pdf", with: "")
    }

    /// Build the comma-separated DMC reference list shown to the model,
    /// using the top retrieved documents (matches rag_api.py `dmc_list`).
    static func referenceList(for docs: [RetrievedDocument]) -> String {
        let codes = docs.prefix(5)
            .map { extractCode(from: $0.document.docPath) }
            .filter { !$0.isEmpty }
        return codes.isEmpty ? "No specific DMC identified" : codes.joined(separator: ", ")
    }
}
