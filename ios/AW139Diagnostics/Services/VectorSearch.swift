import Foundation
import Accelerate

/// Cosine-similarity retrieval over the in-memory corpus.
/// Equivalent to rag_api.py `cosine_similarity` + top-k selection, but uses
/// the Accelerate framework so scoring tens of thousands of vectors stays fast
/// on the iPad.
enum VectorSearch {

    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard !a.isEmpty, a.count == b.count else { return 0 }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        vDSP_svesq(a, 1, &normA, vDSP_Length(a.count))
        vDSP_svesq(b, 1, &normB, vDSP_Length(b.count))
        let denom = sqrt(normA) * sqrt(normB)
        return denom == 0 ? 0 : dot / denom
    }

    /// Return the top-k documents most similar to the query embedding.
    static func topK(_ k: Int,
                     queryEmbedding: [Float],
                     documents: [CorpusDocument]) -> [RetrievedDocument] {
        let scored = documents.map { doc in
            RetrievedDocument(document: doc,
                              similarity: cosineSimilarity(queryEmbedding, doc.embedding))
        }
        return Array(scored.sorted { $0.similarity > $1.similarity }.prefix(k))
    }
}
