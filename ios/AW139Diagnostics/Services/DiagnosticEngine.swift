import Foundation

/// Runs the full offline pipeline, mirroring the server flow:
///   query → embed → cosine top-k retrieval → classify → build prompts → generate
struct DiagnosticEngine {
    let corpus: CorpusStore
    let embedder: Embedder
    let generator: TextGenerator

    /// Executes a diagnostic. `onToken` streams the cumulative generated answer.
    @MainActor
    func run(_ request: DiagnosticRequest,
             onToken: @escaping (String) -> Void) async throws -> DiagnosticResult {

        let query = request.query

        // 1. Embed the query on-device.
        let queryEmbedding = try await embedder.embed(query)

        // 2. Guard against an embedder/corpus mismatch.
        let dim = corpus.embeddingDimension
        if dim > 0 && queryEmbedding.count != dim {
            throw InferenceError.dimensionMismatch(expected: dim, got: queryEmbedding.count)
        }

        // 3. Retrieve top-5 documents by cosine similarity.
        let retrieved = corpus.search(queryEmbedding: queryEmbedding, topK: 5)

        // 4. Classify + build prompts (port of rag_api.py).
        let kind = PromptBuilder.classify(query)
        let dmcList = DMC.referenceList(for: retrieved)
        let system = PromptBuilder.systemPrompt(kind: kind, dmcList: dmcList)
        let user = PromptBuilder.userPrompt(query: query, retrieved: retrieved, dmcList: dmcList)

        // 5. Generate the answer on-device.
        let answer = try await generator.generate(system: system, user: user, onToken: onToken)

        return DiagnosticResult(kind: kind, answer: answer, retrieved: retrieved, dmcList: dmcList)
    }
}
