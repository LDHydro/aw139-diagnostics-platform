import Foundation

/// Loads and holds the offline document corpus produced by
/// tools/build_offline_corpus.py. The corpus is a JSON array of
/// CorpusDocument records (doc_path, text, embedding).
@MainActor
final class CorpusStore: ObservableObject {

    @Published private(set) var documents: [CorpusDocument] = []
    @Published private(set) var isLoaded = false
    @Published private(set) var loadError: String?

    var count: Int { documents.count }

    /// Embedding dimensionality of the corpus (used to sanity-check the query
    /// embedder produces matching-size vectors).
    var embeddingDimension: Int { documents.first?.embedding.count ?? 0 }

    func loadIfNeeded() {
        guard !isLoaded else { return }
        load()
    }

    func load() {
        let url = AppPaths.corpusFile
        guard AppPaths.exists(url) else {
            self.documents = []
            self.isLoaded = false
            self.loadError = "corpus.json not found. Transfer it into the app's Documents folder."
            return
        }
        do {
            let data = try Data(contentsOf: url)
            let decoded = try JSONDecoder().decode([CorpusDocument].self, from: data)
            self.documents = decoded
            self.isLoaded = true
            self.loadError = nil
        } catch {
            self.documents = []
            self.isLoaded = false
            self.loadError = "Failed to parse corpus.json: \(error.localizedDescription)"
        }
    }

    func search(queryEmbedding: [Float], topK: Int = 5) -> [RetrievedDocument] {
        VectorSearch.topK(topK, queryEmbedding: queryEmbedding, documents: documents)
    }
}
