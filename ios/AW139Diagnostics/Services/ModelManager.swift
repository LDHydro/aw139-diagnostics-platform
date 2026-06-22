import Foundation

/// Owns the on-device models and tracks their load state for the UI.
///
/// Until the LLM is loaded, generation falls back to `StubGenerator` so the app
/// still runs (showing retrieval results). The embedder is required for
/// retrieval; if the embedding model isn't present, embedding calls throw a
/// clear "transfer the model" error surfaced in the UI.
@MainActor
final class ModelManager: ObservableObject {

    enum LoadState: Equatable {
        case idle
        case loading(Double)   // 0...1
        case ready
        case failed(String)
    }

    @Published var llmState: LoadState = .idle
    @Published var embedderState: LoadState = .idle

    /// Optional Hugging Face fallback IDs, used only if no file-transferred model
    /// is present and the device has internet for a one-time download.
    /// Set to nil to enforce strictly offline operation.
    var llmFallbackHubID: String? = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    private(set) var embedder: Embedder = MLXEmbedder()
    private var mlxGenerator: TextGenerator?
    private let stub = StubGenerator()

    /// The generator used by the pipeline: the real MLX model once loaded,
    /// otherwise the stub.
    var activeGenerator: TextGenerator { mlxGenerator ?? stub }

    var llmReady: Bool { if case .ready = llmState { return true } else { return false } }
    var embedderReady: Bool { if case .ready = embedderState { return true } else { return false } }

    func loadEmbedder() async {
        embedderState = .loading(0)
        do {
            try await embedder.load { [weak self] p in self?.embedderState = .loading(p) }
            embedderState = .ready
        } catch {
            embedderState = .failed(error.localizedDescription)
        }
    }

    func loadLLM() async {
        llmState = .loading(0)
        let gen = MLXGenerator(fallbackHubID: llmFallbackHubID)
        do {
            try await gen.load { [weak self] p in self?.llmState = .loading(p) }
            mlxGenerator = gen
            llmState = .ready
        } catch {
            mlxGenerator = nil
            llmState = .failed(error.localizedDescription)
        }
    }

    /// Load everything needed for a fully offline run.
    func loadAll() async {
        AppPaths.ensureDirectories()
        await loadEmbedder()
        await loadLLM()
    }
}
